---
layout: post
title: A Concrete Tutorial of Conditional Random Fields in PyTorch
tags: [graphical-models, conditional-random-fields, iterative-method]
date: 2019-04-09 17:27:00 +0800
---

> Solving an optimization problem with **iterative methodology** often have five ingredient:
> - **Modeling** The problem with a set of *unknowns* or *parameters* to predict a datum's label.
> - **Initialize** A prediction of the datum's output by some mechanism.
> - **Forward propagate** the datum through the computational graph to get its *projections* or *features*.
> - Compute the *residual* of the *projections* between the real label and its prediction by the **loss function**.
> - **Backward propagate** the *residual* to the previous *parameters*, in other words, update the *parameters*.

In this tutorial, we use the **conditional random fields (CRFs)** to model the named-entity recognition (NER) problem. The **parameters** in this CRF is the transition probability between the tags and emission features provided by a `Bi-LSTM` neural network. The **loss function** in CRFs is the negative logarithm of conditional probability $p\left(\mathbf{y}\mid\mathbf{x}\right)$. We **forward pass** the datum through the computational graph to obtain its features, compare with its labels to get the parameters' **residual**, all left is projecting the residual **backward**, here we use gradient descent method. With the help of [pytorch](https://pytorch.org)'s `autograd` packages, we only need to implement the calculation of loss function, then pytorch's computational graph mechanism help us automatic compute the gradient of the loss. Remark here, one of the difficulty in computing the loss function is the calculation of *partition function*, we'll give a detail instruction how partition function is calculated in the log-space.

*Note:* this tutorial is based on Guthrie's[^1].


## Mathematical overview of Conditional Random Fields

**A brief note on notation:** Assume that we have some sequence length $m$, and some set of possible tags $\mathcal{Y}$. Let $\mathbf{y}$ be a tag sequence $y_1\ldots y_m$, and $\mathbf{x}$ an input sequence of words $x_1\ldots x_m$. For any tag sequence $y_1\ldots y_m$ where each $y_i\in\mathcal{Y}$, we define the *potential* for the sequence as[^2]

$$\begin{equation*}
    \psi(y_1\ldots y_m) = \prod^m_{i=1} \psi(y_{i-1},y_i,i),
\end{equation*}$$

where $\psi(y^\prime,y,i)\geq 0$ for $y^\prime, y\in\mathcal{S}$, $i\in\{1\ldots m\}$ is a potential function, which returns a value for the tag transition $y^\prime$ to $y$ at position $i$ in the sequence.

The potential function $\psi(y_{i-1},y_i,i)$ might be defined in various ways. As one example, consider an **Hidden Markov Model (HMM)** applied to an input sentence $y_1\ldots y_m$. If we define

$$\begin{equation*}
    \psi\left(y^\prime,y,i\right) = \psi_\text{EMIT}(x_i \mid y) \cdot \psi_\text{TRANS}(y\mid y^\prime),
\end{equation*}$$

then,

$$\begin{align*}
    \psi(y_1\ldots y_m) &= \prod_{i=1}^m \psi(y_{i-1},y_i,i), \\
    &= \prod_{i=1}^m \psi_\text{EMIT}(x_i \mid y_i) \cdot \psi_\text{TRANS}(y_i\mid y_{i-1}).
\end{align*}$$

Recall that the **Conditional Random Fields** computes a conditional probability:

$$\begin{equation}
    p\left(\mathbf{y}\mid\mathbf{x}\right) = \frac{\exp\left(\text{Score}\left(\mathbf{y}\mid\mathbf{x}\right)\right)}{\sum_{\mathbf{y^\prime}} \exp\left(\text{Score}\left(\mathbf{y^\prime}\mid\mathbf{x}\right)\right)},\label{linear-chain-crf}
\end{equation}$$

where the score is determined by defining some log potentials $\log \psi_k(y_1\ldots y_m)$ such that

$$\text{Score}\left(\mathbf{y}\mid\mathbf{x}\right) = \sum_k \log \psi_k(y_1\ldots y_m).$$

The *marginal probability* in denominator, also called the *partition function*:

$$\begin{equation}
    Z\left(\mathbf{x}\right) = \sum_{\mathbf{y^\prime}} \exp\left(\text{Score}\left(\mathbf{y^\prime}\mid\mathbf{x}\right)\right).\label{eq:partition-function}
\end{equation}$$

To make the partition function tractable, the potentials must look only at local features.

**Relations between HMMs and CRFs** Whereas CRFs throw any bunch of functions together to get a tag score, HMMs take a *generative* approach to tagging relying on the independent assumption. Note that the log of the HMMs probability is

$$\begin{align}
    \log \psi(y_1\ldots y_m) &= \log\left\{\prod_i \psi_\text{EMIT}(x_i \mid y_i) \cdot \psi_\text{TRANS}(y_i\mid y_{i-1})\right\}, \\
    &= \sum_i\left\{\log \psi_\text{EMIT}(x_i \mid y_i) + \log \psi_\text{TRANS}(y_i\mid y_{i-1})\right\}.\label{eq:relation}
\end{align}$$

This has exactly the log-linear form of a CRF if we consider these log-probabilities to be the weights associated to transition and emission indicator features.[^3]


## Model's Parameters

Focus on **Bi-LSTM CRF**[^1] model, we define two kinds of potentials: *emission* and *transition*. The *emission* potential for the word at index $i$ comes from the hidden state of the `Bi-LSTM` at time-step $i$. The *transition* scores are stored in a $\textbf{P}$ parameters matrix with dimension $\vert\mathcal{Y}\vert\times\vert\mathcal{Y}\vert$. In my implementation, $\textbf{P}_{j,k}$ is the score of transitioning to tag $j$ from tag $k$. So,

$$\begin{align}
    \text{Score}\left(\mathbf{y}\mid\mathbf{x}\right) &= \sum_i\left\{\log \psi_\text{EMIT}(x_i \mid y_i) + \log \psi_\text{TRANS}(y_i\mid y_{i-1})\right\},\label{eq:bi-lstm-crf}\\
    &= \sum_i h_i[y_i] + \textbf{P}_{y_i, y_{i-1}},
\end{align}$$

where in this last expression, we think of the tags as being assigned unique non-negative indices.

*Remark,* combine the equation $\eqref{eq:relation}$ and $\eqref{eq:bi-lstm-crf}$, we get:

$$\begin{equation*}
    \text{Score}\left(\mathbf{y}\mid\mathbf{x}\right) = \log \psi(y_1\ldots y_m).
\end{equation*}$$


## Parameter Estimation

We need to find the $\textbf{P}$ parameters that best fit the training data, a given set of tagged sentences:

$$\begin{equation*}
    \left\{\left(\mathbf{x}^1,\mathbf{y}^1\right),\ldots,\left(\mathbf{x}^n,\mathbf{y}^n\right)\right\},
\end{equation*}$$

where each pair $\left(\mathbf{x}^k,\mathbf{y}^k\right)$ is a sentences with the corresponding tags annotated. To find the $\textbf{P}$ parameters that best fitting the training data we need to minimize the **Negative Log Likelihood Loss**:

$$\begin{equation}
    \mathbb{L}\left(\textbf{P}\right) = - \sum_{k=1}^n \log p\left(\mathbf{y}^k\mid\mathbf{x}^k, \textbf{P}\right) \label{eq:nllloss},
\end{equation}$$

where (omit the superscript $k$ and $\textbf{P}$ parameters),

$$\begin{align}
    - \log p\left(\mathbf{y}\mid \mathbf{x}\right) &= \log \sum_{\mathbf{y^\prime}} \exp\left\{\text{Score}\left(\mathbf{y^\prime}\mid\mathbf{x}\right)\right\} - \text{Score}\left(\mathbf{y}\mid\mathbf{x}\right),\\
    &= \underset{\mathbf{y^\prime}}{\operatorname{log\,sum\,exp}}\left\{\text{Score}\left(\mathbf{y^\prime}\mid\mathbf{x}\right)\right\} - \text{Score}\left(\mathbf{y}\mid\mathbf{x}\right). \label{eq:log-partition}
\end{align}$$

Pytorch's `autograd` packages provides automatic differentiation for all operations on `Tensors`, it allows you to not have to write the back propagation gradients yourself. Here we only need to implement the forward part of computing the `NLLLoss` equation $\eqref{eq:nllloss}$, then we use the set of tagged sentences to learn the transition parameters matrix.

Notice how in the left of minus of equation $\eqref{eq:log-partition}$, we're computing the sum over all possible sequences of tags $\vert y^\prime\vert = \vert\mathcal{Y}\vert^m$, a very large set. We observe that it is the partition function in the logarithmic space, and it can be computed efficiently using the **forward-backward algorithm**[^4]. Define a set of *forward variables* $\bar{\alpha_i}, i\in \{1\ldots m\}$, each of which is a vector of size $\vert\mathcal{Y}\vert$ which represents the intermediate sums. These are defined as:

$$\begin{equation}
    \alpha_i(y) \equiv p\left(y_i = y\mid x_1\ldots x_i\right), \label{eq:forward-variable}
\end{equation}$$

with initialization, $\alpha_1(y) = \psi\left(\text{START},y,1\right)$. We can also write $\alpha$ as:

$$\begin{equation}
    \alpha_i(y) = \sum_{y_1^\prime\ldots y_{i-1}^\prime} p\left(y_1^\prime\ldots y_{i-1}^\prime\mid x_1\ldots x_{i-1}\right)\cdot\psi\left(y_{i-1}^\prime,y,i\right), \label{eq:forward-iter}
\end{equation}$$

where the summation over $y_1^\prime\ldots y_{i-1}^\prime$ ranges over all assignments to the sequence of random variables $y_1\ldots y_{i-1}$. The probability $p\left(y_1^\prime\ldots y_{i-1}^\prime\mid x_1\ldots x_{i-1}\right)$ in equation $\eqref{eq:forward-iter}$ is determined by the last tag, so equation $\eqref{eq:forward-variable}$ can be computed by the recursion:

$$\begin{align*}
    \alpha_i(y) &= \sum_{y^\prime\in\mathcal{Y}} \alpha_{i-1}\left(y^\prime\right)\cdot\psi\left(y^\prime,y,i\right), \\
    &= \sum_{y^\prime\in\mathcal{Y}} \alpha_{i-1}\left(y^\prime\right) \cdot \psi_{\text{EMIT}}(x_i\mid y)\cdot \psi_{\text{TRANS}}\left(y\mid y^\prime\right).
\end{align*}$$

In the logarithmic space,

$$\begin{align*}
    \log\alpha_i(y) &= \log\sum_{y^\prime\in\mathcal{Y}} \alpha_{i-1}\left(y^\prime\right) \cdot \psi_{\text{EMIT}}(x_i\mid y)\cdot \psi_{\text{TRANS}}\left(y\mid y^\prime\right), \\
    &= \log\sum_{y^\prime\in\mathcal{Y}} \exp\left(\log\left(\alpha_{i-1}\left(y^\prime\right) \cdot \psi_{\text{EMIT}}(x_i\mid y)\cdot \psi_{\text{TRANS}}\left(y\mid y^\prime\right)\right)\right), \\
    &= \log\sum_{y^\prime\in\mathcal{Y}}\exp\left(\log \alpha_{i-1}\left(y^\prime\right) + h_i[y] + \textbf{P}_{y, y^\prime}\right), \\
    &= \underset{y^\prime\in\mathcal{Y}}{\operatorname{log\,sum\,exp}}\left\{\log\alpha_{i-1}\left(y^\prime\right) + h_i[y] + \textbf{P}_{y, y^\prime}\right\}.
\end{align*}$$

Using equation $\eqref{eq:relation}$, the left of minus of equation $\eqref{eq:log-partition}$ can be written as:

$$\begin{align}
    \underset{\mathbf{y^\prime}}{\operatorname{log\,sum\,exp}}\left\{\text{Score}\left(\mathbf{y^\prime}\mid\mathbf{x}\right)\right\} &= \log\sum_{\mathbf{y^\prime}}\exp\left(\text{Score}\left(\mathbf{y^\prime}\mid\mathbf{x}\right)\right), \\
    &= \log\sum_{\mathbf{y^\prime}}\exp\left(\log\psi\left(y_1^\prime\ldots y_m^\prime\right)\right), \\
    &= \log\sum_{\mathbf{y^\prime}}\psi\left(y_1^\prime\ldots y_m^\prime\right). \label{eq:bi-lstm-crf-partition}\\
\end{align}$$

Also, the probability $\psi\left(y_1^\prime\ldots y_m^\prime\right)$ in equation $\eqref{eq:bi-lstm-crf-partition}$ is determined by the last tag in the sequence, so:

$$\begin{align*}
    \underset{\mathbf{y^\prime}}{\operatorname{log\,sum\,exp}}\left\{\text{Score}\left(\mathbf{y^\prime}\mid\mathbf{x}\right)\right\} &= \log\sum_{y_m^\prime\in\mathcal{Y}}p\left({y_m^\prime}\mid\mathbf{x}\right)\cdot\psi\left(y_m^\prime,\text{STOP},m+1\right), \\
    &= \log\sum_{y_m^\prime\in\mathcal{Y}}a_m\left(y_m^\prime\right)\cdot\psi\left(y_m^\prime,\text{STOP},m+1\right), \\
    &= \log\sum_{y^\prime\in\mathcal{Y}}\exp\left\{\log a_m\left(y^\prime\right)+\log\psi\left(y^\prime,\text{STOP},m+1\right)\right\}, \\
    &= \underset{y^\prime\in\mathcal{Y}}{\operatorname{log\,sum\,exp}}\left\{\log a_m\left(y^\prime\right)+\log\psi\left(y^\prime,\text{STOP},m+1\right)\right\}.
\end{align*}$$


## Computational Algorithms and Implementations

The algorithms of computing partition function is concluded in the pseudocode in below.

- **Input:** Tagged sentences $\left(\mathbf{x}^1,\mathbf{y}^1\right),\ldots,\left(\mathbf{x}^n,\mathbf{y}^n\right)$, features $h[y]$.

- **Definitions:** Define $\mathcal{Y}_i = \mathcal{Y}$ for $i=1\ldots m$.

- **Initialization (forward terms):** For all $y\in\mathcal{Y}_1$,

$$\begin{equation*}
    \log\alpha_1(y) = h_1[y] + \textbf{P}_{y, \text{START}}.
\end{equation*}$$

- **Recursion (forward terms):** For all $i = 2\ldots m$, $y\in\mathcal{Y}_i$,

$$\begin{equation*}
    \log\alpha_i(y) = \underset{y^\prime\in\mathcal{Y}}{\operatorname{log\,sum\,exp}}\left\{\log\alpha_{i-1}\left(y^\prime\right) + h_i[y] + \textbf{P}_{y, y^\prime}\right\}.
\end{equation*}$$

- **Calculations:**

$$\begin{equation*}
    \underset{\mathbf{y}}{\operatorname{log\,sum\,exp}}\left\{\text{Score}\left(\mathbf{y}\mid\mathbf{x}\right)\right\} = \underset{y\in\mathcal{Y}}{\operatorname{log\,sum\,exp}}\left\{\log \alpha_m(y)+\log\psi\left(y,\text{STOP},m+1\right)\right\}.
\end{equation*}$$

The following code is what the algorithms look like. Check out the [python script](https://github.com/zhiqwang/crf.pytorch/blob/master/model.py) to see the whole network `class`.

```python
def _forward_variables(self, feats):

    # Do the forward algorithm to compute the partition function
    init_log_alpha = torch.full((1, self.tagset_size), -10000.)
    # START_TAG has all of the score.
    init_log_alpha[0, self.tag_to_ix[START_TAG]] = 0.

    # Wrap in a variable so that we will get automatic backprop
    log_alpha = init_log_alpha

    # Iterate through the sentences
    for feat in feats:
        log_alpha[0] = torch.logsumexp(log_alpha + self.transitions + feat.view(-1,1), dim=1)

    terminal_var = log_alpha + self.transitions[self.tag_to_ix[STOP_TAG]].view(1,-1)
    score = torch.logsumexp(terminal_var, dim=1)
    return score
```

Now we have learnt the $\textbf{P}$ parameters of the CRFs.


## Sequence Prediction

For an untagged sentence $\mathbf{x}$, we want to know the tag of $\mathbf{x}$ that maximum the probability $p\left(\mathbf{y}\mid\mathbf{x}\right)$. The decoding problem is then to find an entire sequence of tags such that the sum of the transition scores is maximized. We solve this problem using the **Viterbi algorithm:**

- **Input:** a sentences $x_1\ldots x_m$, parameters $\textbf{P}_{y, y^\prime}$ and features $h[y]$.

- **Initialization:** Set $\log\pi_0(\text{START}) = 0$, for $y$ in $\mathcal{Y}$,

$$\begin{equation*}
    \log\pi_1(y) = h_1[y] + \textbf{P}_{y, \text{START}}.
\end{equation*}$$

- **Recursion:** For $i = 2\ldots m$, $y\in\mathcal{Y}_i$,

$$\begin{equation*}
    \log \pi_i(y), \operatorname{bp}_i\,(y) = \max_{y^\prime\in\mathcal{Y}_{i-1}} \left\{\log \pi_{i-1}\left(y^\prime\right) + h_i[y] + \textbf{P}_{y, y^\prime}\right\}.
\end{equation*}$$

- **Calculations:**

$$\begin{equation*}
    y_m = \underset{y\in\mathcal{Y}_m}{\operatorname{arg\,max}}\,\log\pi_m(y).
\end{equation*}$$

- **Back pointers** For $i = (m-1)\ldots 1$,

$$\begin{equation*}
    y_i = \operatorname{bp}_{i+1}\,(y_{i+1}).
\end{equation*}$$

- **Return:** the tag sequence $y_1\ldots y_m$.

Code snippet:

```python
def _viterbi_decode(self, feats):
    backpointers = torch.zeros((len(feats), self.tagset_size), dtype=torch.long)

    # Initialize the viterbi variables in log space
    init_log_pi = torch.full((1, self.tagset_size), -10000.)
    init_log_pi[0, self.tag_to_ix[START_TAG]] = 0

    # forward_var at step i holds the viterbi variables for step i-1
    log_pi = init_log_pi
    for i, feat in enumerate(feats):
        # We don't include the emission scores here because the max
        # does not depend on them (we add them in below)
        log_pi[0], backpointers[i] = torch.max(log_pi + self.transitions, dim=1)
        # Now add in the emission scores, and assign log_pi to the set
        # of viterbi variables we just computed
        log_pi = log_pi + feat.view(1, -1)

    # Transition to STOP_TAG
    terminal_var = log_pi + self.transitions[self.tag_to_ix[STOP_TAG]].view(1,-1)
    path_score, best_tag_id = torch.max(terminal_var, dim=1)

    # Follow the back pointers to decode the best path.
    best_path = [best_tag_id.item()]
    for i in range(len(feats)):
        idx = len(feats) - i - 1
        best_tag_id = backpointers[idx, best_tag_id]
        best_path.append(best_tag_id.item())
    # Pop off the start tag (we dont want to return that to the caller)
    start = best_path.pop()
    assert start == self.tag_to_ix[START_TAG]  # Sanity check
    best_path.reverse()
    return path_score, best_path
```


## Conclusions

You can refer to the [notebook](https://github.com/zhiqwang/crf.pytorch/blob/master/demo.ipynb) for more details in a real application.


## References

  [^1]: [Guthrie's tutorial on CRFs.](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
  [^2]: [Collins's write up on CRFs.](http://www.cs.columbia.edu/~mcollins/crf.pdf)
  [^3]: [Chen' introduction to Conditional Random Fields.](http://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields)
  [^4]: [Sutton's overview of CRFs](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf)
