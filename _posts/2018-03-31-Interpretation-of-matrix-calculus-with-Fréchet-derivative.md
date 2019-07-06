---
layout: post
title: Interpretation of matrix calculus with Fréchet derivative
tags: [fréchet-derivative, matrix-calculus]
date: 2018-03-31 20:00:00 +0800
---

> After a period of learning the [Fréchet derivative](https://en.wikipedia.org/wiki/Fréchet_derivative), I look back to think about the problem of computing the derivative of $y = \Vert A x - b \Vert^2$ with respect to the variable $x$. I knew the derivative of this function, $2A^\mathsf{T}(Ax - b)$, when I was a second year master student (2015). And I have used this conclusion effectively in combination with the product rule in multiple applications. Unfortunately, I feel confused about how I came to this conclusion at that time. So I rethink the problem from the viewpoint of *Fréchet derivative*, and then I'll give an answer based on this viewpoint.

Traditionally, *matrix calculus* is presented as a notation for organizing partial derivatives.[^1] It collects the various partial derivatives of a single function with respect to many variables into vectors that can be treated as single entities.[^2] In our case, the derivative of a scalar $y$ by a vector

$$x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix},$$

is written (in denominator layout notation) as

$$
\frac{\partial y}{\partial x} =
    \begin{bmatrix}
        {\frac{\partial y}{\partial x_{1}}} \\
        {\frac{\partial y}{\partial x_{2}}} \\
        \vdots \\
        {\frac{\partial y}{\partial x_{n}}}
    \end{bmatrix}.
$$

Apply this formula to our problem, you can get the derivative is $2A^\mathsf{T}(Ax - b)$.

Now, let's interpret the problem with the help of *Fréchet derivative*. Write $g(x) = A x - b$, and $h(x) = x^\mathsf{T}x$, and $y = h \circ g$, we have $Dh(x)(u) = 2x^\mathsf{T}u$, and $Dg(x)(u) = Au$, use the chain rule, so

$$\begin{align*}
    Dy(x)(u) &= D(h\circ g)(x) (u) \\
    &= Dh(g(x))\circ Dg(x)(u) \\
    &= Dh(g(x))(Au) \\
    &= 2(Ax-b)^\mathsf{T}Au,
\end{align*}$$

and then

$$\frac{\partial y}{\partial x} = 2 A^\mathsf{T}(Ax - b).$$

## References

  [^1]: [Xu's write up on Matrix derivative.](http://www.cs.cmu.edu/~minx/matrixcal.pdf)
  [^2]: [Wiki page of Matrix calculus.](https://en.wikipedia.org/wiki/Matrix_calculus)
