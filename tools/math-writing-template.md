# Math Writing Template (Jekyll + Chirpy)

Use this template when writing posts that contain LaTeX formulas.

## 1) Front Matter (must include `math: true`)

```yaml
---
title: "Your Math Post Title"
date: 2026-03-06 10:00:00 +0800
categories: [Category]
tags: [tag1, tag2]
math: true
---
```

## 2) Inline Formula Syntax

Use `$...$` directly in text.

Correct:

```md
The loss is $\mathcal{L}(\theta) = \sum_i (y_i - \hat{y}_i)^2$.
```

Wrong (will render as code, not math):

```md
The loss is `$\mathcal{L}(\theta) = \sum_i (y_i - \hat{y}_i)^2$`.
```

## 3) Block Formula Syntax

Use a newline after opening `$$` for all display formulas in this site, and keep one blank line after closing `$$` before normal text.

- Single-line display math: use `$$` + `\begin{equation}...\end{equation}` + `$$`.
- Multi-line display math without numbering: use `$$` + `\begin{aligned}...\end{aligned}` + `$$`.
- Multi-line display math with numbering or `\eqref`: use `$$` + `\begin{equation}\begin{aligned}...\end{aligned}\label{...}\end{equation}` + `$$`.
- After `\end{equation}` or `\end{aligned}`, close with `$$` and then add one blank line before the next paragraph.

Correct (single-line display math):

```md
$$
\begin{equation}
Z = \sum_i e^{x_i}
\end{equation}
$$
```

Correct (multi-line display math):

```md
$$
\begin{aligned}
\nabla_x f(x) &= Ax + b \\
\nabla_x^2 f(x) &= A
\end{aligned}
$$
```

Correct (multi-line display math with equation number / `\eqref`):

```md
$$
\begin{equation}
\begin{aligned}
\log p(y\mid x) &= s(x, y) - \log Z(x) \\
\log Z(x) &= \operatorname{log\,sum\,exp}_{y'} s(x, y')
\end{aligned}
\label{eq:log-partition}
\end{equation}
$$

As shown in $\eqref{eq:log-partition}$, ...
```

Wrong (missing outer `$$` delimiters):

```md
\begin{equation}
Z = \sum_i e^{x_i}
\end{equation}
```

Wrong (putting `\label` inside bare `aligned`; often causes `\eqref` to show `(???)`):

```md
$$
\begin{aligned}
\log p(y\mid x) &= s(x, y) - \log Z(x) \\
\log Z(x) &= \operatorname{log\,sum\,exp}_{y'} s(x, y')\label{eq:log-partition}
\end{aligned}
$$
```

Wrong (will render as code block style):

```md
`$$
\nabla_x f(x) = Ax + b
$$`
```

## 4) Equation Labels and References

Use labels on `equation`, not on bare `aligned`, when you need `\eqref`:

```md
$$
\begin{equation}
Z = \sum_i e^{x_i}
\label{eq:partition}
\end{equation}
$$

As shown in $\eqref{eq:partition}$, ...
```

## 5) Quick Checklist Before Commit

- This post has `math: true` in front matter.
- No math expression is wrapped by backticks.
- Inline math uses `$...$`.
- Display math keeps a newline after opening `$$`.
- Display math keeps one blank line after closing `$$`.
- Single-line display math uses `\begin{equation}...\end{equation}`.
- Multi-line display math without references uses `\begin{aligned}...\end{aligned}`.
- If using `\label`/`\eqref`, wrap with `\begin{equation}\begin{aligned}...` and place `\label` on `equation`.
- Local preview confirms formulas render correctly.

## 6) Named Footnote Template

Use named footnotes for references you may reuse or edit later.

Copyable example:

```md
We use the same notation as prior work.[^collins-crf]
The optimization objective follows a standard derivation.[^sutton-crf]

[^collins-crf]: [Collins's write up on CRFs.](https://www.cs.columbia.edu/~mcollins/crf.pdf)
[^sutton-crf]: [Sutton's overview of CRFs.](https://homepages.inf.ed.ac.uk/csutton/publications/crftut-fnt.pdf)
```

Tips:

- Keep IDs short and stable, e.g. `[^author-topic]`.
- Define each footnote once, usually near the end of the post.
- Reuse the same ID when citing the same source multiple times.
