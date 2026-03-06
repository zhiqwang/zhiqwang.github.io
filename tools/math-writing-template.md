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

Use `$$...$$` directly, on separate lines.

Correct:

```md
$$
\begin{aligned}
\nabla_x f(x) &= Ax + b \\
\nabla_x^2 f(x) &= A
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

You can use labels and references when needed:

```md
$$
\begin{equation}
Z = \sum_i e^{x_i}\label{eq:partition}
\end{equation}
$$

As shown in $\eqref{eq:partition}$, ...
```

## 5) Quick Checklist Before Commit

- This post has `math: true` in front matter.
- No math expression is wrapped by backticks.
- Inline math uses `$...$`.
- Display math uses `$$...$$`.
- Local preview confirms formulas render correctly.
