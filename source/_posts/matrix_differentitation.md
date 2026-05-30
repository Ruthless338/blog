---
title: 矩阵求导法则
date: 2026-05-30
categories:
    - 科研
tags:
    - 机器学习
mathjax: true
---

## 一、矩阵求导的本质定义

设

$$
X =
\begin{bmatrix}
x_{11} & \cdots & x_{1n} \\
\vdots &        & \vdots \\
x_{m1} & \cdots & x_{mn}
\end{bmatrix}
$$

共有 $m \times n$ 个变量。

那么定义：

$$
\frac{\partial f}{\partial X} = \left( \frac{\partial f}{\partial x_{ij}} \right)
$$

即

$$
\frac{\partial f}{\partial X} =
\begin{bmatrix}
\frac{\partial f}{\partial x_{11}} & \cdots & \frac{\partial f}{\partial x_{1n}} \\
\vdots &        & \vdots \\
\frac{\partial f}{\partial x_{m1}} & \cdots & \frac{\partial f}{\partial x_{mn}}
\end{bmatrix}
$$

> **结论**
>
> 对于 $f: \mathbb{R}^{m \times n} \to \mathbb{R}$，即：输入是矩阵，输出是标量。
>
> 则：$\dfrac{\partial f}{\partial X}$ 与 $X$ **形状相同**。
>
> 即：$X \in \mathbb{R}^{m \times n}$，那么 $\dfrac{\partial f}{\partial X} \in \mathbb{R}^{m \times n}$。

---

## AI 里常见的矩阵求导公式


### 1. 线性项

$$f = a^T x$$

导数：

$$\frac{\partial f}{\partial x} = a$$

$$f = x^T a$$

导数：

$$\frac{\partial f}{\partial x} = a$$

### 2. 二范数

$$f = x^T x$$

导数：

$$\frac{\partial f}{\partial x} = 2x$$

### 3. 二次型

$$f = x^T A x$$

导数：

$$\frac{\partial f}{\partial x} = (A + A^T)x$$

若 $A = A^T$（对称），则：

$$\frac{\partial f}{\partial x} = 2Ax$$

### 4. Frobenius 范数

$$f = \|W\|_F^2$$

导数：

$$\frac{\partial f}{\partial W} = 2W$$

### 5. 线性变换

$$y = Wx$$

对 $x$ 求导：

$$\frac{\partial y}{\partial x} = W$$

对 $W$ 求导（反向传播常用，其中 $L = L(y)$）：

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \, x^T$$

> 这是神经网络权重更新最核心的公式。

### 6. Trace 技巧

$$\frac{\partial}{\partial X} \, \mathrm{tr}(AX) = A^T$$

$$\frac{\partial}{\partial X} \, \mathrm{tr}(X^T A X) = (A + A^T)X$$

若 $A$ 对称：

$$\frac{\partial}{\partial X} \, \mathrm{tr}(X^T A X) = 2AX$$
