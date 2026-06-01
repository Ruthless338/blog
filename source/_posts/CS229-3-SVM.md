---
title: CS229-3 Support Vector Machines
date: 2026-05-31
categories:
    - 科研
tags:
    - CS229
    - 机器学习
mathjax: true
---
支持向量机SVM的原理。
<!--more-->

## 1. 线性分类器

二分类问题中，标签为 $y^{(i)}\in\{-1,1\}$（SVM 与 logistic regression 的记号差异）。

线性分类器：

$$
h_{w,b}(x)=g(w^T x+b)
$$

其中：

$$
g(z)=\begin{cases}1,&z\ge 0\\-1,&z<0\end{cases}
$$

决策边界是 $w^T x+b=0$。$w^T x+b>0$ 预测正类，$w^T x+b<0$ 预测负类。

## 2. Functional Margin（函数间隔）

对样本 $(x^{(i)},y^{(i)})$，定义其关于参数 $(w,b)$ 的 functional margin：

$$
\hat\gamma^{(i)}=y^{(i)}(w^T x^{(i)}+b)
$$

- 分类正确时 $y^{(i)}(w^T x^{(i)}+b)>0$，错误时 $<0$
- 正负判断对错，大小表示"有多自信"

整个训练集的 functional margin 取最小值：$\hat\gamma=\min_{i=1,\dots,m}\hat\gamma^{(i)}$

### Functional margin 的问题：尺度不唯一

缩放参数 $(w,b)\leftarrow(2w,2b)$ 不改变分类边界（$w^T x+b=0$ 与 $2w^T x+2b=0$ 是同一超平面），但 functional margin 变成 2 倍。因此 functional margin 不是真正的几何距离——这引出了 geometric margin。

## 3. Geometric Margin（几何间隔）

点 $x^{(i)}$ 到超平面 $w^T x+b=0$ 的有符号距离为 $\frac{w^T x^{(i)}+b}{\|w\|}$。考虑标签方向，定义 geometric margin：

$$
\gamma^{(i)}=y^{(i)}\frac{w^T x^{(i)}+b}{\|w\|}
$$

**核心想法：** 在所有能正确分类的超平面里，选择 geometric margin 最大的那个。

$$
\max_{\gamma,w,b}\gamma\quad\text{subject to}\quad y^{(i)}(w^T x^{(i)}+b)\ge\gamma\|w\|,\ i=1,\dots,m
$$

## 4. Canonical Scaling 与 Primal Problem

同时缩放 $(w,b)$ 不改变边界，可规定 $\hat\gamma=1$（令离边界最近的样本满足 $y^{(i)}(w^T x^{(i)}+b)=1$）。此时 geometric margin $\gamma=\frac{\hat\gamma}{\|w\|}=\frac{1}{\|w\|}$。

最大化 geometric margin 等价于最小化 $\|w\|$，方便求导写成 $\frac12\|w\|^2$。hard-margin SVM 的 primal problem：

$$
\min_{w,b}\frac12\|w\|^2\quad\text{subject to}\quad y^{(i)}(w^T x^{(i)}+b)\ge1,\ i=1,\dots,m
$$

## 5. Lagrangian

引入 Lagrange multipliers $\alpha_i\ge0$，将约束合并进目标函数：

$$
L(w,b,\alpha)=\frac12\|w\|^2-\sum_{i=1}^m\alpha_i[y^{(i)}(w^T x^{(i)}+b)-1]
$$

展开：

$$
L(w,b,\alpha)=\frac12\|w\|^2-\sum_{i=1}^m\alpha_i y^{(i)}(w^T x^{(i)}+b)+\sum_{i=1}^m\alpha_i
$$

**对 $w$ 求偏导：**

$$
\nabla_w L=w-\sum_{i=1}^m\alpha_i y^{(i)}x^{(i)}=0\quad\Rightarrow\quad w=\sum_{i=1}^m\alpha_i y^{(i)}x^{(i)}
$$

**对 $b$ 求偏导：**

$$
\frac{\partial L}{\partial b}=-\sum_{i=1}^m\alpha_i y^{(i)}=0\quad\Rightarrow\quad\sum_{i=1}^m\alpha_i y^{(i)}=0
$$

## 6. Dual Problem

将上述结果代回 Lagrangian，得到 dual problem：

$$
\max_\alpha W(\alpha)=\sum_{i=1}^m\alpha_i-\frac12\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_j y^{(i)}y^{(j)}\langle x^{(i)},x^{(j)}\rangle
$$

subject to $\alpha_i\ge0,\ i=1,\dots,m$ 且 $\sum_{i=1}^m\alpha_i y^{(i)}=0$。

**关键结构：** dual problem 只依赖样本之间的 inner product $\langle x^{(i)},x^{(j)}\rangle$——这是使用 kernel 的基础。

## 7. Support Vectors

由 KKT complementary slackness：

$$
\alpha_i[y^{(i)}(w^T x^{(i)}+b)-1]=0
$$

- 若样本不在 margin boundary 上（$y^{(i)}(w^T x^{(i)}+b)>1$），则 $\alpha_i=0$
- 只有 $y^{(i)}(w^T x^{(i)}+b)=1$ 的样本才可能有 $\alpha_i>0$——这些就是 **support vectors**

**直觉：** SVM 的决策边界只由最靠近边界的那些样本决定。

## 8. 预测函数的 Dual Form

由 $w=\sum_{i=1}^m\alpha_i y^{(i)}x^{(i)}$，对新样本 $x$：

$$
w^T x+b=\sum_{i=1}^m\alpha_i y^{(i)}\langle x^{(i)},x\rangle+b
$$

分类器：

$$
h_{w,b}(x)=g\Bigl(\sum_{i=1}^m\alpha_i y^{(i)}\langle x^{(i)},x\rangle+b\Bigr),\quad g(z)=\begin{cases}1,&z\ge0\\-1,&z<0\end{cases}
$$

因大部分 $\alpha_i=0$，实际预测仅由 support vectors 决定。

## 9. Kernel

Kernel 可理解为高维特征空间中的内积，但无需显式做特征映射：$K(x,z)=\phi(x)^T\phi(z)$。

**常见 kernel：**

- **Linear:** $K(x,z)=x^T z$
- **Polynomial:** $K(x,z)=(x^T z+c)^d$
- **Gaussian / RBF:** $K(x,z)=\exp\bigl(-\frac{\|x-z\|^2}{2\sigma^2}\bigr)$

### 合法 Kernel 的条件

定义 kernel matrix $K_{ij}=K(x^{(i)},x^{(j)})$。合法 kernel 要求 kernel matrix symmetric positive semidefinite：$K=K^T$ 且对任意向量 $z$ 有 $z^T Kz\ge0$（Mercer condition）。

## 10. Soft-margin SVM

hard-margin 要求数据完全线性可分，现实中往往做不到。引入 **slack variables** $\xi_i\ge0$，允许部分样本违反 margin：

$$
y^{(i)}(w^T x^{(i)}+b)\ge1-\xi_i
$$

同时在目标中惩罚：

$$
\min_{w,b,\xi}\frac12\|w\|^2+C\sum_{i=1}^m\xi_i
$$

subject to $y^{(i)}(w^T x^{(i)}+b)\ge1-\xi_i$，$\xi_i\ge0$。

**参数 $C$：** 控制 margin 大小与训练误差的 trade-off。$C$ 大 → 更注重分类正确（少违反）；$C$ 小 → 更注重 margin 大。
