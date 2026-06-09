---
title: Matrix theory in machine learning
date: 2026-06-09
categories:
    - 科研
tags:
    - matrix
    - 机器学习
mathjax: true
---
子空间、秩、奇异矩阵、最小二乘.

<!--more-->
---

# 1. 从线性回归开始：矩阵 $X$ 到底在做什么？

在线性回归里，数据矩阵：

$$X \in \mathbb{R}^{m \times n}$$

- $m$ = 样本数
- $n$ = 特征数

参数 $\theta \in \mathbb{R}^n$，预测值 $\hat y = X\theta$。

很多人觉得这只是一个矩阵乘法，但它有一个非常重要的几何意义：

> $X$ 把参数空间 $\mathbb{R}^n$ 中的一个参数向量 $\theta$，映射成样本空间 $\mathbb{R}^m$ 中的一个预测向量 $\hat y$。

即 $X$ 是一个线性变换：$\theta \mapsto X\theta$。后面所有内容都围绕它展开。

---

# 2. Column Space：模型到底能预测出哪些 $y$？

$X\theta$ 是 $X$ 列向量的线性组合。设：

$$
X =
\begin{bmatrix}
| & | &  & | \\
x_1 & x_2 & \cdots & x_n \\
| & | &  & |
\end{bmatrix}
$$

则 $X\theta = \theta_1x_1 + \theta_2x_2 + \cdots + \theta_nx_n$。

所有可能的预测值组成的集合就是 $\operatorname{Col}(X)$，即 $X$ 的列空间：

> 模型通过调节参数 $\theta$，所有可能产生的预测向量集合。

真实标签 $y \in \mathbb{R}^m$，但模型不一定能完美表达它：

- 若 $y \in \operatorname{Col}(X)$：存在某个 $\theta$ 使 $X\theta = y$，训练误差可以为 0。
- 若 $y \notin \operatorname{Col}(X)$：无论怎么选 $\theta$，模型都不能完全拟合 $y$，只能找一个最接近的预测向量——这就是最小二乘。

---

# 3. 最小二乘：不是公式，而是投影

线性回归最小二乘问题：

$$\min_\theta \|X\theta - y\|_2^2$$

几何意义：

> 在 $\operatorname{Col}(X)$ 里面，找一个点 $X\theta$，让它离 $y$ 最近。

即把 $y$ 正交投影到 $\operatorname{Col}(X)$ 上。投影点是 $\hat y = X\theta^*$，残差 $r = y - X\theta^*$。

因为 $\hat y$ 是最接近 $y$ 的点，残差 $r$ 必须和整个列空间正交：

$$X^T(y - X\theta^*) = 0$$

整理得 **normal equation（正规方程）**：

$$X^T X \theta^* = X^T y$$

理解它的核心是：**最优解对应的残差，必须和模型能表达的所有方向正交**，而不是机械地记 $\theta = (X^TX)^{-1}X^Ty$——那只是 $X^TX$ 可逆时的特殊解。

---

# 4. Rank：数据里有多少个真正独立的方向？

矩阵 $X \in \mathbb{R}^{m \times n}$ 的 rank 表示 $X$ 的列向量中有多少个线性无关的方向，且 $\operatorname{rank}(X) \leq \min(m,n)$。

- $\operatorname{rank}(X)=n$：$n$ 个特征列线性无关（列满秩）
- $\operatorname{rank}(X)<n$：特征之间有线性冗余，参数可能不唯一

---

# 5. Null Space：为什么参数会不唯一？

$X$ 的零空间：$\operatorname{Null}(X)=\{v: Xv=0\}$，即所有被 $X$ 映射成 $0$ 的方向。

若存在非零向量 $v$ 使 $Xv=0$，则：

$$X(\theta+v)=X\theta+Xv=X\theta$$

> $\theta$ 和 $\theta+v$ 是两个不同的参数，但预测完全一样。

这在机器学习里很常见：特征冗余、样本数少于特征数、过参数化模型（如深度学习）都会导致非零零空间。例如 100 个样本、10000 个特征时，$X \in \mathbb{R}^{100 \times 10000}$ 最多 rank 100，一定有大量方向 $v$ 满足 $Xv=0$，意味着无数个参数向量可以得到完全一样的训练集预测。

---

# 6. 欠定、超定和机器学习中的三种情况

线性系统 $X\theta = y$，$X \in \mathbb{R}^{m \times n}$。

### 超定（$m > n$）
方程多于未知数，通常不能完全满足 $X\theta=y$，需要最小二乘 $\min_\theta \|X\theta-y\|_2^2$。经典线性回归通常是这种情况。

### 方阵（$m = n$）
若 $X$ 可逆则有唯一解 $\theta = X^{-1}y$；若 $X$ 奇异则可能无解或无穷多解。

### 欠定（$n > m$）
未知数多于方程，通常有无穷多解。这在现代 ML 中非常常见（高维特征、过参数化、大模型参数远多于训练样本）。问题变成：**在所有能拟合训练集的参数里，应该选哪个？** 这就引出最小范数解、伪逆、正则化、implicit bias 等概念。

---

# 7. 为什么最小范数解重要？

假设很多解满足 $X\theta = y$，可写成 $\theta = \theta_0 + v$，其中 $v \in \operatorname{Null}(X)$：

$$X(\theta_0+v)=X\theta_0+Xv=y$$

零空间方向不影响训练集预测。但参数范数可能差很多——过大的参数会导致数值不稳定、对扰动敏感、泛化变差。因此我们偏好范数较小的解：

$$\min_\theta \|\theta\|_2 \quad \text{s.t.} \quad X\theta=y$$

这就是**最小范数解**，它和伪逆有关：$\theta^*=X^+y$，其中 $X^+$ 是 Moore-Penrose pseudoinverse（伪逆）。

> 当参数不唯一时，伪逆会选择其中范数最小的那个解。

---

# 8. 为什么普通逆不够？伪逆和正则化

普通逆矩阵只适用于满秩方阵。但 ML 中的 $X \in \mathbb{R}^{m \times n}$ 经常不是方阵，即使 $X^TX$ 是方阵也可能不可逆——原因包括特征冗余、样本太少、维度太高、共线性强、某些方向方差接近 0。

于是需要：
- **伪逆**：处理非方阵和奇异矩阵
- **SVD**：看清矩阵哪些方向可逆、哪些不可逆
- **Ridge**：给 $X^TX$ 加 $\lambda I$ 使其更稳定

Ridge 形式：

$$\theta = (X^TX+\lambda I)^{-1}X^Ty$$

直觉：若 $X^TX$ 有些方向太弱甚至为 0，加上 $\lambda I$ 后所有方向都被补了一点"强度"，更容易可逆、更稳定。

---

# 9. 用一条链条串起来

对于 $\min_\theta \|X\theta-y\|_2^2$，核心逻辑链：

$$
\operatorname{rank}(X)<n
\;\Longrightarrow\;
\operatorname{Null}(X)\neq \{0\}
\;\Longrightarrow\;
\exists v\neq 0,\ Xv=0
\;\Longrightarrow\;
X(\theta+v)=X\theta
$$

$$
\Longrightarrow\; \text{参数不唯一}
\;\Longrightarrow\; X^TX \text{ 不可逆}
\;\Longrightarrow\; \text{普通 normal equation 公式失效}
\;\Longrightarrow\; \text{需要伪逆 / SVD / Ridge}
$$

---

# 10. 回到机器学习：这些概念解决什么问题？

**为什么线性回归有时候不能直接用公式解？**
$\theta=(X^TX)^{-1}X^Ty$ 要求 $X^TX$ 可逆。特征冗余或样本数少于特征数时 $X$ 列不满秩 → $X^TX$ 奇异 → 公式失效。

**为什么特征冗余会造成不稳定？**
存在 $v \neq 0$ 使 $Xv=0$，则 $\theta$ 和 $\theta+v$ 的训练集预测完全相同，模型可在这些方向上随意漂移。

**为什么最小二乘是投影？**
模型所有可能预测值都在 $\operatorname{Col}(X)$ 里，若 $y$ 不在其中，最近点就是 $y$ 到 $\operatorname{Col}(X)$ 的投影。

**为什么需要正则化？**
矩阵不可逆或接近不可逆时解非常不稳定。正则化相当于说：不要为了拟合训练集，在数据无法可靠确定的方向上乱跑。

---

# 11. 总结

线性回归表面上在找参数 $\theta$，但从矩阵论角度看在做三件事：

1. 用 $X$ 的列空间描述模型能表达的所有预测结果
2. 把真实标签 $y$ 投影到这个预测空间里
3. 在所有能产生该预测的参数中，选择合适的参数

- $X$ 的列空间足够大 → 模型表达能力强
- $\operatorname{rank}(X)$ 不满 → 特征冗余、参数不唯一
- $X^TX$ 奇异 → 普通逆失效，需要 SVD、伪逆、正则化

第一讲的核心不是"会不会求 rank"，而是这个意识：

> **矩阵的 rank 决定信息量，column space 决定模型能表达什么，null space 决定参数有哪些冗余方向，最小二乘就是投影，奇异矩阵意味着某些方向的信息丢失。**
