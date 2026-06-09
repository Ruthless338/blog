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
正定 (PD) 与半正定 (PSD) 矩阵在机器学习中的作用。
<!--more-->

# 1. 二次型：为什么总是出现 $x^TAx$？

$x^TAx$ 的直观理解：**矩阵 $A$ 衡量方向 $x$ 上的"大小""能量""方差""曲率"或"距离"**。在机器学习中反复出现。

### 线性回归
平方损失 $\|X\theta-y\|^2$ 展开为 $(X\theta-y)^T(X\theta-y)$，其中关于 $\theta$ 的二次项是 $\theta^T X^TX \theta$——loss 由二次型控制。

### 协方差
对随机向量 $X$，协方差矩阵 $\Sigma = \mathbb{E}[(X-\mu)(X-\mu)^T]$。任意方向 $a$ 上的方差：

$$\operatorname{Var}(a^TX) = a^T\Sigma a$$

协方差通过二次型描述数据在各方向上的方差。

### 高斯分布
多元高斯密度中的马氏距离：

$$(x-\mu)^T\Sigma^{-1}(x-\mu)$$

衡量样本 $x$ 离均值 $\mu$ 有多远，且考虑了不同方向的方差大小——又是一个二次型。

### 优化
函数 $f(\theta)$ 在某点的二阶近似：

$$f(\theta+\Delta) \approx f(\theta) + \nabla f(\theta)^T\Delta + \frac{1}{2}\Delta^T H \Delta$$

其中 $H=\nabla^2 f(\theta)$ 是 Hessian。$\Delta^T H \Delta$ 表示沿方向 $\Delta$ 的曲率。

---

# 2. 正定与半正定的定义

对称矩阵 $A$：

- **正定 (PD)**：$x^TAx > 0,\ \forall x\neq 0$，记作 $A \succ 0$
- **半正定 (PSD)**：$x^TAx \ge 0,\ \forall x$，记作 $A \succeq 0$

---

# 3. 为什么要求矩阵对称？

二次型 $x^TAx$ 实际上只与 $A$ 的对称部分有关。任意矩阵可拆分：

$$A = \frac{A+A^T}{2} + \frac{A-A^T}{2}$$

反对称部分 $B=\frac{A-A^T}{2}$ 满足 $x^TBx=0$，因此：

$$x^TAx = x^T\left(\frac{A+A^T}{2}\right)x$$

所以正定性讨论只针对对称矩阵。

---

# 4. 从特征值理解正定性

实对称矩阵 $A$ 可特征分解：$A = Q\Lambda Q^T$，其中 $\Lambda = \operatorname{diag}(\lambda_1,\dots,\lambda_n)$。于是：

$$x^TAx = x^T Q\Lambda Q^T x$$

令 $z=Q^Tx$（$Q$ 正交，相当于旋转变换），得：

$$x^TAx = z^T\Lambda z = \sum_{i=1}^n \lambda_i z_i^2$$

> **二次型本质：每个方向的平方 $z_i^2$ 乘上该方向特征值 $\lambda_i$，再求和。**

因此：
- 所有 $\lambda_i > 0$ → $x^TAx > 0\ (\forall x\neq 0)$ → $A$ 正定
- 所有 $\lambda_i \ge 0$ → $x^TAx \ge 0\ (\forall x)$ → $A$ 半正定
- 存在 $\lambda_i < 0$ → 沿该特征方向 $x^TAx$ 变负

即：

$$A \succ 0 \Longleftrightarrow \lambda_i > 0,\ \forall i$$
$$A \succeq 0 \Longleftrightarrow \lambda_i \ge 0,\ \forall i$$

---

# 5. 为什么 $X^TX$ 一定是半正定？

对任意向量 $z$：

$$z^T X^T X z = (Xz)^T(Xz) = \|Xz\|_2^2 \ge 0$$

因此 $X^TX \succeq 0$（恒为 PSD）。

### 什么时候 $X^TX$ 进一步成为正定？

若 $z^T X^T X z = \|Xz\|^2 > 0$ 对所有 $z \neq 0$ 成立，即 $Xz \neq 0,\ \forall z\neq 0$，等价于 $\operatorname{Null}(X)=\{0\}$——$X$ 列满秩。所以：

$$X^TX \succ 0 \Longleftrightarrow X \text{ 列满秩}$$

**与第一讲的衔接：**

- $X$ 列不满秩 → null space 非空 → $X^TX$ 奇异（第一讲）
- $X^TX$ 恒为 PSD；$X$ 列满秩则进一步为 PD；否则只是 PSD、不是 PD（第二讲）

---

# 6. 正定性与线性回归

线性回归平方损失 $J(\theta)=\frac{1}{2}\|X\theta-y\|^2$ 的 Hessian 是 $\nabla^2 J(\theta)=X^TX$。

- $X^TX \succeq 0$ → 损失函数是**凸函数**
- $X^TX \succ 0$（$X$ 列满秩）→ 损失**严格凸**，有唯一最优解
- $X^TX \succeq 0$ 但非 PD（$X$ 列不满秩）→ 凸但不严格凸，可能无穷多最优解

> **PSD → 凸；PD → 严格凸 / 唯一解。**

---

# 7. 正定性与优化：Hessian 在看什么？

一维函数：二阶导 > 0 向上弯（局部最小），< 0 向下弯（局部最大）。多维推广：方向 $v$ 上的二阶曲率 = $v^T H v$。

- $H \succ 0$：所有方向向上弯 → 局部最小，解稳定，牛顿法方向合理
- $H \succeq 0$：没有向下弯的方向，但可能有平坦方向 → 常见于参数冗余、过参数化、特征不满秩
- $H$ 不定（有正有负特征值）：有些方向上升、有些下降 → **鞍点**，深度学习高维空间中大量出现

---

# 8. 正定性与协方差矩阵

协方差矩阵 $\Sigma = \mathbb{E}[(X-\mu)(X-\mu)^T]$ 恒为 PSD，证明：

$$a^T\Sigma a = a^T \mathbb{E}[(X-\mu)(X-\mu)^T] a = \mathbb{E}[a^T(X-\mu)(X-\mu)^T a]$$

$a^T(X-\mu)$ 是标量，故：

$$a^T\Sigma a = \mathbb{E}\left[\left(a^T(X-\mu)\right)^2\right] = \operatorname{Var}(a^TX) \ge 0$$

因此 $\Sigma \succeq 0$。

### 什么时候协方差不是正定？

存在非零方向 $a$ 使 $a^T\Sigma a = 0$，即 $\operatorname{Var}(a^TX)=0$——数据在该方向上无变化。常见于：
- 数据落在低维子空间
- 某些特征是其他特征的线性组合
- 样本数太少，无法撑满整个维度

例如 $x \in \mathbb{R}^{10000}$ 但只有 $m=100$ 个样本，经验协方差矩阵必为低秩（奇异）。这也是 GDA、GMM、Factor Analysis 中常需处理协方差奇异的原因。

---

# 9. 为什么高斯分布需要正定协方差？

多元高斯分布：

$$p(x) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$

要求 $\Sigma \succ 0$，两个原因：
1. **方差必须合法**：若有负特征值 → 某方向方差为负 → 无概率意义
2. **$\Sigma^{-1}$ 必须存在**：若 $\Sigma$ 奇异则无法求逆，普通高斯密度无定义（退化高斯除外）

---

# 10. Mahalanobis Distance：为什么用 $\Sigma^{-1}$？

马氏距离：

$$d_M(x,\mu)^2 = (x-\mu)^T\Sigma^{-1}(x-\mu)$$

与欧氏距离 $\|x-\mu\|^2$（默认各方向尺度相同）的区别：马氏距离考虑了各方向的方差差异。

将 $\Sigma = Q\Lambda Q^T$ 代入，$\Sigma^{-1} = Q\Lambda^{-1}Q^T$，令 $z = Q^T(x-\mu)$：

$$(x-\mu)^T\Sigma^{-1}(x-\mu) = \sum_i \frac{z_i^2}{\lambda_i}$$

- 方差大的方向 $\lambda_i$ 大 → 对距离惩罚小
- 方差小的方向 $\lambda_i$ 小 → 对距离惩罚大

> **马氏距离本质上是用协方差结构重新定义"距离"。**

---

# 11. 正定性与 Kernel Matrix

Kernel matrix $K_{ij} = K(x_i, x_j)$。合法 kernel 的本质是 $K(x,z) = \phi(x)^T\phi(z)$，即某特征空间中的内积。

对任意系数 $c$：

$$c^T K c = \sum_{i,j} c_i c_j K(x_i,x_j) = \sum_{i,j} c_i c_j \phi(x_i)^T\phi(x_j)$$

合并：

$$c^T K c = \left\|\sum_i c_i \phi(x_i)\right\|^2 \ge 0$$

因此 $K \succeq 0$——kernel matrix 必须是 PSD。这不是人为规定，而是因为合法 kernel 必须能解释为某特征空间中的内积，而 Gram matrix 天然半正定。

若 $K$ 不是 PSD，则不能对应任何真实内积空间中的 Gram matrix，SVM 优化问题可能不再凸，dual problem 不稳定。

---

# 12. 正定性与 Ridge Regression

Ridge 目标函数 $J(\theta) = \|X\theta-y\|^2 + \lambda\|\theta\|^2$ 的 Hessian 是 $X^TX + \lambda I$。

已知 $X^TX \succeq 0$，而 $\lambda I \succ 0$（当 $\lambda > 0$），故：

$$X^TX + \lambda I \succ 0$$

Ridge 目标严格凸，有唯一解。若 $X^TX$ 特征值为 $\mu_1,\dots,\mu_n$，则 $X^TX+\lambda I$ 特征值为 $\mu_1+\lambda,\dots,\mu_n+\lambda$。原来 $\mu_i=0$ 导致不可逆，加 $\lambda$ 后全部 $>0$，矩阵变正定、可逆。

---

# 13. 正定矩阵的机器学习作用总结

| 场景 | 矩阵 | 为什么 PSD/PD 重要 |
|------|------|-------------------|
| 线性回归 | $X^TX$ | Hessian，决定凸性和解是否唯一 |
| Ridge | $X^TX+\lambda I$ | 变成 PD，保证可逆和稳定 |
| 协方差 | $\Sigma$ | 任意方向方差不能为负 |
| 高斯分布 | $\Sigma$ | 需要合法方差和可逆协方差 |
| 马氏距离 | $\Sigma^{-1}$ | 用方差结构衡量距离 |
| Kernel SVM | $K$ | 必须能表示内积空间中的 Gram matrix |
| PCA | $\Sigma$ | 特征值表示各方向方差 |
| 优化 | Hessian $H$ | PSD 表示凸，PD 表示严格局部最小 |

---

# 14. 正定、半正定、奇异矩阵的关系

- **正定必可逆**：$A \succ 0$ → 所有 $\lambda_i > 0$ → 无可逆性障碍
- **半正定不一定可逆**：$A \succeq 0$ → $\lambda_i \ge 0$，若有 $\lambda_i = 0$ 则奇异。例如 $A = \begin{bmatrix}1 & 0\\ 0 & 0\end{bmatrix}$ 是 PSD 但不可逆
- **$X^TX$ 的情况**：永远 PSD；$X$ 列满秩 → PD（可逆）；$X$ 列不满秩 → 仅 PSD（奇异）

---

# 15. 总结

> **正定和半正定不是在背定义，而是在保证机器学习中的"方差、距离、能量、曲率、内积"在所有方向上都是合理的。**

- 协方差 PSD：任意方向方差非负
- Hessian PSD：优化问题没有向下弯的方向
- Hessian PD：局部像碗一样稳定
- Kernel PSD：相似度能解释成合法内积
- Ridge 变 PD：保证矩阵可逆、解唯一、数值稳定

核心直觉：

> **凡是机器学习里出现 $x^TAx$，都要问：这个量表示什么？它能不能为负？如果不能，$A$ 就应该是 PSD；如果还要求每个非零方向都严格有意义，$A$ 就应该是 PD。**
