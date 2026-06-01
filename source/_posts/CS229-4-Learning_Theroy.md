---
title: CS229-4 Learning Theory
date: 2026-06-01
categories:
    - 科研
tags:
    - CS229
    - 机器学习
mathjax: true
---
学习理论研究训练误差与泛化误差的关系。有限假设类用 Hoeffding inequality + union bound 得到 uniform convergence；无限假设类用 VC dimension 衡量模型复杂度，分析泛化所需样本量。
<!--more-->

## 1. Bias / Variance Tradeoff

- **Bias：** 模型的系统性偏差
- **Variance：** 模型对训练数据扰动的敏感程度

| 情况 | 训练误差 | 测试误差 | 问题 |
| --- | :---: | :---: | --- |
| High Bias | 高 | 高 | 模型太简单 |
| High Variance | 低 | 高 | 模型太复杂 |
| 合适 | 较低 | 较低 | 泛化较好 |

## 2. 训练误差与泛化误差

**训练误差（Empirical Error）：**

$$
\hat\varepsilon(h)=\frac1m\sum_{i=1}^m\mathbf{1}\{h(x^{(i)})\neq y^{(i)}\}
$$

模型在训练集上犯错的比例。

**泛化误差（Generalization Error / True Error）：**

$$
\varepsilon(h)=P_{(x,y)\sim\mathcal{D}}(h(x)\neq y)
$$

模型在真实分布上新样本犯错的概率。

## 3. Empirical Risk Minimization (ERM)

从假设类 $\mathcal{H}$ 中选择训练误差最小的模型：

$$
\hat h=\arg\min_{h\in\mathcal{H}}\hat\varepsilon(h)
$$

其中 $\mathcal{H}$ 叫 hypothesis class（候选模型集合）。

## 4. Uniform Convergence

我们希望对所有 $h\in\mathcal{H}$，训练误差都接近真实误差：

$$
|\hat\varepsilon(h)-\varepsilon(h)|\le\gamma
$$

如果这个性质对所有 $h$ 同时成立，就叫 **uniform convergence**——意味着每个模型的训练误差都是它真实误差的可靠估计。

### 为什么 uniform convergence 能保证 ERM 可靠？

设 $\hat h=\arg\min_{h\in\mathcal{H}}\hat\varepsilon(h)$，$h^*=\arg\min_{h\in\mathcal{H}}\varepsilon(h)$。若 uniform convergence 成立（$|\hat\varepsilon(h)-\varepsilon(h)|\le\gamma$ 对所有 $h$ 成立），则：

$$
\varepsilon(\hat h)\le\hat\varepsilon(\hat h)+\gamma\le\hat\varepsilon(h^*)+\gamma\le\varepsilon(h^*)+2\gamma
$$

即 ERM 选出的模型，真实误差最多比候选类中最好的模型差 $2\gamma$。（完整推导见[附录 A](#附录-a-erm-界推导)）

## 5. 有限假设类

假设 $|\mathcal{H}|=k$，有限个候选模型。

### 5.1 Hoeffding Inequality

对固定 $h$，每个样本的分类对错可视为 Bernoulli 随机变量 $Z_i=\mathbf{1}\{h(x^{(i)})\neq y^{(i)}\}$，则 $\hat\varepsilon(h)=\frac1m\sum_{i=1}^m Z_i$ 是 $\varepsilon(h)$ 的估计。Hoeffding inequality 给出：

$$
P(|\hat\varepsilon(h)-\varepsilon(h)|>\gamma)\le2\exp(-2\gamma^2 m)
$$

**含义：** 对固定模型，训练误差偏离真实误差的概率随样本数指数级下降。

### 5.2 Union Bound → Uniform Convergence

用 union bound 扩展到整个 $\mathcal{H}$（推导见[附录 B](#附录-b-union-bound-推导)）：

$$
P(\exists h\in\mathcal{H}:|\hat\varepsilon(h)-\varepsilon(h)|>\gamma)\le\sum_{h\in\mathcal{H}}P(|\hat\varepsilon(h)-\varepsilon(h)|>\gamma)\le2k\exp(-2\gamma^2 m)
$$

### 5.3 Sample Complexity

设失败概率不超过 $\delta$，即 $2k\exp(-2\gamma^2 m)\le\delta$，解得：

$$
m\ge\frac{1}{2\gamma^2}\log\frac{2|\mathcal{H}|}{\delta}
$$

只要样本数满足此条件，就能以至少 $1-\delta$ 的概率保证 $\forall h\in\mathcal{H},|\hat\varepsilon(h)-\varepsilon(h)|\le\gamma$。

- 想要误差更准（$\gamma$ 小）→ 样本量 $1/\gamma^2$ 级别增长
- 想要置信度更高（$\delta$ 小）→ 样本量 $\log(1/\delta)$ 级别增长
- 模型集合更大（$|\mathcal{H}|$ 大）→ 样本量 $\log|\mathcal{H}|$ 级别增长

## 6. 无限假设类与 VC Dimension

当 $|\mathcal{H}|=\infty$ 时，上述 $\log|\mathcal{H}|$ 的 bound 失效，需要新的复杂度度量——**VC Dimension**。

### 6.1 Shattering

给定点集 $S=\{x^{(1)},x^{(2)},\dots,x^{(d)}\}$，如果对任意标签分配 $y^{(i)}\in\{0,1\}$，都存在 $h\in\mathcal{H}$ 能完美分类，则称 $\mathcal{H}$ **shatters** $S$。

**VC dimension** $\text{VC}(\mathcal{H})$ 定义为 $\mathcal{H}$ 能 shatter 的最大点集大小。

**例子：** 二维平面线性分类器 $h_{w,b}(x)=\mathbf{1}\{w^T x+b\ge0\}$ 的 VC dimension 为 $3$。更一般地，$n$ 维线性分类器的 VC dimension 为 $n+1$。

### 6.2 泛化界

VC dimension 为 $d$ 的假设类，泛化界依赖 $d$ 而非 $|\mathcal{H}|$，大致形式：

$$
m=O\!\left(\frac{1}{\gamma^2}\left(d\log\frac{1}{\gamma}+\log\frac{1}{\delta}\right)\right)
$$

核心结论：

- **有限 $\mathcal{H}$：** 复杂度由 $\log|\mathcal{H}|$ 控制
- **无限 $\mathcal{H}$：** 复杂度由 VC dimension 控制

---

## 附录 A：ERM 界推导

第 4 节中 uniform convergence 保证 ERM 可靠的推导，分步展开如下。

设 $\hat h$ 为 ERM 选出的模型，$h^*$ 为 $\mathcal{H}$ 中真实最优模型。若 $|\hat\varepsilon(h)-\varepsilon(h)|\le\gamma$ 对所有 $h\in\mathcal{H}$ 成立，则：

**第一步** — 对 $\hat h$ 使用 uniform convergence：

$$
\varepsilon(\hat h)\le\hat\varepsilon(\hat h)+\gamma
$$

**第二步** — $\hat h$ 是训练误差最小的，所以其训练误差不超过 $h^*$ 的训练误差：

$$
\hat\varepsilon(\hat h)\le\hat\varepsilon(h^*)
$$

**第三步** — 对 $h^*$ 使用 uniform convergence：

$$
\hat\varepsilon(h^*)\le\varepsilon(h^*)+\gamma
$$

串联三步即得：

$$
\boxed{\varepsilon(\hat h)\le\varepsilon(h^*)+2\gamma}
$$

## 附录 B：Union Bound 推导

第 5.2 节中将 Hoeffding bound 从单个假设扩展到整个 $\mathcal{H}$ 的过程。

设 $\mathcal{H}=\{h_1,h_2,\dots,h_k\}$，定义事件 $A_i$ 为"第 $i$ 个假设的训练误差偏离真实误差超过 $\gamma$"：

$$
A_i=\bigl\{|\hat\varepsilon(h_i)-\varepsilon(h_i)|>\gamma\bigr\}
$$

那么"$\mathcal{H}$ 中存在某个假设偏差超过 $\gamma$"即所有 $A_i$ 的并集：

$$
P(\exists h\in\mathcal{H}:|\hat\varepsilon(h)-\varepsilon(h)|>\gamma)=P(A_1\cup A_2\cup\cdots\cup A_k)
$$

由 **union bound**：

$$
P(A_1\cup A_2\cup\cdots\cup A_k)\le\sum_{i=1}^k P(A_i)
$$

代入每个 $A_i$ 的定义和 Hoeffding bound $P(A_i)\le2\exp(-2\gamma^2 m)$：

$$
P(\exists h\in\mathcal{H}:|\hat\varepsilon(h)-\varepsilon(h)|>\gamma)\le\sum_{i=1}^k 2\exp(-2\gamma^2 m)=2k\exp(-2\gamma^2 m)
$$
