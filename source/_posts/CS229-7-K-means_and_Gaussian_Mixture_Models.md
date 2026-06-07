---
title: CS229-7 K-means and EM / Gaussian Mixture Models
date: 2026-06-07
categories:
    - 科研
tags:
    - CS229
    - 机器学习
mathjax: true
---

本节主要讲 K-means 聚类、Gaussian Mixture Models。

<!--more-->

## 1. K-means Clustering

### 1.1 问题设定

K-means 是一个 unsupervised learning 算法。监督学习中有 $\{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$，但聚类问题只有 $\{x^{(i)}\}_{i=1}^{m}$，没有标签 $y^{(i)}$。

目标是把数据分成 $k$ 个 cluster，使得同一类内部样本尽量相似，不同类之间尽量分开。

### 1.2 核心变量

K-means 中有两个核心变量：

- **Cluster assignment**：$c^{(i)} \in \{1, 2, \dots, k\}$，表示第 $i$ 个样本属于哪个 cluster。
- **Cluster centroids**：$\mu_1, \mu_2, \dots, \mu_k$，其中 $\mu_j \in \mathbb{R}^d$ 表示第 $j$ 个 cluster 的中心。

### 1.3 算法流程

初始化 $\mu_1, \mu_2, \dots, \mu_k$，然后重复两步：

**第一步：Cluster assignment step**

对每个样本 $x^{(i)}$，找到离它最近的中心：

$$c^{(i)} := \arg\min_j \|x^{(i)} - \mu_j\|^2$$

**第二步：Move centroid step**

对每个 cluster，重新计算它的中心：

$$\mu_j := \frac{\sum_{i=1}^{m} \mathbf{1}\{c^{(i)} = j\} \, x^{(i)}}{\sum_{i=1}^{m} \mathbf{1}\{c^{(i)} = j\}}$$

即每个 cluster 的中心更新为该 cluster 内所有点的平均值。

### 1.4 K-means 在优化什么？

K-means 实际上是在最小化 **distortion function**：

$$J(c, \mu) = \sum_{i=1}^{m} \|x^{(i)} - \mu_{c^{(i)}}\|^2$$

其中 $\mu_{c^{(i)}}$ 表示样本 $x^{(i)}$ 所属 cluster 的中心。直觉：每个点都希望离自己所属 cluster 的中心尽量近。

### 1.5 为什么每一步都会让目标函数下降？

K-means 可以看成一种 **coordinate descent**，交替优化 $c$ 和 $\mu$：

- **Assignment step**：固定 $\mu_1, \dots, \mu_k$，对每个样本选择最近的中心——在固定中心的情况下让 $J(c, \mu)$ 关于 $c$ 最小化，$J$ 不会增加。
- **Centroid step**：固定 $c^{(i)}$，把每个 cluster center 更新为该 cluster 内样本的均值——在固定 $c$ 的情况下让 $J(c, \mu)$ 关于 $\mu$ 最小化，$J$ 也不会增加。

因此每轮迭代 $J(c, \mu)$ 单调不增。

### 1.6 为什么 centroid 是均值？

固定 cluster $j$，我们要最小化 $\sum_{i: c^{(i)} = j} \|x^{(i)} - \mu_j\|^2$。对 $\mu_j$ 求导：

$$\frac{\partial}{\partial \mu_j} \sum_{i: c^{(i)} = j} \|x^{(i)} - \mu_j\|^2 = 2 \sum_{i: c^{(i)} = j} (\mu_j - x^{(i)})$$

令导数为 0：

$$\sum_{i: c^{(i)} = j} (\mu_j - x^{(i)}) = 0 \;\Rightarrow\; n_j \mu_j = \sum_{i: c^{(i)} = j} x^{(i)} \;\Rightarrow\; \mu_j = \frac{1}{n_j} \sum_{i: c^{(i)} = j} x^{(i)}$$

这就是该 cluster 的均值。

### 1.7 K-means 的特点

**优点**：简单、速度快、容易实现、适合大规模数据。

**缺点**：

- 需要预先指定 $k$
- 对初始化敏感，容易陷入 local optimum
- 只能得到 hard assignment（每个样本只能属于一个 cluster，没有不确定性）
- 更适合近似球形、大小相近的 cluster
- 对 outlier 敏感

## 2. Gaussian Mixture Models（GMM）

K-means 的问题是只做硬分配，没有概率解释。GMM 则是一个概率模型，假设数据由多个 Gaussian component 混合生成。

### 2.1 生成过程

假设有 $k$ 个 Gaussian component。对每个样本：

1. 先选择它来自哪个 component：$z^{(i)} \sim \text{Multinomial}(\phi)$，其中 $z^{(i)} \in \{1, 2, \dots, k\}$ 是 latent variable。
2. 根据选中的 component 生成样本：$x^{(i)} \mid z^{(i)} = j \sim \mathcal{N}(\mu_j, \Sigma_j)$。

模型可写为 $p(x^{(i)}, z^{(i)}) = p(x^{(i)} \mid z^{(i)}) \, p(z^{(i)})$。

### 2.2 参数

GMM 的参数是 $\phi, \mu, \Sigma$：

- $\phi_j = P(z = j)$：第 $j$ 个 component 的先验概率（mixture weight）。
- $\mu_j$：第 $j$ 个 Gaussian component 的均值。
- $\Sigma_j$：第 $j$ 个 Gaussian component 的协方差矩阵。

### 2.3 为什么直接最大似然不好做？

如果我们能观察到 $z^{(i)}$，参数估计会很简单。但 $z^{(i)}$ 是隐藏变量，likelihood 为：

$$\ell(\phi, \mu, \Sigma) = \sum_{i=1}^{m} \log p(x^{(i)}; \phi, \mu, \Sigma) = \sum_{i=1}^{m} \log \sum_{j=1}^{k} \phi_j \, \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)$$

难点在于 $\log\sum$ 这个结构——如果是 $\sum\log$ 通常好处理，但 $\log\sum$ 让求导和解析解变得困难。EM algorithm 就是为这种 latent variable 问题设计的。

## 3. EM Algorithm

EM（Expectation-Maximization）交替做两步：

- **E-step**：估计隐藏变量 $z$ 的后验分布，即每个点属于每个 component 的概率。
- **M-step**：把这些概率当成 soft weights，重新估计模型参数。

### 3.1 E-step：计算 Responsibilities

定义 responsibility：

$$w_j^{(i)} = P(z^{(i)} = j \mid x^{(i)}; \phi, \mu, \Sigma)$$

它表示第 $j$ 个 Gaussian component 对样本 $x^{(i)}$ "负责"的程度。根据 Bayes rule：

$$w_j^{(i)} = \frac{P(x^{(i)} \mid z^{(i)} = j; \mu_j, \Sigma_j) \, P(z^{(i)} = j; \phi)}{\sum_{\ell=1}^{k} P(x^{(i)} \mid z^{(i)} = \ell; \mu_\ell, \Sigma_\ell) \, P(z^{(i)} = \ell; \phi)} = \frac{\phi_j \, \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)}{\sum_{\ell=1}^{k} \phi_\ell \, \mathcal{N}(x^{(i)}; \mu_\ell, \Sigma_\ell)}$$

并且 $\sum_{j=1}^{k} w_j^{(i)} = 1$。

### 3.2 M-step：更新参数

将 $w_j^{(i)}$ 当作样本 $i$ 属于 cluster $j$ 的 soft assignment，更新参数：

$$\phi_j = \frac{1}{m} \sum_{i=1}^{m} w_j^{(i)}$$

$$\mu_j = \frac{\sum_{i=1}^{m} w_j^{(i)} x^{(i)}}{\sum_{i=1}^{m} w_j^{(i)}}$$

$$\Sigma_j = \frac{\sum_{i=1}^{m} w_j^{(i)} (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\sum_{i=1}^{m} w_j^{(i)}}$$

这三个公式是 note7b 最核心的公式。

## 4. EM 和 K-means 的关系

两者结构非常相似，核心区别在于是 hard 还是 soft assignment：

- **K-means**：$c^{(i)} = \arg\min_j \|x^{(i)} - \mu_j\|^2$，每个样本只属于一个 cluster（hard）。
- **GMM/EM**：$w_j^{(i)} = P(z^{(i)} = j \mid x^{(i)})$，一个点可以以不同概率属于多个 component（soft）。例如 $w_1^{(i)} = 0.7, w_2^{(i)} = 0.3$ 表示样本更可能来自 component 1，但也有一定概率来自 component 2。

可以说 K-means 是 GMM 在 $\Sigma_j \to 0$（或协方差趋于 $\epsilon I$ 且 $\epsilon \to 0$）时的极限情况。

## 5. EM 的一般形式

对于有 hidden variables 的模型，EM 可以写成统一框架：

**E-step**：计算后验 $Q_i(z^{(i)}) = P(z^{(i)} \mid x^{(i)}; \theta)$。

**M-step**：更新 $\theta := \arg\max_\theta \sum_{i=1}^{m} \sum_{z^{(i)}} Q_i(z^{(i)}) \log p(x^{(i)}, z^{(i)}; \theta)$。

（分母 $Q_i(z^{(i)})$ 与 $\theta$ 无关，故可从优化目标中省略。）

> **补充**：关于 EM 为什么能保证 likelihood 单调不降的详细证明——Jensen's inequality 构造 lower bound、E-step 如何让 bound 取等、M-step 如何最大化 bound、以及核心不等式链 $\ell(\theta^{\text{new}}) \geq \mathcal{L}(Q, \theta^{\text{new}}) \geq \mathcal{L}(Q, \theta^{\text{old}}) = \ell(\theta^{\text{old}})$——见 [CS229-8 EM Algorithm](../CS229-8-EM_Algorithm/)。

## 6. 必须掌握的公式

**K-means assignment**
$$c^{(i)} := \arg\min_j \|x^{(i)} - \mu_j\|^2$$

**K-means centroid update**
$$\mu_j := \frac{\sum_{i=1}^{m} \mathbf{1}\{c^{(i)} = j\} \, x^{(i)}}{\sum_{i=1}^{m} \mathbf{1}\{c^{(i)} = j\}}$$

**Distortion function**
$$J(c, \mu) = \sum_{i=1}^{m} \|x^{(i)} - \mu_{c^{(i)}}\|^2$$

**GMM model**
$$z^{(i)} \sim \text{Multinomial}(\phi), \quad x^{(i)} \mid z^{(i)} = j \sim \mathcal{N}(\mu_j, \Sigma_j)$$

**Responsibility**
$$w_j^{(i)} = \frac{\phi_j \, \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)}{\sum_{\ell=1}^{k} \phi_\ell \, \mathcal{N}(x^{(i)}; \mu_\ell, \Sigma_\ell)}$$

**EM update for GMM**
$$\phi_j = \frac{1}{m} \sum_{i=1}^{m} w_j^{(i)}, \quad \mu_j = \frac{\sum_{i=1}^{m} w_j^{(i)} x^{(i)}}{\sum_{i=1}^{m} w_j^{(i)}}, \quad \Sigma_j = \frac{\sum_{i=1}^{m} w_j^{(i)} (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\sum_{i=1}^{m} w_j^{(i)}}$$
