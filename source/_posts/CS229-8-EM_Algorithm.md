---
title: CS229-8 The Expectation-Maximization Algorithm
date: 2026-06-07
categories:
    - 科研
tags:
    - CS229
    - 机器学习
mathjax: true
---

本节详细讲解 EM 算法的原理与证明。

<!--more-->

## 1. EM 要解决的问题

很多模型里，我们观察到的数据只有 $x^{(i)}$，但模型内部还存在一些看不见的 latent / hidden / unobserved variables $z^{(i)}$。

比如在 GMM 中，$z^{(i)}$ 表示第 $i$ 个样本来自哪个 Gaussian component。

如果我们能观察到完整数据 $(x^{(i)}, z^{(i)})$，最大似然会很简单。但现实中只能看到 $x^{(i)}$，所以 log-likelihood 是：

$$\ell(\theta) = \sum_{i=1}^{m} \log p(x^{(i)}; \theta) = \sum_{i=1}^{m} \log \sum_{z^{(i)}} p(x^{(i)}, z^{(i)}; \theta)$$

难点就在 $\log\sum$ 这个结构——它通常让解析优化变得很难。如果能知道 $z^{(i)}$，就会变成 $\sum\log p(x^{(i)}, z^{(i)}; \theta)$（complete-data log-likelihood），通常好优化很多。

**EM 的核心思想**：hidden variable 看不见，那就先估计它的分布，再用这个估计来更新参数。

## 2. Jensen's Inequality：EM 的数学基础

如果函数 $f$ 是 concave 函数（如 $f(x) = \log x$），那么：

$$f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$$

对于 $\log$ 函数，就是 $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$。

EM 之所以能构造 lower bound，正是因为 $\log$ 是 concave function。

## 3. 构造 Lower Bound

对第 $i$ 个样本，引入一个任意分布 $Q_i(z^{(i)})$（满足 $Q_i(z^{(i)}) \geq 0$ 且 $\sum_z Q_i(z^{(i)}) = 1$），做"乘除同一个东西"的变形：

$$
\log p(x^{(i)}; \theta) = \log \sum_{z^{(i)}} p(x^{(i)}, z^{(i)}; \theta)
= \log \sum_{z^{(i)}} Q_i(z^{(i)}) \cdot \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}
$$

右边可以看成期望 $\mathbb{E}_{z^{(i)} \sim Q_i}\!\left[\frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}\right]$ 取 $\log$。由 Jensen：

$$
\log p(x^{(i)}; \theta) \geq \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}
$$

对所有样本求和，得到 log-likelihood 的 **lower bound**：

$$
\ell(\theta) \geq \sum_{i=1}^{m} \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} \triangleq \mathcal{L}(\theta, Q)
$$

对于任意合法的 $Q_i$，都有 $\ell(\theta) \geq \mathcal{L}(\theta, Q)$。

## 4. E-step：让 Lower Bound 贴紧

E-step 不是随便选 $Q_i$，而是取当前参数下 hidden variable 的后验分布：

$$Q_i(z^{(i)}) = p(z^{(i)} \mid x^{(i)}; \theta^{\text{old}})$$

这一步的效果是：让 lower bound 在当前参数 $\theta^{\text{old}}$ 处和真实 likelihood **相等**：

$$\mathcal{L}(\theta^{\text{old}}, Q) = \ell(\theta^{\text{old}})$$

在 GMM 中，E-step 就是计算 responsibility：

$$w_j^{(i)} = p(z^{(i)} = j \mid x^{(i)}; \theta^{\text{old}}) = \frac{\phi_j \, \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)}{\sum_{\ell=1}^{k} \phi_\ell \, \mathcal{N}(x^{(i)}; \mu_\ell, \Sigma_\ell)}$$

## 5. M-step：最大化 Lower Bound

M-step 固定 $Q_i$（即 E-step 得到的后验），优化 $\theta$ 以最大化 $\mathcal{L}(\theta, Q)$：

$$\theta^{\text{new}} = \arg\max_\theta \mathcal{L}(\theta, Q)$$

由于 $\mathcal{L}(\theta, Q) = \sum_i \sum_z Q_i(z) \log \frac{p(x^{(i)}, z; \theta)}{Q_i(z)}$，而 $Q_i(z)$ 在 M-step 中固定（不依赖 $\theta$），所以等价于最大化：

$$\theta^{\text{new}} = \arg\max_\theta \sum_{i=1}^{m} \sum_{z^{(i)}} Q_i(z^{(i)}) \log p(x^{(i)}, z^{(i)}; \theta)$$

这就是 **weighted complete-data log-likelihood**。在 GMM 中，M-step 更新 $\phi_j, \mu_j, \Sigma_j$ 的公式正是由此导出。

## 6. EM 为什么能保证 Likelihood 不下降？

这是 note8 的核心结论。三条不等式串起来：

$$
\ell(\theta^{\text{new}}) \;\geq\; \mathcal{L}(\theta^{\text{new}}, Q) \;\geq\; \mathcal{L}(\theta^{\text{old}}, Q) \;=\; \ell(\theta^{\text{old}})
$$

| 步骤 | 不等式 | 依据 |
|---|---|---|
| ① | $\ell(\theta^{\text{new}}) \geq \mathcal{L}(\theta^{\text{new}}, Q)$ | Jensen：lower bound 永远 ≤ 真实 likelihood |
| ② | $\mathcal{L}(\theta^{\text{new}}, Q) \geq \mathcal{L}(\theta^{\text{old}}, Q)$ | M-step：$\theta^{\text{new}}$ 在固定 $Q$ 下最大化 $\mathcal{L}$ |
| ③ | $\mathcal{L}(\theta^{\text{old}}, Q) = \ell(\theta^{\text{old}})$ | E-step：选 $Q_i = p(z \mid x; \theta^{\text{old}})$ 使等号成立 |

因此 $\ell(\theta^{\text{new}}) \geq \ell(\theta^{\text{old}})$，likelihood 单调不降。

> 要背的不是复杂推导，而是这条链：$\ell(\theta^{\text{new}}) \geq \mathcal{L}(\theta^{\text{new}}, Q) \geq \mathcal{L}(\theta^{\text{old}}, Q) = \ell(\theta^{\text{old}})$。

## 7. E-step 为什么能让 Bound 取等号？

Jensen's inequality 取等号的条件是：被求期望的随机变量为常数。

在 EM 中，这个随机变量是 $\frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}$。如果选择：

$$Q_i(z^{(i)}) = p(z^{(i)} \mid x^{(i)}; \theta)$$

那么：

$$\frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} = \frac{p(x^{(i)}, z^{(i)}; \theta)}{p(z^{(i)} \mid x^{(i)}; \theta)} = p(x^{(i)}; \theta)$$

它与 $z^{(i)}$ 无关，是常数，因此 Jensen 取等号。这就是为什么 E-step 必须选择后验分布，而不是随便选一个分布。

## 8. EM 与 KL Divergence / ELBO 的关系

这部分在 VAE 和 variational inference 中很重要。lower bound 在现代深度生成模型里常叫 **ELBO**（Evidence Lower Bound）：

$$\mathcal{L}(\theta, Q) = \sum_i \sum_z Q_i(z) \log \frac{p(x^{(i)}, z; \theta)}{Q_i(z)}$$

并且有关系：

$$\ell(\theta) = \mathcal{L}(\theta, Q) + \sum_i D_{\text{KL}}(Q_i(z) \parallel p(z \mid x^{(i)}; \theta))$$

因为 KL divergence 非负（$D_{\text{KL}} \geq 0$），所以 $\ell(\theta) \geq \mathcal{L}(\theta, Q)$。

E-step 做的是让 $Q_i(z) = p(z \mid x^{(i)}; \theta^{\text{old}})$，于是 KL divergence 变为 0，lower bound 贴住 true log-likelihood。M-step 则最大化 ELBO。

> CS229 速通阶段记住三点即可：EM 的 lower bound = ELBO；E-step 让 KL = 0；M-step 最大化 ELBO。

## 9. EM 的一般算法模板

初始化参数 $\theta^{(0)}$，重复直到收敛：

**E-step**：对每个样本计算 $Q_i(z^{(i)}) = p(z^{(i)} \mid x^{(i)}; \theta^{(t)})$。

**M-step**：更新 $\theta^{(t+1)} = \arg\max_\theta \sum_{i=1}^{m} \sum_{z^{(i)}} Q_i(z^{(i)}) \log p(x^{(i)}, z^{(i)}; \theta)$。

停止条件：$|\ell(\theta^{(t+1)}) - \ell(\theta^{(t)})| < \epsilon$。

## 10. EM 不能保证什么？

EM 保证 $\ell(\theta)$ 单调不降，但**不保证找到 global maximum**。因为很多 latent variable model 的 likelihood 是非凸的，EM 通常只保证收敛到 local maximum / stationary point / saddle point。

这也是为什么 GMM 对初始化很敏感——不同初始化可能得到不同结果。常见初始化方法：

- random initialization
- 用 K-means 初始化 GMM 的 $\mu$
- 多次随机初始化后选 likelihood 最大的结果

## 11. 与 K-means / GMM 的关系

note8 的 EM 是一般理论，GMM 是一个具体例子：

| 概念 | EM 一般形式 | GMM 具体形式 |
|---|---|---|
| 隐变量 | $z^{(i)}$ | 样本属于哪个 Gaussian component |
| E-step | $Q_i(z) = p(z \mid x; \theta)$ | $w_j^{(i)} = p(z^{(i)} = j \mid x^{(i)}; \theta)$ |
| M-step | 最大化 weighted complete-data log-likelihood | 用 $w_j^{(i)}$ 加权更新 $\phi_j, \mu_j, \Sigma_j$ |

K-means 可看作 GMM/EM 的极限情况（hard assignment vs soft assignment）。

---

**一句话总结**：EM 的本质是——看不见隐藏变量，就先用当前模型估计它的后验分布（E-step），然后在这个估计基础上重新估计参数（M-step）。数学上，通过交替"贴紧"和"抬高" likelihood 的 lower bound，保证 log-likelihood 单调不降。
