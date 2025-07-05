---
title: 熵、交叉熵与KL散度
categories: 
    - 科研
tags: 
    - 概率
    - 机器学习
---

# 熵
熵（Entropy）是概率论中的一个概念，用于度量概率分布的混乱程度。熵越小，概率分布越均匀，越容易预测。  
熵的计算公式为：
$$ H(X) = - \sum_ {x \in \mathcal{X}} P(x) \log P(x) $$

# 交叉熵
交叉熵是一种度量两个概率分布之间的差异的度量方法。
$$ H(P, Q) = - \sum_ {x \in \mathcal{X}} P(x) \log Q(x) = - \int P(x) \log Q(x) dx $$

# KL散度

KL散度（Kullback - Leibler Divergence），也被称为相对熵，是一种衡量两个概率分布差异的方法。它用于量化两个概率分布之间的距离，可以用来评估一个概率分布与另一个概率分布的相似程度。KL散度是非负的，当两个概率分布完全相同时，其值为0，否则为正数。


对于离散概率分布，KL散度的公式为：

$$
\begin{aligned}
D_{KL}(P \| Q) &= H(P, Q) - H(P)  \\ 
&= - \sum_ {x \in \mathcal{X}} P(x) \log Q(x) + \sum_ {x \in \mathcal{X}} P(x) \log P(x) \\
&= \sum_ {x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}
\end{aligned}
$$

其中：
- P 和 Q 是定义在相同样本空间 $\mathcal{X}$ 上的两个概率分布。
- P(x) 和 Q(x) 分别是概率分布 P 和 Q 在样本 x 处的概率值。

对于连续概率分布，KL散度的公式为：
$$
\begin{aligned}
D_{KL}(P \| Q) &= \int_{-\infty}^{\infty} p(x) \log \frac{p(x)}{q(x)} dx \\
 &= E_{x \in p}[\log p(x) - \log q(x)] 
\end{aligned}
$$
其中：
- p(x) 和 q(x) 分别是概率分布 P 和 Q 的概率密度函数。
- 积分范围是整个样本空间。

KL散度的值越大，表示两个概率分布之间的差异越大；值越小，表示两个概率分布越相似。  
KL散度具有非对称性。