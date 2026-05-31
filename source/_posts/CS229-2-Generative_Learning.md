---
title: CS229-2 Generative Learning
date: 2026-05-31
categories:
    - 科研
tags:
    - CS229
    - 机器学习
mathjax: true
---
Gaussian Discriminant Analysis (GDA) 与 朴素贝叶斯
<!--more-->

## 1. Discriminative vs Generative

### 1.1 Discriminative Learning Algorithm

判别式学习直接学习：

$$p(y|x)$$

或者直接学习映射：

$$h(x)$$

典型例子：Logistic Regression、SVM、Neural Network、Softmax Regression。

它关心的是：**给定特征 $x$，标签 $y$ 是什么？** 即直接学习分类边界。

### 1.2 Generative Learning Algorithm

生成式学习建模的是：

$$p(x|y) \quad \text{和} \quad p(y)$$

然后通过 Bayes rule 得到：

$$p(y|x) = \frac{p(x|y)p(y)}{p(x)}$$

其中：
- $p(y)$ 叫 **class prior**（类别先验概率）
- $p(x|y)$ 表示给定类别 $y$ 时特征 $x$ 的分布

分类时，$p(x)$ 对所有类别相同，因此：

$$\arg\max_y p(y|x) = \arg\max_y p(x|y)p(y)$$

这就是生成式分类器的核心。

## 2. Gaussian Discriminant Analysis (GDA)

### 2.1 GDA Model

GDA 用于分类，假设输入是连续特征 $x \in \mathbb{R}^d$，核心假设是：**每一类内部的特征分布都是 multivariate Gaussian**。

$$y \sim \text{Bernoulli}(\phi)$$
$$x|y=0 \sim \mathcal{N}(\mu_0, \Sigma)$$
$$x|y=1 \sim \mathcal{N}(\mu_1, \Sigma)$$

参数含义：

| 参数 | 含义 |
|------|------|
| $\phi$ | 类别 1 的先验概率，即 $P(y=1)$ |
| $\mu_0$ | 类别 0 的特征均值向量 |
| $\mu_1$ | 类别 1 的特征均值向量 |
| $\Sigma$ | 两类共享的协方差矩阵 |

### 2.2 Multivariate Gaussian

多元高斯分布记为 $\mathcal{N}(\mu, \Sigma)$，其中：
- $\mu \in \mathbb{R}^d$ 是均值向量
- $\Sigma \in \mathbb{R}^{d \times d}$ 是协方差矩阵

概率密度函数：

$$p(x;\mu,\Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$$

其中最关键的量是 **Mahalanobis distance**：

$$(x-\mu)^T\Sigma^{-1}(x-\mu)$$

它衡量样本 $x$ 到均值 $\mu$ 的"协方差归一化距离"。当 $\Sigma = I$ 时退化为欧氏距离 $\|x-\mu\|_2^2$。

### 2.3 Prediction

训练得到 $\phi, \mu_0, \mu_1, \Sigma$ 后，对新样本 $x$ 计算：

$$p(y=1|x) = \frac{p(x|y=1)\phi}{p(x|y=1)\phi + p(x|y=0)(1-\phi)}$$

预测规则：

$$\hat{y} = \arg\max_y p(x|y)p(y)$$

若 $p(x|y=1)\phi > p(x|y=0)(1-\phi)$ 则预测为 1，否则预测为 0。

### 2.4 GDA vs Logistic Regression

在 GDA 的假设下，可以推导出后验概率具有 sigmoid 形式：

$$p(y=1|x) = \frac{1}{1 + \exp(-\theta^T x)}$$

因此 GDA 和 Logistic Regression 的决策边界都是线性的。但二者并不等价：

| 模型 | 学什么 | 假设强度 | 特点 |
|------|--------|----------|------|
| Logistic Regression | 直接学 $p(y\|x)$ | 较弱 | MLE on $p(y^{(i)}\|x^{(i)})$ |
| GDA | 学 $p(x\|y)$ 和 $p(y)$ | 更强（假设 Gaussian） | MLE on $p(x^{(i)},y^{(i)})$ |

**结论：**
- 若 Gaussian 假设成立，GDA 数据效率更高，能更快收敛到好的分类器
- 若 Gaussian 假设不成立，Logistic Regression 更 robust
- 实践中数据量足够大时，Logistic Regression 通常是更安全的选择

## 3. Naive Bayes

### 3.1 Motivation & Assumption

GDA 适合连续特征。Naive Bayes 适合**离散特征**，常用于文本分类（spam classification、sentiment analysis、document classification）。

Naive Bayes 也是生成式模型，建模 $p(x|y)$ 和 $p(y)$，但它做了一个强假设：

**给定类别 $y$ 后，各特征 $x_j$ 条件独立。**

即 Naive Bayes (conditional independence) assumption：

$$p(x_1, x_2, \ldots, x_n | y) = \prod_{j=1}^n p(x_j | y)$$

"Naive" 的含义：现实中特征通常不独立，但这个假设极大简化了建模和计算。

### 3.2 Prediction Formula

由 Bayes rule 和条件独立假设：

$$p(y|x) \propto p(y) \prod_{j=1}^n p(x_j | y)$$

分类时：

$$\hat{y} = \arg\max_y p(y) \prod_{j=1}^n p(x_j | y)$$

实际计算取 log 避免下溢：

$$\hat{y} = \arg\max_y \left[\log p(y) + \sum_{j=1}^n \log p(x_j | y)\right]$$

### 3.3 Parameter Estimation

以二分类、离散特征为例，使用 MLE（本质上就是数频率）。

类别先验 $\phi_y = P(y=1)$：

$$\phi_y = \frac{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}}{m}$$

特征条件概率 $\phi_{j|y=1} = P(x_j = 1 | y = 1)$：

$$\phi_{j|y=1} = \frac{\sum_{i=1}^m \mathbf{1}\{x_j^{(i)} = 1 \land y^{(i)} = 1\}}{\sum_{i=1}^m \mathbf{1}\{y^{(i)} = 1\}}$$

类似地，$\phi_{j|y=0}$ 在 $y=0$ 的样本中统计即可。

### 3.4 Laplace Smoothing

**为什么需要平滑？** 若某个词在训练集中从未在一类中出现，$P(\text{word}=k|y=c)=0$，则整篇文档的概率会因为连乘变为 0。训练集中没出现不代表真实概率为 0。

对一个取 $k$ 个值的离散变量，原 MLE 为：

$$\phi_j = \frac{\sum_{i=1}^m \mathbf{1}\{z^{(i)} = j\}}{m}$$

Laplace smoothing 给每个类别"假装多出现一次"：

$$\phi_j = \frac{1 + \sum_{i=1}^m \mathbf{1}\{z^{(i)} = j\}}{m + k}$$

## 4. Event Models for Text Classification

核心问题：一篇文档如何表示成特征？有两种常见思路。

### 4.1 Multi-variate Bernoulli Event Model

只关心**某个词是否出现过**。特征是二值向量 $x_j \in \{0,1\}$，$x_j = 1$ 表示词 $j$ 在文档中出现过。此模型**忽略词频**——出现 1 次和 10 次等价。

### 4.2 Multinomial Event Model

关心**文档中每个位置出现了哪个词**（等价于关心词频）。设文档长度 $n_i$，每个位置的词来自词表 $\{1, 2, \ldots, |V|\}$。

模型假设：

$$p(x|y) = \prod_{j=1}^{n_i} p(x_j | y)$$

分类时（取 log）：

$$\hat{y} = \arg\max_y \left[\log p(y) + \sum_{j=1}^{n_i} \log p(x_j | y)\right]$$

Multinomial 模型通常比 Bernoulli 更适合文本分类。

### 4.3 Parameter Estimation

对于类别 $y=c$ 下词 $k$ 的概率 $\phi_{k|y=c} = P(\text{word}=k | y=c)$，加入 Laplace smoothing 后：

$$\phi_{k|y=c} = \frac{1 + \sum_{i=1}^m \sum_{j=1}^{n_i} \mathbf{1}\{x_j^{(i)} = k \land y^{(i)} = c\}}{|V| + \sum_{i=1}^m \mathbf{1}\{y^{(i)} = c\} \cdot n_i}$$

- 分子：类别 $c$ 的所有文档中词 $k$ 出现次数 + 1
- 分母：类别 $c$ 的所有文档的总词数 + 词表大小 $|V|$
