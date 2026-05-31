---
title: CS229-1 Supervised Learning
date: 2026-05-30
categories:
    - 科研
tags:
    - CS229
    - 机器学习
mathjax: true
---
从Linear Regression开始，到线性回归的Normal Equation与Probabilistic Interpretation；再引入Logistic Regression与Softmax Regression；最后使用Generalized Linear Models统一这三种回归。
<!--more-->
## 1. Supervised Learning 基本设定

监督学习中，我们有训练集：

$$
\{(x^{(i)}, y^{(i)})\}_{i=1}^{m}
$$

其中：

- $x^{(i)}$ 是第 $i$ 个样本的输入，也叫 features / input variables；
- $y^{(i)}$ 是第 $i$ 个样本的输出，也叫 target / label / output variable。

模型要学习一个函数：

$$h: \mathcal{X} \to \mathcal{Y}$$

这个函数叫 **hypothesis**（假设函数）。

目标是：给定新的输入 $x$，希望预测：

$$h(x) \approx y$$

---

## 2. Linear Regression（线性回归）

### 2.1 模型形式

线性回归假设输出是输入特征的线性组合：

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$$

也就是：

$$h_\theta(x) = \theta^T x$$

其中 $\theta$ 是模型参数，也叫 parameters。

线性回归的本质是：**找到一个超平面，使它尽可能拟合训练数据。** 在二维中是拟合一条线；在高维中是拟合一个 hyperplane。

### 2.2 代价函数

线性回归使用平方误差：

$$J(\theta) = \frac{1}{2} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$$

这里的 $\frac{1}{2}$ 是为了求导时抵消平方的系数，没有本质影响。

目标是：

$$\min_\theta J(\theta)$$

也就是寻找参数 $\theta$，让预测值和真实值之间的平方误差最小。

### 2.3 LMS Algorithm（最小均方算法）

LMS 是 Least Mean Squares 的缩写。它其实就是对平方误差做梯度下降。

对于单个样本：

$$J(\theta) = \frac{1}{2} \left( h_\theta(x) - y \right)^2$$

对参数 $\theta_j$ 求导：

$$\frac{\partial J(\theta)}{\partial \theta_j} = (h_\theta(x) - y) \, x_j$$

梯度下降更新：

$$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

代入得到：

$$\theta_j := \theta_j + \alpha \, (y - h_\theta(x)) \, x_j$$

这就是 **LMS update rule**。

---

## 3. Normal Equation（正规方程）

线性回归也可以不用迭代，直接求解析解。

把所有样本堆成设计矩阵：

$$
X =
\begin{bmatrix}
(x^{(1)})^T \\
(x^{(2)})^T \\
\vdots \\
(x^{(m)})^T
\end{bmatrix}
$$

标签写成：

$$
y =
\begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(m)}
\end{bmatrix}
$$

模型预测为 $X\theta$。目标函数：

$$J(\theta) = \frac{1}{2} (X\theta - y)^T (X\theta - y)$$

对 $\theta$ 求导并令梯度为零：

$$\nabla_\theta J(\theta) = X^T X \theta - X^T y = 0$$

得到：

$$X^T X \theta = X^T y$$

所以：

$$\theta = (X^T X)^{-1} X^T y$$

这就是 **normal equation**。

### 几何意义

正规方程对应的是最小二乘问题：

$$\min_\theta \|X\theta - y\|_2^2$$

几何意义是：在 $X$ 的列空间中，找到一个点 $X\theta$，使它最接近 $y$。也就是说：**$X\theta$ 是 $y$ 在 $\text{Col}(X)$ 上的正交投影**。

---

## 4. Probabilistic Interpretation

假设真实标签由下面的过程生成：

$$y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$$

其中噪声满足高斯分布：

$$\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$$

那么：

$$y^{(i)} \mid x^{(i)}; \theta \sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2)$$

所以条件概率为：

$$p(y^{(i)} \mid x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)$$

整个数据集的 likelihood 是：

$$L(\theta) = \prod_{i=1}^{m} p(y^{(i)} \mid x^{(i)}; \theta)$$

取 log 得到 log-likelihood：

$$\ell(\theta) = \log L(\theta)$$

最大化 log-likelihood 等价于最小化：

$$\sum_{i=1}^{m} \left( y^{(i)} - \theta^T x^{(i)} \right)^2$$

> **结论：Least Squares 可以从 Gaussian noise assumption + Maximum Likelihood Estimation 推出来。**

---

## 5. Locally Weighted Linear Regression（局部加权线性回归）

普通线性回归是全局拟合一个参数 $\theta$。但如果数据本身不是全局线性的，可以在预测某个点 $x$ 时，更重视附近的训练样本。

局部加权线性回归的目标函数是：

$$\min_\theta \sum_{i=1}^{m} w^{(i)} \left( y^{(i)} - \theta^T x^{(i)} \right)^2$$

其中权重通常取：

$$w^{(i)} = \exp\left( -\frac{(x^{(i)} - x)^2}{2\tau^2} \right)$$

如果是多维输入，可以写成：

$$w^{(i)} = \exp\left( -\frac{(x^{(i)} - x)^T (x^{(i)} - x)}{2\tau^2} \right)$$

其中 $\tau$ 叫 **bandwidth parameter**：

| $\tau$ 很小 | $\tau$ 很大 |
|:---:|:---:|
| 模型非常局部 | 模型更接近普通线性回归 |
| 容易过拟合 | 更加平滑 |

---

## 6. Classification and Logistic Regression（分类与逻辑回归）

接下来从回归转到分类。

二分类任务中：

$$y \in \{0, 1\}$$

- $0$ = negative class
- $1$ = positive class

如果直接用线性回归做分类，会有一个问题：$h_\theta(x) = \theta^T x$ 可能小于 $0$ 或大于 $1$，不能解释成概率。因此引入 **sigmoid function**。

### 6.1 Sigmoid / Logistic Function

定义：

$$g(z) = \frac{1}{1 + e^{-z}}$$

于是 logistic regression 的假设函数是：

$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

它的输出范围是：

$$0 < h_\theta(x) < 1$$

可以解释为：

$$h_\theta(x) = P(y = 1 \mid x; \theta)$$

因此：

$$
\begin{aligned}
P(y = 1 \mid x; \theta) &= h_\theta(x) \\
P(y = 0 \mid x; \theta) &= 1 - h_\theta(x)
\end{aligned}
$$

可以合并写成：

$$P(y \mid x; \theta) = h_\theta(x)^y \, (1 - h_\theta(x))^{1-y}$$

### 6.2 似然函数

给定训练集，likelihood 是：

$$L(\theta) = \prod_{i=1}^{m} h_\theta(x^{(i)})^{y^{(i)}} \, \left(1 - h_\theta(x^{(i)})\right)^{1 - y^{(i)}}$$

log-likelihood 是：

$$\ell(\theta) = \sum_{i=1}^{m} \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log\left(1 - h_\theta(x^{(i)})\right) \right]$$

我们最大化这个 log-likelihood。

等价地，也可以最小化 negative log-likelihood，也就是 **binary cross entropy**：

$$J(\theta) = -\sum_{i=1}^{m} \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log\left(1 - h_\theta(x^{(i)})\right) \right]$$

### 6.3 梯度

一个非常重要的结论是：

$$\frac{\partial \ell(\theta)}{\partial \theta_j} = \sum_{i=1}^{m} \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$$

所以梯度上升更新为：

$$\theta_j := \theta_j + \alpha \sum_{i=1}^{m} \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$$

如果是单样本 SGD：

$$\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$$

> **注意它和线性回归 LMS 的形式完全一样：**
>
> $$\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$$
>
> 但是含义不同：
>
> | | Linear Regression | Logistic Regression |
> |:---|:---|:---|
> | $h_\theta(x)$ | $\theta^T x$ | $\text{sigmoid}(\theta^T x)$ |
> | 来源 | Gaussian likelihood | Bernoulli likelihood |

---

## 7. Generalized Linear Models（广义线性模型）

### 7.1 Exponential Family（指数族分布）

如果一个分布可以写成：

$$p(y; \eta) = b(y) \exp\left( \eta^T T(y) - a(\eta) \right)$$

那么它属于 **exponential family**。

- $\eta$ = natural parameter（自然参数）
- $T(y)$ = sufficient statistic（充分统计量）
- $a(\eta)$ = log partition function（对数配分函数）
- $b(y)$ = base measure（基础测度）

### 7.2 GLM 的三个假设

构造 GLM 时通常有三个 assumptions：

**Assumption 1：** 给定 $x$ 和参数 $\theta$，$y \mid x; \theta$ 服从某个 exponential family 分布：

$$y \mid x; \theta \sim \text{ExponentialFamily}(\eta)$$

**Assumption 2：** 预测目标是条件期望：

$$h_\theta(x) = \mathbb{E}[y \mid x; \theta]$$

**Assumption 3：** natural parameter $\eta$ 和输入 $x$ 线性相关：

$$\eta = \theta^T x$$

这三个假设非常关键。

### 7.3 Logistic Regression 是 GLM 的特例

对于二分类：$y \in \{0, 1\}$，使用 Bernoulli distribution。

Bernoulli 分布可以写成指数族形式，并且它的 natural parameter 满足：

$$\eta = \log\frac{\phi}{1 - \phi}$$

反过来：

$$\phi = \frac{1}{1 + e^{-\eta}}$$

由于 GLM 假设 $\eta = \theta^T x$，所以：

$$\phi = \frac{1}{1 + e^{-\theta^T x}}$$

这正是 logistic regression。

> **所以 sigmoid 不是随便拍脑袋选出来的，而是：Bernoulli distribution + GLM assumptions 自然推出的 inverse link function。**

### 7.4 Linear Regression 是 GLM 的特例

对于连续值回归，假设：

$$y \mid x; \theta \sim \mathcal{N}(\mu, \sigma^2)$$

在 Gaussian 的指数族形式中，预测均值为 $\mu = \eta$。

而 GLM 假设 $\eta = \theta^T x$，所以：

$$h_\theta(x) = \theta^T x$$

这就是 linear regression。

---

## 8. Softmax Regression

对于多分类：

$$y \in \{1, 2, \dots, k\}$$

使用 multinomial distribution。Softmax regression 的输出是：

$$P(y = i \mid x; \theta) = \frac{\exp(\theta_i^T x)}{\sum_{j=1}^{k} \exp(\theta_j^T x)}$$

它是 logistic regression 在多分类场景下的推广。

也可以写成向量形式：

$$
h_\theta(x) =
\begin{bmatrix}
P(y = 1 \mid x; \theta) \\
P(y = 2 \mid x; \theta) \\
\vdots \\
P(y = k \mid x; \theta)
\end{bmatrix}
$$

Softmax 的核心作用是把任意实数 logits 变成概率分布：

$$\sum_{i=1}^{k} P(y = i \mid x; \theta) = 1$$

并且：

$$P(y = i \mid x; \theta) > 0$$

---

> **GLM 的核心思路：** 先把输出分布写成 exponential family，找出它的 natural parameter $\eta$，然后假设 $\eta = \theta^T x$。再利用 $h_\theta(x) = \mathbb{E}[y \mid x]$，就能推出对应的 regression model。
