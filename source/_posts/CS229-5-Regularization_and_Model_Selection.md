---
title: CS229-5 Regularization and Model Selection
date: 2026-06-01
categories:
    - 科研
tags:
    - CS229
    - 机器学习
mathjax: true
---

本文整理 CS229 课程中正则化与模型选择的核心内容，涵盖交叉验证、L1/L2 正则化、特征选择方法（wrapper / filter），以及从贝叶斯先验推导正则化的视角。

<!--more-->

## 1. Model Selection

### Train / Validation / Test Split

| 数据集 | 作用 |
|---|---|
| training set | 用来训练每个候选模型的参数 |
| validation set / dev set | 用来选择模型结构和超参数 |
| test set | 最后只用一次，用来估计最终模型的泛化性能 |

### Cross Validation

如果数据量较小，单独划出 validation set 会让 training set 变小，导致训练不充分。这时可以用 cross validation。

#### Hold-out Validation

最简单的方法：随机拿出一部分数据作为 validation set，剩下作为 training set。

#### k-fold Cross Validation

更常用的是 k-fold cross validation。

**步骤：**

1. 把训练数据分成 $k$ 份：$S_1, S_2, \ldots, S_k$
2. 每次拿其中一份作为 validation set，剩下 $k-1$ 份作为 training set
3. 训练 $k$ 次，得到 $k$ 个 validation errors
4. 平均这些误差作为该模型的 cross validation error：

$$\hat{\varepsilon}_{\text{CV}}(M) = \frac{1}{k} \sum_{j=1}^{k} \hat{\varepsilon}_{\text{val}}^{(j)}(M)$$

然后选择最优模型：

$$M^* = \arg\min_M \hat{\varepsilon}_{\text{CV}}(M)$$

**优点：** 每个样本都被用作 validation 一次，也大多时候参与 training，评估比单次 hold-out 更稳定。

**常见取值：** $k=5$ 或 $k=10$。

#### Leave-one-out Cross Validation (LOOCV)

当 $k=m$ 时，就是 leave-one-out cross validation。即每次只留一个样本作为 validation，其余 $m-1$ 个样本训练。

- 需要训练 $m$ 次，计算代价大
- 估计方差可能较高，但训练数据利用率较高

## 2. Regularization

### 2.1 正则化要解决什么？

正则化的目标：在训练误差之外，额外惩罚模型复杂度，从而缓解 overfitting。

普通经验风险最小化：

$$\min_\theta J(\theta)$$

加入正则化后：

$$\min_\theta J(\theta) + \lambda R(\theta)$$

其中 $R(\theta)$ 是 regularization term，$\lambda$ 是 regularization strength。

### 2.2 L2 Regularization / Ridge Regression

L2 正则使用 $R(\theta) = \|\theta\|_2^2$，目标函数为：

$$\min_\theta J(\theta) + \lambda \|\theta\|_2^2$$

对于线性回归，$J(\theta) = \sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2$，加入 L2 正则后：

$$\min_\theta \sum_{i=1}^{m} (y^{(i)} - \theta^T x^{(i)})^2 + \lambda \|\theta\|_2^2$$

这也叫 **ridge regression**。

**L2 正则的效果：** 鼓励参数整体变小、使模型更平滑、降低 variance。通常不会让参数严格变成 0。

### 2.3 L2 正则的解析解

普通最小二乘解：$\theta = (X^T X)^{-1} X^T y$。

加入 L2 正则后，目标写为：

$$J(\theta) = \frac{1}{2} \|X\theta - y\|_2^2 + \frac{\lambda}{2} \|\theta\|_2^2$$

对 $\theta$ 求导并令梯度为 0：

$$\nabla_\theta J = X^T (X\theta - y) + \lambda \theta = 0$$

$$X^T X\theta - X^T y + \lambda \theta = 0$$

$$(X^T X + \lambda I)\theta = X^T y$$

得到 ridge regression 的解析解：

$$\theta = (X^T X + \lambda I)^{-1} X^T y$$

**为什么 L2 正则能防止过拟合？** 如果模型想拟合训练集中的噪声，通常需要比较大的参数。L2 正则通过惩罚 $\|\theta\|_2^2$ 限制参数变大，从而抑制过复杂的函数。

- 几何上：L2 把可行参数限制在一个球形区域内
- 优化上：加上 $\lambda I$ 后矩阵更容易可逆，缓解 $X^T X$ 不可逆或病态的问题

### 2.4 L1 Regularization / Lasso

L1 正则使用 $R(\theta) = \|\theta\|_1 = \sum_j |\theta_j|$，目标函数为：

$$\min_\theta J(\theta) + \lambda \|\theta\|_1$$

在线性回归中，这叫 **lasso regression**。

**L1 的效果：** 鼓励稀疏解、会让很多参数直接变成 0、可用于 feature selection。

这与 L2 不同——L2 通常只是让参数变小但不会精确等于 0，而 L1 产生 sparse parameters。

**为什么 L1 会产生稀疏解？** 直觉上：

- L1 约束对应**菱形区域**：$\|\theta\|_1 \leq t$
- L2 约束对应**圆形区域**：$\|\theta\|_2^2 \leq t$

优化目标的等高线与 L1 菱形区域相切时，更容易切在坐标轴上（即某些参数为 0）。因此：L1 → sparse solution，L2 → shrinkage but not sparse。

## 3. Feature Selection

### 3.1 为什么需要特征选择？

特征太多可能导致模型复杂度高、容易 overfit、训练成本高、解释性差。特征选择的目标是自动选出一小部分有用特征。

### 3.2 Wrapper Model Feature Selection

核心思想：用模型性能作为特征子集的评价标准。对于每个特征子集，训练模型并计算 validation error。

但 $n$ 个特征有 $2^n$ 个子集，枚举通常不可行，因此需要贪心方法。

#### Forward Search

1. 从空特征集开始：$F = \emptyset$
2. 每次尝试加入一个当前不在 $F$ 中的特征
3. 选择使 validation error 降低最多的那个特征加入
4. 重复，直到性能不再明显提升或达到预设特征数量

简记：**start with no features, greedily add the best feature.**

- 优点：比枚举所有子集便宜很多
- 缺点：贪心，可能不是全局最优；早期选错后面难以纠正

#### Backward Search

1. 从所有特征开始：$F = \{1, 2, \ldots, n\}$
2. 每次尝试删除一个特征
3. 删除后若 validation error 最小，则真正删除它
4. 重复，直到性能下降明显或达到目标特征数

简记：**start with all features, greedily remove the least useful feature.**

- 优点：考虑特征间交互更多
- 缺点：特征数很大时初始训练成本高；也是贪心，不能保证全局最优

### 3.3 Filter Feature Selection

Filter method 不反复训练模型，而是直接用统计指标给特征打分。常见思路：计算每个特征和标签之间的相关性（如 mutual information、correlation、chi-square score），选择得分最高的若干特征。CS229 中强调 mutual information。

#### Mutual Information

互信息衡量两个变量之间的依赖程度：

$$MI(X, Y) = \sum_x \sum_y p(x, y) \log \frac{p(x, y)}{p(x) p(y)}$$

若 $X$ 和 $Y$ 独立，则 $p(x, y) = p(x) p(y)$，故 $MI(X, Y) = 0$。若 $X$ 对 $Y$ 有信息，互信息就大于 0。

特征选择时，计算 $MI(X_j, Y)$ 并选择互信息最大的特征。直觉：一个特征越能减少对标签的不确定性，它越有用。

## 4. Bayesian Statistics and Regularization

这部分解释了 regularization 不只是工程技巧，也可以从 Bayesian prior 推导出来。

### 4.1 Frequentist vs Bayesian

- **Frequentist view**：参数 $\theta$ 是固定但未知的量，数据是随机的，我们通过数据估计参数
- **Bayesian view**：参数 $\theta$ 也被视为随机变量，我们可以给它设置一个 prior $p(\theta)$

看到数据后，根据 Bayes rule 得到 posterior：

$$p(\theta \mid D) = \frac{p(D \mid \theta) \, p(\theta)}{p(D)}$$

其中 $p(D \mid \theta)$ 为 likelihood，$p(\theta)$ 为 prior，$p(\theta \mid D)$ 为 posterior。

### 4.2 MLE vs MAP

MLE（maximum likelihood estimation）：

$$\theta_{\text{MLE}} = \arg\max_\theta p(D \mid \theta)$$

Bayesian view 中加入 prior 得到 MAP（maximum a posteriori）：

$$\theta_{\text{MAP}} = \arg\max_\theta p(\theta \mid D)$$

由 Bayes rule，$p(\theta \mid D) \propto p(D \mid \theta) \, p(\theta)$，所以：

$$\theta_{\text{MAP}} = \arg\max_\theta p(D \mid \theta) \, p(\theta)$$

取 log：

$$\theta_{\text{MAP}} = \arg\max_\theta \big[ \log p(D \mid \theta) + \log p(\theta) \big]$$

等价于最小化 $-\log p(D \mid \theta) - \log p(\theta)$，其中 $-\log p(\theta)$ 就是正则项。

### 4.3 Gaussian Prior → L2 Regularization

假设参数先验为高斯分布 $\theta_j \sim \mathcal{N}(0, \tau^2)$，则：

$$p(\theta) \propto \exp\!\left(-\frac{1}{2\tau^2} \|\theta\|_2^2\right)$$

取 log 得 $\log p(\theta) = -\frac{1}{2\tau^2} \|\theta\|_2^2 + C$，MAP 目标中出现 $\|\theta\|_2^2$，即 L2 regularization。

因此：**Gaussian prior → MAP estimation → L2 regularization。**

### 4.4 Laplace Prior → L1 Regularization

假设参数先验为 Laplace distribution：

$$p(\theta_j) \propto \exp(-\lambda |\theta_j|)$$

取 log 得 $\log p(\theta) = -\lambda \|\theta\|_1 + C$，MAP 目标中出现 $\|\theta\|_1$，即 L1 regularization。

因此：**Laplace prior → MAP estimation → L1 regularization。**

---

## 关键公式速查

| 名称 | 公式 |
|---|---|
| Model Selection | $M^* = \arg\min_{M_i} \hat{\varepsilon}_{\text{val}}(M_i)$ |
| k-fold CV | $\hat{\varepsilon}_{\text{CV}}(M) = \frac{1}{k} \sum_{j=1}^{k} \hat{\varepsilon}_{\text{val}}^{(j)}(M)$ |
| Regularized Objective | $\min_\theta J(\theta) + \lambda R(\theta)$ |
| L2 / Ridge | $\min_\theta J(\theta) + \lambda \|\theta\|_2^2$ |
| Ridge 解析解 | $\theta = (X^T X + \lambda I)^{-1} X^T y$ |
| L1 / Lasso | $\min_\theta J(\theta) + \lambda \|\theta\|_1$ |
| MAP | $\theta_{\text{MAP}} = \arg\max_\theta p(D \mid \theta) \, p(\theta)$ |
| Gaussian Prior → L2 | $p(\theta) \propto \exp\!\left(-\frac{1}{2\tau^2} \|\theta\|_2^2\right)$ |
| Laplace Prior → L1 | $p(\theta) \propto \exp(-\lambda \|\theta\|_1)$ |
