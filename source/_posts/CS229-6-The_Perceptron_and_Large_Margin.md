---
title: CS229-6 The Perceptron and Large Margin
date: 2026-06-07
categories:
    - 科研
tags:
    - CS229
    - 机器学习
mathjax: true
---

## 感知机

<!--more-->

## 1. Batch Learning vs Online Learning

### 1.1 Batch Learning

前面的大多数算法都是 batch learning。

比如 logistic regression、SVM、GDA、Naive Bayes，通常都是：

$$\{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$$

全部训练数据先给你，然后你训练模型 $h_\theta(x)$，最后在测试集上评估。

这种学习方式关注的是 **training error** 和 **generalization error**。

### 1.2 Online Learning

Online learning 的流程是：

- 第 1 步，给模型看 $x^{(1)}$，模型先预测 $\hat{y}^{(1)}$，然后真实标签 $y^{(1)}$ 被揭晓。如果预测错了，模型更新参数。
- 接着给模型看 $x^{(2)}$，再预测、揭晓、更新。
- 一直到 $x^{(m)}$。

在 online learning 中，我们关心的是 **number of mistakes**——也就是总共犯了多少次错，而不是单纯关心最后模型在测试集上的表现。

## 2. Perceptron Algorithm：感知机算法

感知机模型为：

$$h_\theta(x) = g(\theta^T x)$$

其中：

$$g(z) = \begin{cases} 1, & z \geq 0 \\ -1, & z < 0 \end{cases}$$

### 2.1 Perceptron Update Rule

给定一个训练样本 $(x, y)$：

- 如果预测正确 $h_\theta(x) = y$，那么不更新参数。
- 如果预测错误 $h_\theta(x) \neq y$，则更新：

$$\theta := \theta + yx$$

> CS229 notes6 中给出的更新规则正是：如果预测正确则不改变参数；如果预测错误，则执行 $\theta := \theta + yx$。

### 2.2 为什么更新是 $\theta := \theta + yx$？

我们希望正确分类时满足：

$$y(\theta^T x) > 0$$

如果某个样本被分错，说明：

$$y(\theta^T x) \leq 0$$

更新后 $\theta_{new} = \theta + yx$，于是：

$$
\begin{aligned}
y(\theta_{new}^T x) &= y((\theta + yx)^T x) \\
&= y\theta^T x + y^2 x^T x \\
&= y\theta^T x + \|x\|^2 \quad (\because y^2 = 1)
\end{aligned}
$$

也就是说，更新之后，这个样本的分类 margin 至少往正确方向增加了 $\|x\|^2$。

所以这个更新规则的直觉是：

- 如果正样本被误判为负，就把 $\theta$ 往 $x$ 的方向推；
- 如果负样本被误判为正，就把 $\theta$ 往 $-x$ 的方向推。

## 3. Online Mistake Bound：感知机犯错次数上界

这是 note6 的核心定理。

假设存在一个单位向量 $u$（$\|u\|_2 = 1$），并且对所有样本满足：

$$y^{(i)}(u^T x^{(i)}) \geq \gamma$$

这表示数据不仅线性可分，而且存在一个 margin 至少为 $\gamma$ 的分隔超平面。

同时假设所有输入都有界：

$$\|x^{(i)}\| \leq D$$

那么感知机算法在整个序列上犯错次数至多为：

$$\left(\frac{D}{\gamma}\right)^2$$

### 3.1 第一部分：证明 $\theta$ 沿着正确方向增长

考虑 $(\theta^{(k+1)})^T u$。由更新式 $\theta^{(k+1)} = \theta^{(k)} + y^{(i)} x^{(i)}$，有：

$$(\theta^{(k+1)})^T u = (\theta^{(k)})^T u + y^{(i)} (x^{(i)})^T u$$

由于假设存在 margin $y^{(i)} u^T x^{(i)} \geq \gamma$，因此：

$$(\theta^{(k+1)})^T u \geq (\theta^{(k)})^T u + \gamma$$

每犯一次错，参数向量在正确方向 $u$ 上的投影至少增加 $\gamma$。所以经过 $k$ 次错误后：

$$(\theta^{(k+1)})^T u \geq k\gamma$$

> CS229 notes6 里也是通过这个归纳得到 $(\theta^{(k+1)})^T u \geq k\gamma$。

### 3.2 第二部分：证明 $\|\theta\|$ 增长不会太快

现在看参数范数。由更新式 $\theta^{(k+1)} = \theta^{(k)} + y^{(i)} x^{(i)}$：

$$
\begin{aligned}
\|\theta^{(k+1)}\|^2 &= \|\theta^{(k)} + y^{(i)} x^{(i)}\|^2 \\
&= \|\theta^{(k)}\|^2 + \|x^{(i)}\|^2 + 2y^{(i)}(\theta^{(k)})^T x^{(i)}
\end{aligned}
$$

因为这一步发生在"犯错"的样本上，所以 $y^{(i)}(\theta^{(k)})^T x^{(i)} \leq 0$，交叉项 $\leq 0$。因此：

$$\|\theta^{(k+1)}\|^2 \leq \|\theta^{(k)}\|^2 + \|x^{(i)}\|^2$$

又因为 $\|x^{(i)}\| \leq D$，所以：

$$\|\theta^{(k+1)}\|^2 \leq \|\theta^{(k)}\|^2 + D^2$$

每次犯错，参数范数平方最多增加 $D^2$。经过 $k$ 次错误后：

$$\|\theta^{(k+1)}\|^2 \leq kD^2 \quad\Rightarrow\quad \|\theta^{(k+1)}\| \leq \sqrt{k}D$$

> CS229 notes6 的证明中也用犯错条件得到交叉项不大于 0，从而推出 $\|\theta^{(k+1)}\|^2 \leq kD^2$。

### 3.3 第三部分：把两个不等式合起来

我们已有 $(\theta^{(k+1)})^T u \geq k\gamma$ 和 $\|\theta^{(k+1)}\| \leq \sqrt{k}D$。

由 Cauchy-Schwarz inequality（$\|u\| = 1$）：

$$(\theta^{(k+1)})^T u \leq \|\theta^{(k+1)}\| \|u\| = \|\theta^{(k+1)}\|$$

因此：

$$k\gamma \leq (\theta^{(k+1)})^T u \leq \|\theta^{(k+1)}\| \leq \sqrt{k}D$$

即 $k\gamma \leq \sqrt{k}D$，两边除以 $\sqrt{k}\gamma$ 得 $\sqrt{k} \leq \frac{D}{\gamma}$，平方得：

$$k \leq \left(\frac{D}{\gamma}\right)^2$$

这就证明了 perceptron 最多犯 $\left(\frac{D}{\gamma}\right)^2$ 次错误。

## 4. Perceptron 和 Neural Network 的关系

Perceptron 可以看成最简单的神经元：

$$h_\theta(x) = g(\theta^T x)$$

其中 $g$ 是 step function。现代神经网络把这个想法推广成：

$$h(x) = \sigma(Wx + b)$$

区别在于：

| Perceptron | Neural Network |
|---|---|
| step function，不可微 | 使用 sigmoid / tanh / ReLU / SiLU 等可训练激活函数 |
| 只能处理线性分类 | 多层组合可以表达非线性函数 |
| 无反向传播 | 通过 backpropagation 训练 |
