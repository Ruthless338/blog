---
title: 变分推理、ELBO与变分自编码器
categories:
  - 科研
tags: 
- 概率
- 机器学习
---


# 变分推理与ELBO
## 定义

变分推理是一种近似推断方法，用于估计难以直接计算的概率分布（如贝叶斯后验分布）。其核心思想是：

- 选择一个简单的参数化分布族 $q(z;\lambda)$（称为变分分布）
- 优化参数 $\lambda$，使 $q(z)$ 尽可能接近目标分布 $p(z|x)$
- 用 $q(z)$ 作为 $p(z|x)$ 的近似

## 公式推导

### 1. 问题设定

给定：观测数据 $x$，隐变量 $z$。

目标：计算后验分布

$$
p(z|x) = \frac{p(x, z)}{p(x)}
$$

其中边缘似然

$$
p(x) = \int p(x, z) dz
$$

通常难以计算（高维积分）。

### 2. KL 散度最小化

用变分分布 $q(z)$ 近似 $p(z|x)$，最小化 KL 散度：

$$
\boldsymbol{\lambda}^* = \arg\min_{\boldsymbol{\lambda}} \mathrm{KL}\big( q(\mathbf{z}; \boldsymbol{\lambda}) \parallel p(\mathbf{z} \mid \mathbf{x}) \big)
$$

KL 散度展开：

$$
\mathrm{KL}(q \parallel p) = \int q(\mathbf{z}) \log \frac{q(\mathbf{z})}{p(\mathbf{z} \mid \mathbf{x})}  d\mathbf{z}
$$

### 3. 导出证据下界 (ELBO)

将贝叶斯公式 $p(z|x) = \frac{p(x, z)}{p(x)}$ 代入：

$$
\begin{align*}
\mathrm{KL}(q \parallel p) 
&= \int q(\mathbf{z}) \left[ \log q(\mathbf{z}) - \log p(\mathbf{z} \mid \mathbf{x}) \right] d\mathbf{z} \\
&= \int q(\mathbf{z}) \left[ \log q(\mathbf{z}) - \log \frac{p(\mathbf{x}, \mathbf{z})}{p(\mathbf{x})} \right] d\mathbf{z} \\
&= \int q(\mathbf{z}) \log q(\mathbf{z})  d\mathbf{z} - \int q(\mathbf{z}) \log p(\mathbf{x}, \mathbf{z})  d\mathbf{z} + \log p(\mathbf{x})
\end{align*}
$$

整理得：

$$
\boxed{
\mathrm{KL}(q \parallel p) = -\underbrace{\left[ \mathbb{E}_{q}[\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q}[\log q(\mathbf{z})] \right]}_{\text{ELBO}} + \log p(\mathbf{x})
}
$$

其中 $\log p(x)$ 是常数（证据），因此：

$$
\min \mathrm{KL}(q \parallel p) \quad \Leftrightarrow \quad \max \mathrm{ELBO}(\boldsymbol{\lambda})
$$

### 4. ELBO 分解

ELBO 可分解为两部分：

$$
\mathrm{ELBO} = \mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x} \mid \mathbf{z})] - \mathrm{KL}\big( q(\mathbf{z}) \parallel p(\mathbf{z}) \big)
$$

推导：

$$
\begin{align*}
\mathrm{ELBO} 
&= \mathbb{E}_{q}[\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q}[\log q(\mathbf{z})] \\
&= \mathbb{E}_{q}[\log (p(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z}))] - \mathbb{E}_{q}[\log q(\mathbf{z})] \\
&= \mathbb{E}_{q}[\log p(\mathbf{x} \mid \mathbf{z})] + \mathbb{E}_{q}[\log p(\mathbf{z})] - \mathbb{E}_{q}[\log q(\mathbf{z})] \\
&= \mathbb{E}_{q}[\log p(\mathbf{x} \mid \mathbf{z})] - \underbrace{\left( \mathbb{E}_{q}[\log q(\mathbf{z})] - \mathbb{E}_{q}[\log p(\mathbf{z})] \right)}_{\mathrm{KL}(q \parallel p)}
\end{align*}
$$

### 5. 变分优化

最大化 ELBO 的梯度上升更新：

$$
\boldsymbol{\lambda}^{(t+1)} = \boldsymbol{\lambda}^{(t)} + \eta \nabla_{\boldsymbol{\lambda}} \mathrm{ELBO}
$$

梯度计算使用重参数化技巧（Reparameterization Trick）：

令 $z = g(\epsilon; \lambda)$，其中 $\epsilon \sim p(\epsilon)$

梯度：

$$
\nabla_{\boldsymbol{\lambda}} \mathrm{ELBO} = \mathbb{E}_{p(\boldsymbol{\epsilon})} \left[ \nabla_{\boldsymbol{\lambda}} \log p(\mathbf{x}, g(\boldsymbol{\epsilon}; \boldsymbol{\lambda})) - \nabla_{\boldsymbol{\lambda}} \log q(g(\boldsymbol{\epsilon}; \boldsymbol{\lambda})) \right]
$$


## 关键点总结

- 本质：用优化问题替代积分计算
- ELBO 性质：
  - $\mathrm{ELBO} \leq \log p(\mathbf{x})$（故名"下界"）
  - 最大化 ELBO 等价于最小化 $\mathrm{KL}(q \parallel p)$
- 变分族选择：
  - 平均场近似：$q(\mathbf{z}) = \prod_{i} q_i(z_i)$
  - 高斯分布：$q(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$
- 优势：将推断转化为可扩展的优化问题（适合大数据）
- 应用：贝叶斯神经网络、主题模型（LDA）、变分自编码器（VAE）

#变分自编码器

# 变分自编码器（VAE）

## 1. VAE 的定义

变分自编码器（Variational Autoencoder, VAE）是一种深度生成模型，结合了自编码器结构和变分推断。核心思想：
- 用神经网络参数化概率分布（编码器为近似后验，解码器为生成模型）
- 通过重参数化技巧实现端到端训练
- 目标：学习数据的潜在表示并生成新样本

## 2. VAE 的概率图模型

$$
\mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(0, I), \quad 
\mathbf{x} \sim p_{\theta}(\mathbf{x} \mid \mathbf{z})
$$

其中：
- $z \in \mathbb{R}^d$：潜在变量（低维表示）
- $x \in \mathbb{R}^D$：观测数据（$D \gg d$）
- $\theta$：生成模型（解码器）参数

## 3. VAE 的架构

- **编码器（推断网络）：**
  $$
  q_{\phi}(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_{\phi}(\mathbf{x}), \boldsymbol{\sigma}_{\phi}^2(\mathbf{x})I)
  $$
- **解码器（生成网络）：**
  $$
  p_{\theta}(\mathbf{x} \mid \mathbf{z}) = 
  \begin{cases} 
  \mathcal{N}(\boldsymbol{\mu}_{\theta}(\mathbf{z}), \boldsymbol{\sigma}^2I) & \text{(连续数据)} \\ 
  \text{Bernoulli}(\boldsymbol{\pi}_{\theta}(\mathbf{z})) & \text{(二值数据)}
  \end{cases}
  $$
- $\phi$：编码器参数
- $\theta$：解码器参数
- 神经网络实现：
  $$
  \boldsymbol{\mu}_{\phi}, \boldsymbol{\sigma}_{\phi} = \mathrm{Encoder}_{\phi}(x)
  $$
  $$
  \boldsymbol{\mu}_{\theta} = \mathrm{Decoder}_{\theta}(z)
  $$

## 4. VAE 的推导：目标函数

### 步骤 1：变分下界 (ELBO)

VAE 最大化观测数据的对数似然下界：

$$
\log p_{\theta}(\mathbf{x}) \geq \mathrm{ELBO}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_{\theta}(\mathbf{x}|\mathbf{z})] - \mathrm{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z}))
$$

### 步骤 2：KL 散度项解析解

当 $p(z)=\mathcal{N}(0,I)$ 且 $q_{\phi}=\mathcal{N}(\mu,\sigma^2)$ 时：

$$
\mathrm{KL}(q_{\phi} \parallel p) = -\frac{1}{2} \sum_{j=1}^d \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)
$$

其中 $d$ 是潜在空间维度。

### 步骤 3：重建项估计

使用蒙特卡洛采样(随机抽样取平均)估计：

$$
\mathbb{E}_{q_{\phi}}[\log p_{\theta}(\mathbf{x}|\mathbf{z})] \approx \frac{1}{L} \sum_{l=1}^L \log p_{\theta}(\mathbf{x} \mid \mathbf{z}^{(l)})
$$

其中 $\mathbf{z}^{(l)} = g_{\phi}(x, \epsilon^{(l)}), \epsilon^{(l)} \sim \mathcal{N}(0, I)$，通常 $L=1$。

## 5. 重参数化技巧 (Reparameterization Trick)

关键创新：将随机采样转化为确定性计算：

$$
\mathbf{z} = \boldsymbol{\mu}_{\phi}(\mathbf{x}) + \boldsymbol{\sigma}_{\phi}(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, I)
$$

$\odot$ 表示逐元素乘法，$\epsilon$ 是随机采样的一个噪声，这样保证了"随机采样"的计算过程可以反向传播，即梯度可反向传播至编码器参数 $\phi$。

## 6. 完整目标函数

对于数据集 $\mathcal{D} = \{x^{(i)}\}_{i=1}^N$，优化：

$$
\max_{\theta, \phi} \mathcal{L}(\theta, \phi; \mathcal{D}) = \sum_{i=1}^N \mathcal{L}(\theta, \phi; \mathbf{x}^{(i)})
$$

其中单样本目标：

$$
\mathcal{L}(\theta, \phi; \mathbf{x}) = 
\underbrace{\mathbb{E}_{q_{\phi}}[\log p_{\theta}(\mathbf{x}|\mathbf{z})]}_{\text{重建项}} 
- \underbrace{\mathrm{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z}))}_{\text{正则项}}
$$

实际计算（$L=1$）：

$$
\mathcal{L}(\theta, \phi; \mathbf{x}) \approx 
\log p_{\theta}(\mathbf{x} \mid \mathbf{z}) 
- \frac{1}{2} \sum_{j=1}^d \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)
$$

其中 $\mathbf{z} = \mu_{\phi} + \sigma_{\phi} \odot \epsilon$。

## 7. 训练算法（伪代码）

典型的 VAE 训练流程如下：

1. 从数据集中采样一个批次 $\{\mathbf{x}^{(i)}\}_{i=1}^B$
2. 对每个 $\mathbf{x}^{(i)}$：
    - 计算 $\boldsymbol{\mu}_{\phi}(\mathbf{x}^{(i)}), \boldsymbol{\sigma}_{\phi}(\mathbf{x}^{(i)})$
    - 采样 $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$
    - 计算 $\mathbf{z}^{(i)} = \boldsymbol{\mu}_{\phi} + \boldsymbol{\sigma}_{\phi} \odot \boldsymbol{\epsilon}$
    - 计算重建 $\hat{\mathbf{x}}^{(i)} = \mathrm{Decoder}_{\theta}(\mathbf{z}^{(i)})$
3. 计算损失：

$$
\mathcal{L} = \frac{1}{B} \sum_{i=1}^B \left[ 
-\log p_{\theta}(\mathbf{x}^{(i)} \mid \mathbf{z}^{(i)}) + \frac{1}{2} \sum_{j=1}^d (\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1)
\right]
$$

4. 反向传播，更新 $\theta$ 和 $\phi$

## 8. 生成新样本

$$
\mathbf{z}_{\text{new}} \sim p(\mathbf{z}) = \mathcal{N}(0, I), \quad 
\mathbf{x}_{\text{new}} \sim p_{\theta}(\mathbf{x} \mid \mathbf{z}_{\text{new}})
$$

## 关键创新点总结

- **概率自编码器**：
  - 编码器 $\to$ 近似后验 $q_{\phi}(z|x)$
  - 解码器 $\to$ 生成模型 $p_{\theta}(x|z)$
- **可微训练**：重参数化技巧解决随机节点梯度问题
- **正则化潜在空间**：KL 散度项强制潜在分布匹配先验 $\mathcal{N}(0, I)$
- **生成能力**：从 $p(z)$ 采样 $\to$ 解码器 $\to$ 生成新样本
