---
title: Toward Accurate Cardiac MRI Segmentation
    With Variational Autoencoder-Based
    Unsupervised Domain Adaptation
    论文复现
date: 2025-07-07
categories: 
    - 科研
tags: 
    - 论文
    - 机器学习
    - 语义分割
    - CMR
    - 变分推理
---

论文链接：[Toward Accurate Cardiac MRI Segmentation With Variational Autoencoder-Based Unsupervised Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/10483021)  
论文主要解决了心肌分割的问题，提出无监督域适应方法，将bSSFP(源域)的知识迁移到LGE(目标域)中，实现无需目标域标注的高精度分割。  
关于论文的前置知识，可见KL散度、ELBO、VAE等博客。

<!--more-->

![架构](/images/CMR1-1.png)

传统VAE即变分自编码器只有Encoder与Decoder两部分，论文中的VAMCEI增加了分割器部分，并且通过若干个损失函数来对齐源域和目标域的特征空间。

根据架构图，源域和目标域图像都通过UNet风格的Encoder进行特征提取(到潜在z空间)，z空间通过Decoder进行重建；z空间通过分割器进行预测。根据论文的复现，有7个损失函数：
1. 源域预测结果与真实掩码的分割损失
2. 源域和目标域图像的重建损失
3. 源域和目标域的潜在空间分布分别与标准高斯分布的 KL 散度损失(VAM正则化损失)
4. 源域的整体潜在分布与目标域的整体潜在分布之间的双向 KL 散度损失(全局特征对齐损失)
5. 原型对比损失(局部特征对齐损失)
6. 源域和目标域上生成器与判别器的对抗损失(隐式特征对齐损失)

论文复现代码见：[cardiac_uda_vamcei](https://github.com/Ruthless338/VAMCEI)

接下来重点解析论文中的关键数学推导，包括：
 1. 变分自编码器（VAE）的目标函数
 2. 显式全局特征对齐（KL散度推导）
 3. 显式局部特征对齐（原型对比损失）
 4. 隐式特征对齐（对抗损失）
 5. 多阶段框架中的知识蒸馏损失
 以下逐一详细解释：

## 关键公式与推导

### 1. VAE基础：变分下界（公式1）

VAE的核心目标是最大化观测数据 $(x, y)$ 的对数似然，通过变分推断转化为可优化的下界：

$$
\log p_\theta(x, y) \geq LB_{VAE}(\theta, \phi) = -D_{KL}(q_\phi(z|x) \| p_\theta(z)) + \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|y, z)] + \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(y|z)]
$$

**变量说明：**

- $x$：输入图像（心脏MRI）
- $y$：分割标签（LV/RV/Myo）
- $z$：潜在变量
- $q_\phi(z|x)$：编码器输出的后验分布（近似真实后验 $p_\theta(z|x)$）
- $p_\theta(z)$：先验分布（标准正态 $\mathcal{N}(0, I)$）
- $p_\theta(x|y, z)$：解码器重建的图像分布
- $p_\theta(y|z)$：分割器预测的标签分布

**三项分解：**

- **KL散度项：**

  $$
  -D_{KL}(q_\phi(z|x) \| p_\theta(z))
  $$
  强制潜在空间 $z$ 服从标准正态分布（正则化）。  
  具体计算（公式2，两正态分布的KL散度有公式）：

  $$
  D_{KL} = \frac{1}{2} \sum_{j=1}^M \sum_{i=1}^n ( \sigma_{ij}^2 + \mu_{ij}^2 - \log \sigma_{ij}^2 - 1 )
  $$
  其中 $M$ 为 batch 大小，$n$ 为潜在空间维度，$\mu_{ij}, \sigma_{ij}$ 为第 $j$ 个样本第 $i$ 维的均值和方差。

- **重建项：**

  $$
  \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|y, z)]
  $$
  最大化重建图像 $\hat{x}$ 的似然，对应二值交叉熵损失（公式3）：

  $$
  L_R = -\sum_{i=1}^M \hat{x}_i \log x_i + (1-\hat{x}_i) \log (1-x_i)
  $$

- **分割项：**

  $$
  \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(y|z)]
  $$
  分割预测损失（公式4）：

  $$
  L_{seg} = \sum_{i=1}^M \left[ L_{CE}(y_i, \hat{y}_i) + L_{Dice}(y_i, \hat{y}_i) \right]
  $$
  结合交叉熵和 Dice 损失处理类别不平衡。

---

### 2. 显式全局特征对齐（公式5-8）

**核心问题：** 源域和目标域潜在空间分布不一致，导致域偏移。

**解决方案：** 最小化两域潜在分布的 KL 散度。

- **双向 KL 散度（公式5）：**

  $$
  D[q_{\phi_s}(z), q_{\phi_t}(z)] = D_{KL}[q_{\phi_s}(z) \| q_{\phi_t}(z)] + D_{KL}[q_{\phi_t}(z) \| q_{\phi_s}(z)]
  $$

  传统方法用 L2 距离，本文创新性地采用对称 KL 散度更准确度量分布差异。

- **小批量近似（公式6）：**

  $$
  D_{KL}[q_{\phi_s}(z) \| q_{\phi_t}(z)] = \int \left[ \frac{1}{M} \sum_{i=1}^M q_{\phi_s}(z|x_{Si}) \right] \ln \frac{\frac{1}{M} \sum q_{\phi_s}(z|x_{Si})}{\frac{1}{M} \sum q_{\phi_t}(z|x_{Ti})} dz
  $$

- **高斯近似（公式7）：**

  $$
  D_{KL} \approx \frac{1}{M^2} \sum_{i=1}^M \sum_{j=1}^M \mathbb{E}_{q_{\phi_s}(z|x_{Sj})} [ \ln q_{\phi_s}(z|x_{Sj}) - \ln q_{\phi_t}(z|x_{Tj}) ]
  $$

- **独立维度分解（公式8）：**

  $$
  D_{KL} = \frac{1}{M^2} \sum_{k=1}^n \sum_{j=1}^M \sum_{i=1}^M \left[ \ln \frac{\sigma_{Tik}}{\sigma_{Sik}} - \frac{1}{2} + \frac{\sigma_{Sjk}^2 + (\mu_{Sjk} - \mu_{Sik})^2}{2\sigma_{Tik}^2} + \frac{\sigma_{Sjk}^2 + (\mu_{Sjk} - \mu_{Tik})^2}{2\sigma_{Sik}^2} \right]
  $$

  其中 $\mu_{Sik}, \sigma_{Sik}$ 为源域第 $i$ 个样本第 $k$ 维的均值和方差，$\mu_{Tik}, \sigma_{Tik}$ 为目标域对应值。  
  关键在于将复杂的多维积分转化为可计算的求和。
---

### 3. 显式局部特征对齐（公式9-10）

**目标：** 对齐同类特征，分离异类特征（跨域）。  
举例来说，是为了对齐源域和目标域中心肌 Myo 的特征，分离源域心肌 Myo 与目标域右心室 RV 的特征。

- **类别原型计算（公式9）：**

  $$
  C_k = \frac{\sum_{i=1}^M \sum_{j=1}^n z_{ij} I(y_{ij}=k)}{\sum_{i=1}^M \sum_{j=1}^n I(y_{ij}=k)}
  $$

  其中 $z_{ij}$ 为第 $i$ 个样本第 $j$ 像素的特征向量，$I(y_{ij}=k)$ 为指示函数（像素属于类别 $k$ 时为 1）。

- **原型对比损失（公式10）：**

  $$
  Pro(q_s, q_T) = \frac{1}{K} \sum_{k=1}^K -\ln \left[ \frac{\exp(\langle C_{Sk}, C_{Tk} \rangle / \tau)}{\sum_{i \neq k} \exp(\langle C_{Sk}, C_{Ti} \rangle / \tau) + \exp(\langle C_{Tk}, C_{Ti} \rangle / \tau)} \right]
  $$

  其中 $\langle \cdot, \cdot \rangle$ 为余弦相似度，$\tau$ 为温度系数。

---

### 4. 隐式特征对齐（公式11-12）

通过输出空间域判别器实现。

- **判别器损失（公式11）：**

  $$
  L_{dis_d} = \mathbb{E}_{x_S \sim X_S} [\log(Dis(P_S))] + \mathbb{E}_{x_T \sim X_T} [\log(1-Dis(P_T))]
  $$

  目标：区分源域/目标域分割图 $P_S, P_T$。

- **生成器（编码器）损失（公式12）：**

  $$
  L_{dis_g} = \mathbb{E}_{x_T \sim X_T} [\log(Dis(P_T))]
  $$

  编码器试图"欺骗"判别器，使目标域分割图 $P_T$ 被误判为源域，实现隐式特征对齐。

---

### 5. 多阶段框架蒸馏（公式16）

**目标：** 融合互补模型知识，避免语义错误。

- **知识蒸馏损失（公式16）：**

  $$
  L_{distill} = - \sum_{i=1}^K \frac{\exp(p_i / T)}{\sum_j \exp(p_j / T)} \log \left( \frac{\exp(q_i / T)}{\sum_j \exp(q_j / T)} \right)
  $$

  其中：
  - $p_i$：教师模型平均概率（Target VAMCEI + Source VAMCEI）
  - $q_i$：学生模型预测概率
  - $T$：蒸馏温度（软化概率分布）
  - $K$：类别数

**物理意义：** 最小化学生与教师输出的 KL 散度，传递"暗知识"（dark knowledge）。

---
