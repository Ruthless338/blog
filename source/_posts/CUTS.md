---
title: CUTS -- A Deep Learning and Topological Framework for Multigranular Unsupervised Medical Image Segmentation
date: 2025-07-10
categories: 
    - 科研
tags: 
    - 核函数
    - 聚类
    - 扩散映射
    - 机器学习
    - 语义分割
    - 论文
    - CMR
---

论文链接：[CUTS: A Deep Learning and Topological Framework for Multigranular Unsupervised Medical Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-72111-3_15)

此论文并不是基于迁移学习，理论基础是扩散映射。扩散映射（Diffusion Maps）是由 Coifman 和 Lafon 于 2006 年提出的一种非线性降维和数据结构分析方法，其核心思想是通过模拟**高维数据上的 “热扩散过程”**，捕捉数据的内在几何结构（如流形结构），并将高维数据映射到低维空间以保留关键的拓扑和几何信息。

<!--more-->

## 1. 基本符号与目标

设图像像素总数为 $N = W \times H$（$W, H$ 为图像宽高），每个像素的嵌入向量为 $z_{ij} \in \mathbb{R}^d$（$d$ 为嵌入维度），所有嵌入构成数据矩阵 $X \in \mathbb{R}^{N \times d}$。

**目标**：通过扩散凝聚将 $X$ 粗粒度化为不同层次的簇（clusters），生成从细到粗的多尺度分割结果。

## 2. 亲和矩阵（Affinity Matrix）的构建

亲和矩阵用于衡量数据点（嵌入向量）之间的相似性，是扩散过程的基础。

文档中通过高斯核函数定义亲和矩阵 $K$：

$$K(x_m, x_n) = \exp\left(-\frac{\|x_m - x_n\|^2}{\epsilon}\right) \tag{3}$$

其中：
- $x_m, x_n \in \mathbb{R}^d$ 是数据矩阵 $X$ 中的两个样本（嵌入向量）
- $\|\cdot\|^2$ 是欧氏距离的平方，衡量两样本的差异
- $\epsilon$ 是带宽参数（bandwidth），控制邻域范围：
  - $\epsilon$ 越大，更多远处的点会被视为"相似"，邻域越广

$K$ 是一个 $N \times N$ 的对称矩阵（Gram 矩阵），其元素 $K(x_m, x_n)$ 表示样本 $x_m$ 与 $x_n$ 的相似度（值越大越相似）。

## 3. 扩散算子（Diffusion Operator）的定义

扩散算子将亲和矩阵转换为"概率转移矩阵"，模拟数据点之间的"扩散"过程（类似马尔可夫随机游走）。

首先定义度矩阵（degree matrix）$D$：

$$D(x_m, x_m) = \sum_{n=1}^{N} K(x_m, x_n) \tag{4b}$$

$D$ 是**对角矩阵**，对角线元素 $D(x_m, x_m)$ 是 $K$ 中第 $m$ 行的总和，代表样本 $x_m$ 与所有其他样本的总相似度。

扩散算子 $P$ 定义为：

$$P = D^{-1}K \tag{4a}$$

$P$ 是行归一化的矩阵（每行元素和为 1），其元素 $P(m,n)$ 表示从样本 $x_m$ 扩散到 $x_n$ 的单步概率，即"转移概率"。

## 4. 扩散凝聚的迭代过程

扩散凝聚通过时变扩散过程（time-inhomogeneous diffusion）实现粗粒度化：随着迭代，数据点逐渐向"局部重心"聚集，小簇合并为大簇，形成从细到粗的粒度层次。

### 迭代步骤：

1. **初始化**：设初始数据矩阵为 $X_0 = X$（原始嵌入向量）

2. **迭代更新**（对于 $t = 1, 2, \ldots, T$）：
   - 计算当前数据 $X_{t-1}$ 的亲和矩阵 $K_{t-1}$（使用公式 3）
   - 计算度矩阵 $D_{t-1}$（使用公式 4b）
   - 计算扩散算子 $P_{t-1} = D_{t-1}^{-1}K_{t-1}$（使用公式 4a）
   - 更新数据矩阵：$X_t = P_{t-1}X_{t-1}$

即：

$$\begin{cases}
X_0 \leftarrow X \\
\text{for } t \in [1, \ldots, T]: \\
\quad K_{t-1} \leftarrow K(X_{t-1}) \text{ （公式3）} \\
\quad D_{t-1} \leftarrow D(K_{t-1}) \text{ （公式4b）} \\
\quad P_{t-1} \leftarrow D_{t-1}^{-1}K_{t-1} \text{ （公式4a）} \\
\quad X_t \leftarrow P_{t-1}X_{t-1}
\end{cases}$$

## 5. 核心机制：从细粒度到粗粒度的演化

扩散凝聚的本质是交替执行"相似性计算"和"数据聚合"：

每次迭代中：
- $K_{t-1}$ 重新计算当前数据点的相似性
- $P_{t-1}$ 将相似性转换为转移概率
- $X_t = P_{t-1}X_{t-1}$ 使每个数据点向"相似点的加权平均"（局部重心）移动，实现"凝聚"

随着迭代次数 $t$ 增加：
- **初始阶段**（小 $t$）：仅最相似的点聚集，形成细粒度的小簇
- **后期阶段**（大 $t$）：小簇逐渐合并为大簇，形成粗粒度的分割