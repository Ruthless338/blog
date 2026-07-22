---
title: Matrix theory in machine learning
date: 2026-07-05
categories:
    - 科研
tags:
    - matrix
    - 机器学习
mathjax: true
---
前三讲分别从列空间、二次型、SVD 三个角度建立了矩阵论的基础工具。第四讲把这些工具串起来，回答一个核心问题：

> 机器学习为什么反复关心"方向"？哪些方向有信息？哪些方向是噪声？哪些方向应该被正则化或丢弃？

<!--more-->

# 第四讲：统一视角——方向、信息与稳定性

## 1. 从线性回归到"方向"思维

线性回归 $\hat{y}=X\theta$ 的本质是：用 $X$ 的列向量线性组合去逼近 $y$。

- 若 $y\in\operatorname{Col}(X)$：训练误差可以为零
- 若 $y\notin\operatorname{Col}(X)$：只能找最接近的投影 → 最小二乘

对 $X$ 做 SVD：$X=U\Sigma V^T$。奇异值 $\sigma_i$ 刻画了数据在不同方向上的信息强弱：

$$
\sigma_i \text{ 大 → 该方向信息强，参数容易确定}
$$
$$
\sigma_i \text{ 小 → 该方向信息弱，噪声容易淹没信号}
$$
$$
\sigma_i = 0 \text{ → 该方向落在 null space，数据完全不可见}
$$

普通最小二乘解包含 $\frac{u_i^T y}{\sigma_i}v_i$ 项——当 $\sigma_i$ 很小时，除以小数会放大噪声，导致参数剧烈波动、泛化变差。所以线性回归出问题的深层原因不是"不可逆"，而是：

> **在训练数据几乎看不到的方向上，求解器却试图精确拟合。**

---

## 2. Ridge：压制不可靠方向

Ridge 的目标函数和解：

$$
\min_\theta \|X\theta-y\|^2 + \lambda\|\theta\|^2,
\quad \theta = (X^TX + \lambda I)^{-1}X^T y
$$

从 SVD 视角看，普通最小二乘在第 $i$ 个方向的缩放系数是 $\frac{1}{\sigma_i}$，而 Ridge 变为：

$$
\frac{\sigma_i}{\sigma_i^2 + \lambda}
$$

- $\sigma_i$ 大时：$\frac{\sigma_i}{\sigma_i^2+\lambda} \approx \frac{1}{\sigma_i}$，基本不干预
- $\sigma_i$ 小时：$\frac{\sigma_i}{\sigma_i^2+\lambda} \approx \frac{\sigma_i}{\lambda} \ll 1$，强烈压制

所以 Ridge 的本质是：**数据能看清的方向正常拟合，看不清的方向不让参数乱跑。** 从二次型角度看，$X^TX \succeq 0$ 可能奇异，加上 $\lambda I$ 后 $X^TX+\lambda I \succ 0$，保证稳定可逆。

---

## 3. Lasso：换一种控制方式

Ridge 用 $\ell_2$ 惩罚让参数整体变小但不归零；Lasso 用 $\ell_1$ 惩罚：

$$
\min_\theta \|X\theta-y\|^2 + \lambda\|\theta\|_1
$$

鼓励稀疏——部分参数直接变成 0，等价于自动特征选择。

```
Ridge：压制所有不可靠方向（软抑制）
Lasso：直接删除部分方向（硬删除）
```

两者都在解决同一个问题：**模型不应该在不可靠或冗余的方向上过度拟合。**

---

## 4. PCA 与协方差：数据变化的主方向

PCA 通过 $X=U\Sigma V^T$ 提取主方向：$V$ 的列是特征空间中的主方向，$\sigma_i^2/m$ 是各方向的方差。PCA 保留大奇异值方向、丢弃小奇异值方向，因为小奇异值方向通常对应噪声、冗余或不稳定结构。

协方差矩阵 $\Sigma = Q\Lambda Q^T$ 遵循同一逻辑：$Q$ 的列是数据变化的主方向，$\lambda_i$ 是各方差的方差。若某 $\lambda_i=0$，说明数据完全落在低维子空间中，协方差矩阵奇异——这会使 GDA、GMM 中的 $\Sigma^{-1}$ 无法计算。

**PCA 和 Ridge 的共同点**：都不信任数据中极弱的方向。
- PCA / Truncated SVD：直接丢掉小奇异值方向
- Ridge：保留所有方向但强烈压制小奇异值方向

---

## 5. 高维小样本：本质是"方向不足"

样本数 $m=100$，特征维度 $n=10000$。$\operatorname{rank}(X) \le 100$，意味着在 10000 维空间中数据最多撑起 100 维子空间，大量方向完全不可见。后果包括：

$$
X^TX \text{ 奇异，协方差矩阵不可逆，解不唯一，容易过拟合}
$$

这不是"维度太大"的问题，而是：**可观测方向太少，不可观测方向太多。** 解决手段：Ridge、PCA、降维、对角协方差、低秩建模、Factor Analysis。

---

## 6. Factor Analysis：低秩协方差 + 噪声

Factor Analysis 假设高维数据由少数潜在因子生成：

$$
x = \mu + \Lambda z + \epsilon, \quad z \in \mathbb{R}^k,\ k \ll n
$$

$$
\operatorname{Cov}(x) = \Lambda\Lambda^T + \Psi
$$

- $\Lambda\Lambda^T$：低秩部分，少数潜在因子带来的主要相关性
- $\Psi$：对角矩阵，各维度的独立噪声

与 PCA 一样强调"高维结构来自少数方向"，但 Factor Analysis 走概率建模路线，PCA 走线性代数路线。

---

## 7. Kernel：PSD 保证合法的几何空间

Kernel matrix $K_{ij} = K(x_i, x_j)$ 必须 PSD。原因是合法 kernel 需对应某个特征空间的内积：

$$
K(x,z) = \phi(x)^T\phi(z)
$$

于是对任意系数 $c$：

$$
c^T K c = \left\|\sum_i c_i \phi(x_i)\right\|^2 \ge 0 \quad\Rightarrow\quad K \succeq 0
$$

若 $K$ 不是 PSD，SVM 的 dual problem 可能非凸，相似度也没有可靠的几何解释。Kernel 方法的本质是：**不显式构造高维特征 $\phi(x)$，但通过 PSD 约束保证它隐式地对应某个合法内积空间。**

---

## 8. Hessian：优化中的方向曲率

Loss $L(\theta)$ 的 Hessian $H = \nabla^2 L(\theta)$ 刻画各方向的二阶曲率：$v^T H v$。

- $H \succeq 0$：所有方向不下弯，函数凸
- $H \succ 0$：所有方向正曲率，局部像碗，最优解稳定
- $H$ 有负特征值：存在向下弯的方向（saddle / 非凸）

---

## 9. 条件数与梯度下降

二次函数 $f(\theta) = \frac{1}{2}\theta^T A\theta$（$A \succ 0$）的条件数：

$$
\kappa(A) = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

条件数大 → loss surface 是又窄又长的椭圆谷 → 陡峭方向震荡、平坦方向缓慢 → 收敛变慢。**不同方向的曲率差异太大，优化很难走。** 这解释了标准化、预条件、二阶方法的必要性。

---

## 10. 特征标准化的几何意义

特征尺度差异大（如 $[0,1]$ vs $[0,100000]$）→ $X^TX$ 各方向尺度悬殊 → 条件数增大 → loss surface 被拉长 → 梯度下降收敛慢。标准化后各特征尺度接近，条件数改善。标准化不是工程细节，而是在**改善优化问题的几何形状。**

---

## 11. 深度学习：矩阵视角仍然成立

深度网络每层 $h_{l+1} = f(W_l h_l + b_l)$，权重矩阵 $W_l$ 控制特征的旋转、放大、压缩和梯度传播。

- 奇异值太大 → 梯度爆炸
- 奇异值太小 → 梯度消失

因此深度学习同样关心：权重初始化、Batch/Layer Norm、残差连接、谱范数正则化、Hessian 特征值、loss landscape——这些都是矩阵论概念的延伸。

---

## 12. LoRA：低秩思想的现代应用

大模型微调时，LoRA 假设权重更新是低秩的：

$$
\Delta W = BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll d,k
$$

直觉：**下游任务不需要在所有参数方向上修改模型，只需在少数关键方向上调整。** 这与 PCA、低秩近似的核心信念一致——高维空间中真正重要的变化方向往往很少。

---

## 13. 总结：一条主线串起所有工具

整门课的核心线索是**方向**：

| 问题 | 矩阵论工具 | 关键结论 |
|------|-----------|---------|
| 哪些方向有信息？ | SVD、协方差特征分解 | 大奇异值/大特征值方向 |
| 哪些方向不可靠？ | 伪逆、条件数 | 小奇异值方向放大噪声 |
| 如何控制不可靠方向？ | Ridge、Lasso | 压制或删除弱信息方向 |
| 如何降维？ | PCA、截断 SVD | 保留大奇异值方向 |
| 为什么高维难？ | Rank 不等式 | 可观测方向远少于特征维度 |
| 概率模型如何低秩化？ | Factor Analysis | 低秩协方差 + 对角噪声 |
| Kernel 为何需要 PSD？ | 二次型非负 | 保证对应合法内积空间 |
| 优化为何慢？ | Hessian、条件数 | 各方向曲率差异过大 |
| 深度学习如何受益？ | 权重 SVD、梯度传播 | 控制奇异值 → 稳定训练 |
| 微调为何能低秩？ | LoRA、低秩分解 | 任务适配只需少数方向 |

一句话：**机器学习中的矩阵论，归根到底就是在研究"方向"——哪些方向有信息，哪些方向没有，哪些方向不可靠，以及如何据此做出更好的模型选择。**
