---
title: EvoHead
date: 2026-05-13
categories:
    - 科研
tags:
    - Generative AI
    - PPO
    - GRPO
---

EvoHead笔记，方便之后回忆。
<!--more-->
## 1. 论文概况

**论文题目：**
*EvoHead: Dyadic Head Motion Generation via Hierarchical Evolution of Heterogeneous Dynamics*

**任务目标：**
在双人对话场景中，给定参与者 A 的音频与头部运动，以及参与者 B 自身的音频，生成参与者 B 的 3D 头部运动序列。运动表示采用 56 维 FLAME 参数，包括 facial expression、jaw articulation 与 head pose 等组成部分。

形式化表示为：

$$
\{A^A, M^A, A^B\} \rightarrow M^B
$$

其中：

* $(A^A)$：对话伙伴 A 的音频audio；
* $(M^A)$：对话伙伴 A 的头部运动motion；
* $(A^B)$：目标人物 B 自身的音频audio；
* $(M^B)$：需要生成的 B 的头部运动motion。

**一句话总结本文：**

EvoHead 将双人对话中的头部运动生成重新表述为一个“异质动态逐层演化”的问题：首先由目标人物自身语音生成稳定的发音相关运动基础，再利用双方对话上下文逐步补充具有交互意义的残差运动，从而同时保证口型/下颌稳定性与互动反应的真实性。

---

## 2. 研究背景：为什么双人数字人生成比单人 talking head 更难

传统 speech-driven talking head 任务通常关注：

$$
\text{Self Audio} \rightarrow \text{Facial / Head Motion}
$$

也就是说，模型主要学习“一个人说什么，就产生什么样的嘴型与头部动作”。

但是在双人对话场景中，一个人的运动不仅受到自身语音影响，还会受到对方行为的影响。

因此，双人对话头部运动生成并不是简单的 audio-to-motion regression，而需要同时处理两类因素：

1. **稳定、强约束的自身语音驱动运动**：例如 jaw articulation；
2. **较弱但关键的上下文交互运动**：例如 nodding、subtle pose changes、attentive expressions 与 emphasis-related reactions。

本文将这两类运动称为具有不同依赖关系与时间尺度的 **heterogeneous dynamics**。

---

## 3. Related Work：现有研究的发展脉络

### 3.1 Audio-driven 3D Talking Head Generation

第一类工作关注由语音直接生成 3D 面部或头部运动。

早期方法通常采用直接回归方式，从语音特征预测 facial motion 或 head motion，例如传统的 speech-to-face regression。

---

### 3.2 Conversational Motion Generation 与 Dyadic Interaction Modeling

第二类工作开始关注对话场景中的互动行为生成。

早期工作通常将角色分开处理：

* speaker motion generation
* listener response generation

其中，listener generation 研究已经意识到一个重要问题：同一个输入条件下，合理的听者反应并非唯一，因此需要利用随机变量、多模态生成或离散表示来描述反应多样性。

进一步地，近期工作如 DIM、DualTalk 与 UniLS 开始尝试统一建模双人对话互动。

---

### 3.3 Discrete Motion Representation Learning

第三类工作关注使用离散表示建模复杂运动分布。

VQ-VAE 及其层次化变体说明，连续运动可以通过离散 codebook 转化为 motion tokens，从而提高生成模型对复杂运动模式的建模能力。后续的人体动作生成方法，如 MoMask 以及其他 coarse-to-fine motion generation 方法，也证明了层次化 token modeling 对于运动真实性和细节表达具有优势。

Residual Vector Quantization，简称 RVQ，进一步通过逐层量化 reconstruction residual 的方式，提高离散表示的表达能力。标准 RVQ 的基本思想是：

* 第一层编码主要信息；
* 后续层不断修正前面层没有重建好的残差；
* 各层共同服务于最终 reconstruction accuracy。

---

## 4. 现有研究的不足

### 4.1 不足一：不同因果来源的运动被混合建模

> 现有整体式建模方式没有显式区分 self-audio-driven motion 与 context-sensitive motion，导致稳定发音动态压制了微弱但关键的交互动态。

---

### 4.2 不足二：缺少对多时间尺度交互动态的建模

> 双人对话运动天然具有 coarse-to-fine 的时间层次结构，而单层表示会削弱模型对长期交互趋势和局部细节反应的共同表达能力。

---

### 4.3 不足三：单纯重建损失无法迫使模型学习交互信息

> Reconstruction supervision 强调“像不像 ground truth”，却没有显式要求模型识别“这个反应是否与当前对话伙伴在时间上和语义上匹配”。

---

## 5. EvoHead核心思想

为解决上述问题，EvoHead 提出的核心观点是：

> 双人对话头部运动不应被视为一个整体直接回归目标，而应被分解为稳定的 articulation anchor 与依赖上下文的 residual dynamics，并按照时间粒度逐层生成。

---

## 6. 创新一：Context-aware Hierarchical Dynamics Allocation（CHDA）

### 6.1 解决的问题

CHDA 主要解决的是：
**如何在 latent representation 层面，将稳定发音动态与上下文相关动态分离开，并形成结构化的离散 token 空间。**

---

### 6.2 具体做法

CHDA 将目标人物 B 的运动表示为：

$$
Z^B = \{z_0, z_1, \dots, z_{R-1}\}
$$

其中：

* $(z_0)$：articulation-anchor token；
* $(z_1, \dots, z_{R-1})$：从粗到细的 residual tokens。

首先，模型只抽取目标人物 B 的 jaw 参数 $(J^B)$，通过一个独立的 vector-quantized autoencoder 进行编码和重建：

$$
J^B \rightarrow z_0 \rightarrow \hat{J}^B
$$

得到的 $(\hat{J}^B)$ 被放回完整的 56 维运动空间中，其他维度暂时置零，从而形成 articulation-oriented base motion：

$$
M^B_{\text{base}}
$$

随后，将真实完整运动与 base motion 做差：

$$
R^B = M^B - M^B_{\text{base}}
$$

这里的 $(R^B)$ 就是 articulation-compensated residual dynamics，包含：

* 未被 jaw anchor 覆盖的 expression；
* head pose 变化；
* 剩余 articulation variation；
* 更重要的 partner-aware reaction。

---

## 7. 创新二：Temporal-Aligned Residual Quantizer（TARQ）

### 7.1 解决的问题

在得到 articulation-compensated residual 后，仍然存在一个问题：
这些 residual dynamics 本身也不是同质的。

其中既包括：

* 较长时间尺度上的姿态趋势；
* 中等时间尺度上的互动变化；
* 极短时间尺度上的局部 expression 与细微动作。

标准 RVQ 虽然也会分层编码 residual，但它通常只是不断修正 reconstruction error，并没有让每一层对应特定的时间尺度或动态意义。

因此，本文提出 TARQ，将 residual quantization 从简单的误差修正过程，重新定义为按照时间粒度组织的 dynamics allocation 过程。

---

### 7.2 具体做法

TARQ 首先将 residual motion 编码为连续 latent sequence，然后使用不同的 temporal smoothing scale 对其进行逐级量化。

论文中采用的时间尺度为：

$$
\{4, 2, 1\}
$$

可以理解为：

* 较大的 smoothing scale 优先捕获较粗粒度、较平稳的动态趋势；
* 较小的 smoothing scale 则进一步补充局部、高频、细粒度变化。

每一层量化后，已解释的动态会从当前 residual 中移除，下一层继续建模剩余部分：

$$
r_l = r_{l-1} - \tilde{H}_l
$$

于是 residual hierarchy 形成了从 coarse 到 fine 的动态分解。

---

### 7.3 为什么 TARQ 优于普通 RVQ

普通 RVQ 的逻辑更接近：

> 第一层没重建好的地方，交给第二层补；第二层仍没重建好的地方，再交给第三层补。

但 EvoHead 希望不同层具有更明确的动态含义：

> 先编码较稳定、较长范围的 residual trend，再逐步编码更局部、更可能反映交互反应的细节变化。

---

## 8. 创新三：Dyadic Context Distillation（DCD）

### 8.1 解决的问题

即使通过 CHDA 和 TARQ 构建了 residual hierarchy，也不能自动保证 residual tokens 真正包含 interaction information。

原因是：

* 仅凭 reconstruction loss，模型仍然可能利用目标人物自身音频或平均运动模式完成重建；
* residual tokens 可能表达了丰富运动细节，但这些细节未必和正确 partner context 对齐；
* 模型需要额外监督，明确区分“正确互动上下文”和“被扰乱的上下文”。

因此，本文引入 Dyadic Context Distillation，用于将具有时间辨别能力的互动信息注入 residual latent hierarchy。

---

### 8.2 Teacher 的构造方式

DCD 首先训练一个 teacher model，用于判断目标人物运动是否与双方对话上下文在时间上匹配。

对于目标人物 B 的 motion $(M^B)$，保持：

* $(M^B)$ 不变；
* 目标人物自身音频 $(A^B)$ 不变；

只扰乱对话伙伴 A 的音频：

$$
T^+ = (M^B, A^B, A^A)
$$

$$
T^- = (M^B, A^B, \tilde{A}^A)
$$

其中：

* $(T^+)$ 是正确匹配的 dyadic context；
* $(T^-)$ 中的 partner audio 经过 chunk-level shuffling，被打乱了长时间范围内与目标反应之间的对应关系。

这种负样本构造非常关键，因为它避免了一个简单问题：
如果同时替换目标人物自身音频，那么 teacher 可能仅仅依赖 lip-sync 或 articulation mismatch 来区分正负样本，而没有真正学习 partner-related interaction。

本文固定 $(M^B)$ 和 $(A^B)$，只扰乱 $(A^A)$，目的就是迫使 teacher 学习：

> 在目标人物自身语音相同的前提下，目标人物的动作是否与正确的对话伙伴上下文相匹配。

---

### 8.3 蒸馏到 residual representation 的原因

Teacher 学到的是 context-discriminative motion representation，即具有互动识别能力的运动特征。

随后，EvoHead 将该表示蒸馏到 residual token 构成的 pre-decoder feature 上，而不是直接作用于 articulation anchor。

这一选择在概念上十分合理：

* jaw articulation 主要应由自身音频控制，不应过度受到 partner context 干扰；
* partner-dependent cue 更应该进入 residual dynamics；
* residual space 正是用于表达 nodding、subtle expression、pose changes 和 reaction-related movement 的位置。

因此，DCD 并不是一个附加的普通 loss，而是在明确规定：

> 交互监督应当进入负责互动变化的 residual hierarchy，而不是污染稳定的 articulation anchor。

---

## 9. 创新四：Articulation-Anchored Progressive Dynamics Evolving

### 9.1 解决的问题

前面的 CHDA、TARQ 和 DCD 主要负责学习一个良好的 structured token space。
但在生成阶段，模型仍需决定：这些 token 应如何被预测。

因此，EvoHead 进一步提出 progressive evolving scheme，使生成顺序与运动的结构分解保持一致。

---

### 9.2 生成过程

整体生成概率被分解为：

$$
p(Z^B|C) = p(z_0|A^B) \prod_{l=1}^{R-1} p(z_l|z_{<l}, C)
$$

其中：

* $(z_0)$ 只由目标人物自身音频 $(A^B)$ 生成；
* residual tokens $(z_l)$ 则由已生成的低层 token 与完整 dyadic condition $(C)$ 联合生成；
* $(C)$ 包含目标音频、伙伴音频与伙伴运动。

生成过程分为两步。

#### 第一步：Masked Articulation Token Generation

模型首先根据目标人物自身音频，以 masked token prediction 的方式生成 articulation-anchor token $(z_0)$。

训练时：

* 随机 mask 一部分 articulation token；
* 模型利用未 mask 的 token 和目标音频预测被 mask 的位置。

推理时：

* 初始所有 token 均被 mask；
* 模型通过多轮置信度更新逐步恢复完整 articulation anchor。

该步骤的目标是优先构建稳定、可靠、与 speech articulation 紧密对齐的运动基础。

#### 第二步：Hierarchical Residual Dynamics Refinement

得到 $(z_0)$ 之后，模型依次预测各层 residual token：

$$
z_1 \rightarrow z_2 \rightarrow \cdots \rightarrow z_{R-1}
$$

每一层预测均依赖：

* 已经生成的 lower-level tokens；
* 完整的 dyadic context。

---

## 10. 总结

### 研究问题

现有双人 talking head 方法虽然引入了双方对话上下文，但通常仍将所有运动混合在统一表示空间中学习。由于 jaw articulation 与自身语音之间的相关性最强，模型容易优先拟合口型和显著运动，而忽略幅度更小、但更能体现互动性的 partner-aware responses，最终导致生成动作平均化、上下文敏感性不足。

### 方法动机

我们认为双人对话头部运动包含两类本质不同的动态：一类是由自身语音可靠决定的稳定 articulation；另一类是由双方上下文共同决定的 interaction-sensitive residual motion。因此，模型应首先建立稳定的 articulation anchor，再在其基础上逐步补充由对话上下文触发的多尺度残差动态。

### 方法创新

EvoHead 首先通过 Context-aware Hierarchical Dynamics Allocation(CHDA) 和Temporal-Aligned Residual Quantizer(TARQ)将完整运动分解为 articulation anchor 与 coarse-to-fine residual tokens；随后通过 Dyadic Context Distillation(DCD)将时间匹配的互动信息注入 residual hierarchy；最后利用 Articulation-Anchored Progressive Dynamics Evolving 先生成语音稳定相关的基础运动，再逐层预测上下文相关残差，从而实现更自然、更丰富且更具互动一致性的双人头部运动生成。

### 实验结论

在 DualTalk 标准测试集和 OOD 测试集上，EvoHead 在 FD、P-FD、SID 和 rPCC 等关键指标上均表现突出，说明其能够提高运动真实性、生成多样性以及双人互动同步性。消融实验进一步证明，articulation anchor、hierarchical dynamics allocation、dyadic context distillation 和 temporal-aligned residual quantization 都对最终性能具有重要贡献。
