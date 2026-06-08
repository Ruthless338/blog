---
title: RLHF与DPO
date: 2026-06-08
categories:
    - 科研
tags:
    - 强化学习
    - DPO
    - RLHF
---
DPO，全称 **Direct Preference Optimization**，可以理解为：**不用显式训练奖励模型、也不用 PPO，而是直接用偏好数据微调 LLM，让模型更偏向人类喜欢的回答。**

它是 RLHF 的一种简化替代方案。
<!--more-->
---

## 1. 先从 RLHF 说起

传统 RLHF（Reinforcement Learning from Human Feedback）通常分三步：

### 第一步：SFT

先用高质量问答数据做监督微调：

$$
x \rightarrow y
$$

其中 $x$ 是 prompt，$y$ 是理想回答。

---

### 第二步：训练 Reward Model

给同一个 prompt，人工标注两个回答哪个好：

$$
(x, y_w, y_l)
$$

其中：

- $y_w$：winner，人类更喜欢的回答；
- $y_l$：loser，人类不喜欢的回答。

然后训练奖励模型 $r_\phi(x,y)$，希望：

$$
r_\phi(x, y_w) > r_\phi(x, y_l)
$$

---

### 第三步：用 PPO 优化策略模型

让当前语言模型 $\pi_\theta$ 最大化奖励模型给出的 reward，同时不能偏离原来的 SFT 模型太远：

$$
\max_{\pi_\theta} \mathbb{E}_{y \sim \pi_\theta} [r_\phi(x,y)] - \beta D_{KL}(\pi_\theta \parallel \pi_{\text{ref}})
$$

这里的 $\pi_{\text{ref}}$ 通常是 SFT 模型。

问题是：**PPO 很难训，工程复杂，超参数敏感，还容易不稳定。**

DPO 就是为了解决这个问题。

---

## 2. DPO 的核心思想

DPO 的想法非常直接：

> 既然我们的目标是让模型更喜欢 $y_w$，更不喜欢 $y_l$，那为什么不直接优化这个偏好关系？

也就是说，不再显式训练 reward model，也不再使用 PPO，而是直接用偏好数据训练语言模型。

数据仍然是：

$$
(x, y_w, y_l)
$$

DPO 希望模型满足：

$$
\pi_\theta(y_w|x) > \pi_\theta(y_l|x)
$$

但它不是简单地让 winner 概率变大、loser 概率变小，而是会参考一个固定的 reference model。

---

## 3. DPO 的目标函数

DPO 的 loss 是：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)}
\left[
\log \sigma
\left(
\beta
\left[
\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
- \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\right]
\right)
\right]
$$

看起来有点复杂，但可以拆开理解。

---

## 4. 公式直觉

里面最核心的是这一项：

$$
\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
- \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

也就是：

$$
\bigl[ \log \pi_\theta(y_w|x) - \log \pi_{\text{ref}}(y_w|x) \bigr]
- \bigl[ \log \pi_\theta(y_l|x) - \log \pi_{\text{ref}}(y_l|x) \bigr]
$$

它比较的是：

> 当前模型相对于 reference model，到底更提升了 winner，还是更提升了 loser？

DPO 不只是要求：

$$
\pi_\theta(y_w|x) > \pi_\theta(y_l|x)
$$

而是要求：

$$
\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
>
\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
$$

也就是说：

> 相比原来的 SFT 模型，新模型应该更偏向 winner，而不是 loser。

这就相当于把 KL 约束隐式地融入到了目标函数里面。

---

## 5. 为什么需要 reference model？

假设没有 reference model，我们可能只优化：

$$
\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)
$$

这会有一个问题：模型可能为了让 winner 和 loser 拉开差距，疯狂降低 loser 的概率，甚至造成语言能力退化。

而 reference model 的作用是：

> 约束当前模型不要离原来的 SFT 模型太远。

所以 DPO 里的 $\pi_{\text{ref}}$ 类似于 PPO 里的 KL reference model。

在实际训练中：

- $\pi_\theta$：正在训练的模型；
- $\pi_{\text{ref}}$：冻结的 SFT 模型；
- 二者通常初始化相同。

---

## 6. $\beta$ 的作用

DPO 里有一个重要超参数：

$$
\beta
$$

它控制当前模型可以偏离 reference model 的程度。

直觉上：

- $\beta$ 大：更强地优化偏好差异，winner 和 loser 拉得更开；
- $\beta$ 小：训练更保守，更接近 reference model。

不过不同论文和实现中对 $\beta$ 的解释会有一点表述差异，核心就是：**它控制偏好优化强度和 KL 约束之间的平衡。**

---

## 7. DPO 和 Reward Model 的关系

DPO 并不是完全抛弃 reward 的思想。

它实际上利用了一个重要结论：

在带 KL 约束的 RLHF 目标下，最优策略和 reward 之间有关系：

$$
r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

其中 $Z(x)$ 是归一化项。

这个式子的意思是：

> 如果一个回答的 reward 高，那么最优策略相对于 reference model，会更倾向于生成它。

DPO 利用这个关系，把 reward model 隐式替换成了：

$$
\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}
$$

所以 DPO 不是没有 reward，而是：

> 不显式训练 reward model，而是把 reward 直接参数化为 policy 和 reference model 的 log-prob ratio。

---

## 8. DPO 的训练过程

实际训练时，一条数据是：

$$
(x, y_w, y_l)
$$

模型分别计算：

$$
\log \pi_\theta(y_w|x)
$$

和

$$
\log \pi_\theta(y_l|x)
$$

reference model 也计算：

$$
\log \pi_{\text{ref}}(y_w|x)
$$

和

$$
\log \pi_{\text{ref}}(y_l|x)
$$

然后带入 DPO loss：

$$
-\log \sigma
\left(
\beta
\left[
\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
- \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\right]
\right)
$$

如果当前模型已经明显更偏向 winner，那么 loss 小。

如果当前模型对 loser 的提升更大，loss 就大，梯度会推动模型：

- 增大 winner 的 log-prob；
- 减小 loser 的 log-prob。

---

## 9. DPO 的优点

DPO 最大的优点是简单。

相比 PPO-RLHF，它不需要：

- 显式训练 reward model；
- 在线采样；
- PPO clipping；
- value model；
- advantage estimation；
- 复杂 RL 训练流程。

它本质上更像一个 supervised fine-tuning loss，只不过数据格式是 pairwise preference。

所以工程上更稳定、更容易复现。

---

## 10. DPO 的缺点

DPO 也不是完美的。

它的主要问题有：

### 第一，依赖偏好数据质量

如果 winner/loser 数据质量不好，DPO 会直接学歪。

比如：

$$
y_w
$$

只是比 $y_l$ 稍微好一点，但本身也不优秀，模型也会被迫偏向它。

---

### 第二，容易过优化偏好数据

如果偏好数据分布比较窄，DPO 可能会让模型在这些风格上过拟合。

例如变得特别啰嗦、模板化、过度迎合人类偏好。

---

### 第三，不是真正意义上的 online RL

PPO 可以让模型在线采样新回答，再根据 reward model 反馈继续优化。

DPO 通常是 offline 的，它只在已有的偏好数据上训练。

所以它更像：

> offline preference optimization

而不是完整的 online RL。

---

### 第四，对多样性和探索能力较弱

因为 DPO 只比较已有的 winner/loser，它没有显式探索机制。

这也是后来很多方法，比如 IPO、KTO、ORPO、SimPO、APO、SPPO 等继续改进的原因之一。

---

## 12. DPO 和 PPO 的区别

| 方法       | 是否需要 Reward Model | 是否需要 RL 训练 | 是否在线采样 | 稳定性 |
| ---------- | --------------------: | --------------: | ----------: | -----: |
| PPO-RLHF   |                  需要 |            需要 |    通常需要 |   较难 |
| DPO        |            不需要显式 RM |       不需要 PPO |   通常不需要 |  较稳定 |

可以粗略理解为：

> PPO-RLHF 是"先学一个打分器，再用 RL 让模型拿高分"；
> DPO 是"直接让模型更偏向人类选择的答案"。

---

## 13. RLHF / DPO / RLVR 对比

RLHF 的 reward 通常来自人类偏好，需要人工偏好数据，传统 PPO-RLHF 还会训练显式 Reward Model，并用 SFT reference model 做 KL 约束；DPO 仍然使用人类偏好数据，但不显式训练 Reward Model，也不用 PPO；RLVR 的 reward 来自可自动验证的规则或环境，不依赖人类偏好标注，通常不需要训练 Reward Model。

RLVR 需要一个能自动验证输出质量的 reward function / verifier。它可以只给最终答案的 outcome reward，也可以给中间步骤的 process reward。计算每一步 action 的 advantage 时，可以把最终 reward 作为未来累计回报传播到每一步 action，因此不一定需要环境显式给出每一步 reward。
