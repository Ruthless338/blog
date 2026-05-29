---
title: APO
date: 2026-05-29
categories:
    - 科研
tags:
    - 强化学习
    - PPO
    - GRPO
---

APO的动机、原理、公式。
<!--more-->
---

## 1. APO解决的问题
APO 关注的是策略更新后的探索空间坍塌问题：

* GRPO 已经可以得到 sequence-level advantage；
* 但当错误回答被负向更新时，释放出的概率质量会主要流向已经高概率的 token；
* 低概率但仍然有效的推理分支可能越来越难被重新采样；
* APO 在负样本更新时，将一部分概率主动拉回 reference model 的高置信候选集合。

---

## 2. 训练背景：RLVR 与 GRPO

RLVR 即 Reinforcement Learning with Verifiable Rewards。

在数学推理任务中：

```text
输入：数学题 Prompt
模型：生成完整推理过程与最终答案
验证器：判断最终答案是否正确
奖励：正确为 1，错误为 0
```

论文以 GRPO 作为基础训练算法。

对于同一道题，GRPO 采样多条回答：

$$
a_1,a_2,\ldots,a_N
$$

每条回答得到结果奖励：

$$
R_i\in\{0,1\}
$$

然后在同一道题内部计算相对 advantage：

$$
A_i=
\frac{R_i-\mu_g}{\sigma_g+\epsilon}
$$

其中：

$$
\mu_g=\frac{1}{N}\sum_i R_i
$$

因此：

* 高于同题平均表现的回答被强化；
* 低于同题平均表现的回答被惩罚。

GRPO 的优势是避免 token-level Critic；但 APO 指出，即使 advantage 已经是 sequence-level，策略更新本身仍可能导致探索空间收缩。

---

## 3. 评价指标：Pass@1 与 Pass@K

### 3.1 Pass@1

对于一道题 $x$，设模型单次生成正确答案的概率为：

$$
p_x=P(\text{correct}\mid x)
$$

则：

$$
\mathrm{Pass@1}(x)=p_x
$$

它表示单次生成的正确率，也可以理解为策略当前的采样效率。

---

### 3.2 Pass@K

对于同一道题，独立采样 $K$ 次，至少一次正确的概率为：

$$
\mathrm{Pass@K}(x)=1-(1-p_x)^K
$$

因此，对同一道固定问题、同一个固定策略而言：

$$
\mathrm{Pass@1}(x)\text{ 固定}
\;\Rightarrow\;
\mathrm{Pass@K}(x)\text{ 固定}
$$

即 Pass@K 不能单独证明同一道题存在更多不同的正确推理路径。


## 4. APO 发现的问题：Squeezing Effect

考虑当前模型在某一 token 位置的分布。假设模型生成了一个错误 token $y_{\mathrm{err}}$，负 advantage 会要求降低它的概率。

对 Softmax 策略而言，压低错误 token 时，其他候选 token $k$ 的 logit 增益近似满足：

$$
\Delta z_k
\propto
\pi_\theta(y_{\mathrm{err}})\;\pi_\theta(k)
$$

因此：

$$
\Delta z_k\propto\pi_\theta(k)
$$

含义是：

> 当前概率越高的替代 token，越容易获得更多释放出的概率质量；当前概率很低的 token，即使是正确候选，也几乎得不到恢复。

这是一种 Rich-get-Richer 现象。

例如：

| Token | 含义         | 当前概率 |
| ----- | ---------- | ---: |
| A     | 当前主导路径     | 0.60 |
| B     | 另一有效路径     | 0.20 |
| C     | 冷门但可能有效    | 0.05 |
| D     | 当前错误 token | 0.15 |

当 D 被惩罚后，概率质量主要流向 A，而不是平均分配给 B、C。于是 C 即使有效，也可能越来越难被采样到。

---

## 5. Recursive Space Contraction：递归式探索空间收缩

APO 将强化学习中的探索坍塌过程称为 Recursive Space Contraction，简称 RSC。

RSC 由两种动态共同导致：

### 5.1 Positive Sharpening

当某条正确路径被采样并获得正奖励，其概率被提高。这样能够改善 Pass@1，但同时也会通过 softmax 竞争压低未被采样的其他路径，包括潜在正确路径。

### 5.2 Negative Squeezing

当错误路径被采样并受到负奖励时，标准策略梯度只压低当前错误 token，却没有告诉模型应该把概率恢复给哪些有效替代路径。由于 Squeezing Effect，概率往往进一步集中到当前已经占优的 token 上。

### 5.3 On-policy 自强化循环

GRPO 属于 on-policy 训练，只有当前策略能够采样到的路径才会获得训练反馈。

因此：

```text
有效但低概率路径被压低
→ 更难被采样
→ 更少获得正奖励
→ 概率进一步下降
→ 最终几乎不可恢复
```

这就是 RSC：训练不断剪枝，但一些本来有效的分支也被永久剪掉。

---

## 6. 为什么只保留 Top-1 路径不够

论文对 Qwen2.5-Math-7B 在 1,000 道 MATH 问题上进行 teacher-forcing 分析，检查 ground-truth next token 是否出现在 reference model 的 Top-K 预测中。

结果：

| Reference 支持范围 | 正确 token 覆盖率 |    丢失率 |
| -------------- | -----------: | -----: |
| Top-1          |       83.84% | 16.16% |
| Top-4          |       95.52% |  4.48% |
| Top-8          |       97.47% |  2.53% |
| Top-16         |       98.46% |  1.54% |

该结果说明：

* Reference model 的 Top-1 主路径并不能覆盖所有有效推理 token；
* 若 RL 只强化当前最强路径，可能删除大量潜在有效支持；
* Top-8 已经能够覆盖 97.47% 的正确 token，同时集合规模仍较小。

因此论文默认使用：

$$
K=8
$$

作为 Safe Manifold 的大小。

---

## 7. 为什么 KL Regularization 不够理想

传统 RLVR 常使用固定 reference model，并加入 KL 正则：

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
$$

它要求当前策略整体不要偏离 reference model 太远。

论文将其称为：

$$
\text{Shape Matching}
$$

因为 KL 约束近似要求：

$$
\pi_\theta(y)\approx\pi_{\mathrm{ref}}(y),
\quad \forall y\in\mathcal{V}
$$

也就是说，当前策略需要模仿 reference model 在完整词表上的分布形状。

问题在于：

* Reference model 中也包含噪声与不确定性；
* 强化学习需要让正确路径变得更尖锐，提高 Pass@1；
* KL 会在正确路径已经确认有效时，仍阻止策略充分 sharpening；
* 因此 reward gradient 与 KL gradient 可能产生冲突。

简单说：

```text
Reward：这条正确路径很好，请更自信。
KL：不要太自信，请回到原始分布。
```

论文认为，真正需要保留的不是 reference 的完整分布形状，而是其中潜在有效的支持范围。

---

## 8. APO 的核心思想：Support Coverage

APO 不要求当前策略完整模仿 reference model，而只要求：

> 当模型犯错时，概率不要流向任意区域，而应恢复到 reference model 的高置信候选集合。

论文定义 Safe Manifold：

$$
\mathcal{M}_{\mathrm{safe}}=
\{\,y\in\mathcal{V}\mid y\in\operatorname{TopK}(\pi_{\mathrm{ref}})\,\}
$$

其中：

* $\pi_{\mathrm{ref}}$：固定 reference model；
* $K$：安全候选集合大小，默认 $K=8$。

理想的 Support Coverage 目标为：

$$
J_{\mathrm{support}}(\theta)=
\sum_{y\in\mathcal{M}_{\mathrm{safe}}}
\pi_\theta(y)
$$

它关注的是：

```text
当前策略在安全候选集合上保留了多少总概率质量。
```

区别如下：

| 方法                     | 目标                       |
| ---------------------- | ------------------------ |
| KL / Shape Matching    | 模仿 reference 的完整概率分布     |
| APO / Support Coverage | 保留 reference 高置信集合上的概率覆盖 |

因此，APO 允许模型在 Safe Manifold 内对某条已验证正确的路径变得更加自信，同时避免错误更新将其他潜在有效候选完全清除。

---

## 9. Exclusive Anchoring：为什么要排除当前错误 token

当前回答出错时，被采样的 token $y_t$ 可能本身也属于 reference model 的 Top-K。

若直接把整个 Top-K 都作为恢复目标，则：

```text
负 advantage：压低当前错误 token
Anchor force：又将当前错误 token 当作安全候选拉高
```

二者会产生信号抵消。

因此论文定义 Anchor Set：

$$
\mathcal{S}_{\mathrm{anchor}}=
\operatorname{TopK}(\pi_{\mathrm{ref}})\setminus\{y_t\}
$$

即：

> 在 reference 的 Top-K 候选中，排除当前这一个已经导致负反馈的 token，只将概率拉回其他高置信候选。

这称为 Exclusive Anchoring。

---

## 10. Virtual Anchor Ratio

APO 需要衡量当前策略对 Anchor Set 覆盖了多少概率质量。

首先定义 reference model 在 Anchor Set 上的总概率：

$$
Z_{\mathrm{ref}}=
\sum_{j\in\mathcal{S}_{\mathrm{anchor}}}
\pi_{\mathrm{ref}}(j)
$$

定义归一化权重：

$$
\hat{\omega}_k=
\frac{\pi_{\mathrm{ref}}(k)}{Z_{\mathrm{ref}}}
$$

Virtual Anchor Ratio 为：

$$
r_{\mathrm{anchor}}=
\sum_{k\in\mathcal{S}_{\mathrm{anchor}}}
\hat{\omega}_k\,
\frac{\pi_\theta(k)}{\pi_{\mathrm{ref}}(k)}
$$

展开后：

$$
r_{\mathrm{anchor}}=
\frac{1}{Z_{\mathrm{ref}}}
\sum_{k\in\mathcal{S}_{\mathrm{anchor}}}
\pi_\theta(k)
$$

所以它本质上表示：

> 当前策略在其他安全候选 token 上分配的总概率质量，并由 reference 支持质量进行归一化。

---

## 11. Ratio Rectification：APO 的核心公式

标准 PPO / GRPO 对当前采样 token 使用 ratio：

$$
r_t(\theta)=
\frac{\pi_\theta(y_t)}{\pi_{\mathrm{old}}(y_t)}
$$

当样本为负 advantage 时，标准方法只会压低当前错误 token 的 ratio。

APO 对负样本定义修正后的 ratio：

$$
\tilde{r}_{\mathrm{APO}}=
\lambda\,
\frac{\pi_\theta(y_t)}{\pi_{\mathrm{old}}(y_t)}
\;-\;
\beta\,r_{\mathrm{anchor}}
$$

其中：

| 符号                    | 含义                        |
| --------------------- | ------------------------- |
| $\lambda$             | Push 系数：增强对当前错误 token 的压制 |
| $\beta$               | Pull 系数：将概率拉向其他安全候选       |
| $r_{\mathrm{anchor}}$ | 当前策略覆盖安全候选的程度             |

默认超参数：

$$
\lambda=1.05,\qquad
\beta=0.1,\qquad
K=8
$$

对于负 advantage 样本，最小化修正 ratio 产生两种效果：

### Push Force

$$
\frac{\pi_\theta(y_t)}{\pi_{\mathrm{old}}(y_t)}\;\downarrow
$$

即降低当前错误 token 的概率。

### Pull Force

$$
r_{\mathrm{anchor}}\;\uparrow
$$

即提高其他高置信候选 token 的总概率。

因此 APO 的错误纠正过程不是：

```text
只把错误 token 压低，然后让概率盲目流动
```

而是：

```text
压低当前错误 token，同时将概率主动拉回潜在安全分支
```

论文将这种机制称为 Elastic Recovery。

---

## 12. 为什么 APO 只在负样本上触发

APO 接受正样本带来的 sharpening：

```text
正确路径得到正奖励
→ 允许其概率进一步增大
→ 有利于提高 Pass@1
```

真正危险的是错误样本的负向更新，因为标准负反馈只会压错误，却不会指明恢复方向。

因此：

| 样本类型        | APO 行为                        |
| ----------- | ----------------------------- |
| 正 advantage | 保持标准 GRPO 更新，让正确路径 sharpening |
| 负 advantage | 启用 Push + Pull，抑制错误并恢复安全支持    |

这也是 APO 相比全局 KL 更合理的地方：它不会在模型已经生成正确路径时强行将策略拉回 reference 的完整分布。

---


## 13. 方法优点

APO 的主要优点包括：

1. **直接针对负样本更新的恢复方向。**
   它不是简单提高 entropy，而是在发生错误时把概率拉回 reference 的高置信候选区域。

2. **避免全局 KL 对正确 sharpening 的干扰。**
   正确样本仍可变得更自信，从而提高 Pass@1。

3. **同时改善效率与覆盖。**
   在主要实验中，APO 同时提高 Avg Pass@1 与 Avg Pass@K。

4. **计算约束稀疏。**
   APO 只使用 Top-K anchor tokens，而 KL 涉及完整词表分布。

5. **与 PPO / GRPO 一类 ratio-based policy gradient 方法兼容。**
   它修改的是负 advantage 时的 ratio，不强依赖某一种 advantage estimator。

---


## 14. 一句话总结

APO 认为 RLVR 中性能提升不能以有效探索空间永久坍塌为代价；它用 reference model 的 Top-K 高置信候选构成 Safe Manifold，仅在错误更新时通过 Push-Pull Ratio Rectification 压低当前错误 token 并恢复其他安全候选的概率质量，从而在提高 Pass@1 的同时维持更好的成功覆盖与生成分布健康度。
