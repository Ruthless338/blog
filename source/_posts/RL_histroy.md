---
title: 强化学习调查
date: 2026-09-01
categories:
    - 科研
tags:
    - 强化学习
mathjax: true
---
强化学习的部分发展。

<!--more-->

2025 年 1 月：DeepSeek-R1

在 R1 之前，大模型里的 RL 给人的典型印象还是 RLHF：

$$ \text{LLM}\rightarrow \text{Human Preference / Reward Model}\rightarrow \text{PPO} $$

核心目标是：“让模型回答得更符合人的偏好。”

而 R1 把 RL 变成了另一件事情：

$$ \text{LLM}\rightarrow \text{solve problem}\rightarrow \text{verifier}\rightarrow R $$

比如数学题：

$$ R= \begin{cases} 1,&\text{答案正确}\\ 0,&\text{答案错误} \end{cases} $$

代码则可以直接运行 testcase。

所以后来大家把这套 paradigm 称为：

RLVR — Reinforcement Learning with Verifiable Rewards

DeepSeek-R1-Zero 的 reward 主要就是 correctness reward + format reward，并没有训练一个 neural process reward model；DeepSeek 给出的原因之一，就是 neural reward model 在大规模 RL 中可能被 reward hacking，而且增加额外训练成本。
这是一种非常激进的尝试，整个reasoning process不监督，只验证最终答案y，结果随着 RL 训练，模型平均 response length 自己增加，而且开始出现 reflection、重新尝试 alternative approach 等行为，大幅提高了AIME2024的pass@1
整个R1-Zero证明通过RL和Sparse outcome reward可以激发模型的reasoning capability。
但也引发了2025年之后几乎所有问题：
1. token-level credit assignment：所有token，包括中间走错的token、并无实际意义的token和真正有效的token都得到了positive advantage。这是token-level / step-level / sequence-level credit assignment问题的源头
2. R1的环境十分特殊，整个response的所有steps可以看成一个action，中间的token并不会直接改变外部世界，问题可以近似看成contextual bandit，所以sequence-level reward是有效的。但这个假设到了 Agent 里会直接坏掉，因为$a_t$真的改变了下一步observation。

$$ s_t \xrightarrow{a_t=\text{search}} o_{t+1} \xrightarrow{} s_{t+1}. $$


2025年3月：DAPO
DAPO 在 Qwen2.5-32B Base 上公开实现了大规模 reasoning RL，并给出四个核心修改：Clip-Higher、Dynamic Sampling、Token-Level Policy Gradient Loss、Overlong Reward Shaping。
1. Clip-Higher —— exploration 开始成为核心问题
PPO/GRPO 有 importance ratio：
$$ \rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)} {\pi_{\theta_{\rm old}}(a_t|s_t)}. $$
然后 clipping：
$$ \min \left[ \rho_tA_t, \operatorname{clip}( \rho_t,1-\epsilon,1+\epsilon )A_t \right]. $$
目的是让 policy 不要一次更新得太猛。
但 DAPO 观察到entropy collapse，即 policy 很快变得 deterministic：
$$ \pi(a^\star|s)\approx1. $$
之后同一个题采 8 个答案：
$$ y_1\approx y_2\approx\dots\approx y_8. $$
这时候 RL 最大的问题甚至已经不是 optimization，而是没有新的 trajectory 可以学了。
所以 DAPO 把上下 clipping 解耦：
$$ 1-\epsilon_{\rm low} \le \rho \le 1+\epsilon_{\rm high}, $$
并让：
$$ \epsilon_{\rm high}>\epsilon_{\rm low}. $$
官方解释就是通过扩大 upper clip range，减轻 entropy collapse、维持更多 diversity/exploration。
2. Dynamic Sampling
DAPO 因此直接过滤 accuracy = 0 或 accuracy = 1 的 prompt groups，并不断补采样，保证 batch 中保留足够多有 reward variance 的 prompt
3. Token-Level Policy Gradient Loss
设第 \(i\) 条 response 长度 $ T_i. $
设每个 token 的 clipped policy loss 为：$ \ell_{i,t}. $
一种 sequence-normalized aggregation 是：
$$ L = \frac1G \sum_i \frac1{T_i} \sum_t\ell_{i,t}. $$
这样所有response无论长短在Loss计算时权重基本相同。
DAPO 使用 token-level aggregation：
$$ L= \frac{ \sum_i\sum_t\ell_{i,t} }{ \sum_iT_i }. $$
于是长 response 里 token 多：
$$ T_i\uparrow \Rightarrow \text{trajectory contribution}\uparrow. $$
官方 DAPO recipe 将这一点称为 Token-Level Policy Gradient Loss，即跨 batch 中所有有效 token 聚合，而不是先对每条 response 做等权平均。
这种Loss认为long CoT是更重要的，但并不一定，究竟什么才应该是 policy optimization 的基本单位？这是一个一直在研究的重要问题。
4. reward noise
DAPO 因而采用更平滑的 overlong punishment，而不是简单把**截断** trajectory 当成纯粹错误；官方实现也明确将其定位为降低 reward noise、稳定训练。

2025年3月：Search-R1
R1：

$$ x \rightarrow \text{think} \rightarrow answer. $$

Search-R1：

$$ x \rightarrow \text{think} \rightarrow \boxed{\text{search query}} \rightarrow \boxed{\text{search result}} \rightarrow \text{think} \rightarrow search \rightarrow answer. $$

模型第一次不只是“在自己的 hidden knowledge 里想，”而是：主动决定何时访问 environment
R1 近似：

$$ x\rightarrow y\rightarrow R. $$

但 Search-R1 已经变成：

$$ s_t \rightarrow a_t^{search} \rightarrow o_{t+1} \rightarrow s_{t+1}. $$

例如：

$$ a_t= \text{“Search: Nobel Prize physics 2024”} $$

environment 返回：

$$ o_{t+1}= \text{search result}. $$

现在$ s_{t+1} $取决于$ a_t. $
这就真正开始有MDP的出现：
$$ P(s_{t+1}|s_t,a_t). $$

而且 Search-R1 只用 final answer reward 就能显著提升 multi-turn retrieval reasoning。论文报告在多个 QA 数据集上，相比相应 baseline，Qwen2.5-7B、3B 和 Llama3.2-3B 分别有明显收益。
但 Search-R1 也立刻暴露了一个新问题：过程中的所有search action显然不都是有用的。
所以从这里开始：

$$ \textbf{credit assignment} $$

不再只是理论洁癖，而变成 Agent 的真实瓶颈。

2025年4月：RAGEN
在静态 reasoning 里，简单 outcome reward 可以诱导出有意思的 reasoning；但在 multi-turn Agent 环境里，如果缺少更细粒度、reasoning-aware 的反馈，模型容易形成 shallow strategy，甚至出现 hallucinated reasoning。

2025年5-6月：RL 到底“学”了什么？
RLVR 可能主要不是创造全新的 reasoning path，而只是把 base model 原来极低概率的正确路径提高概率。
正确 reasoning 可能早就在 base distribution 里面。
RL 做的是：
$$ \boxed{\text{probability mass redistribution}} $$
而不是：
$$ \boxed{\text{new reasoning creation}} $$

一个长 CoT 有几千个 token，但真正 entropy 很高的 token 其实只是少数。一个长 CoT 有几千个 token，但真正 entropy 很高的 token 其实只是少数。

2025年7月：GSPO-"Reward 给 sequence，为什么 optimization 却在 token 上做？”
在GRPO中，先计算sequence-level Advantage，再广播到整个response的所有token上，计算每一个token的importance ratio并Clipping之后求和作为loss。在这里Reward/Advantage是sequence-level的，但是importance ratio和clipping是token-level的。
GSPO 提出“optimization unit 应该和 reward unit 对齐”，将 sequence likelihood 用于 importance ratio 和 clipping。
首先算整条 sequence 的 likelihood ratio：

$$ \frac{ \pi_\theta(y_i|x) }{ \pi_{\theta_{\rm old}}(y_i|x) }. $$

由于：

$$ \pi(y_i|x) = \prod_{t=1}^{T} \pi(y_{i,t}|x,y_{i,<t}), $$

直接乘会非常夸张，所以做 length normalization：

$$ s_i(\theta) = \left( \frac{\pi_\theta(y_i|x)} {\pi_{\theta_{\rm old}}(y_i|x)} \right)^{1/T}. $$

等价于：

$$ s_i = \exp \left[ \frac1T \sum_t \log \frac{ \pi_\theta(y_{i,t}|...) }{ \pi_{\rm old}(y_{i,t}|...) } \right]. $$

然后对**整个 sequence**进行 clipping：

$$ L_{\rm GSPO} = \frac1G \sum_i \min \left[ s_iA_i, \operatorname{clip}(s_i,1-\epsilon,1+\epsilon)A_i \right]. $$


2025年8月：Agent Lighting
Agent Lightning 没有强行把 agent 当成：

$$ \text{prompt}\rightarrow\text{completion}. $$

而应该把它恢复成 RL：

$$ (s_t,a_t,r_t,s_{t+1}). $$

Agent runtime 负责产生：

$$ \tau= (s_0,a_0,s_1,a_1,\dots). $$

然后 credit assignment 模块：

$$ \tau \rightarrow \{ \text{trainable transition} \}. $$

这意味着一个很大的转变：

RL 不再只是“训练一个语言模型 completion”，而开始变成“训练一个运行中的 Agent policy”。

Agent Lightning 允许复杂 multi-agent 和 dynamic workflow，并把 execution trajectory 分解成训练 transition。

sequence-level work 并不是认为 token-level RL 永远错误，而是认为 RL abstraction 应该和真正发生 environment transition 的粒度一致。

2025年10月：IGPO-Information Gain-based Policy Optimization

假设答案是：

$$ y^\star. $$

Agent 还没 search 之前，对正确答案的 belief：

$$ P(y^\star|h_t)=0.2. $$

执行搜索后：

$$ P(y^\star|h_{t+1})=0.6. $$

那么这一步应该得到正 reward。

所以它用“模型对正确答案的 belief 增量”作为 turn-level information-gain reward，再和最终 outcome supervision 结合。论文强调这种信号直接来自模型自身 belief update，不需要额外 process reward model 或大量 Monte-Carlo value estimation。

于是：

$$ R_{\rm final} $$

被变成：

$$ r_1^{IG},r_2^{IG},...,r_T^{IG},R_{\rm final}. $$

也就是sparse reward→dense credit
	​
简单来说IGPO想做的是让模型知道做完这一步之后，是否离最终答案更近了？

但同时IGPO也存在问题：短期来看某action的IG是负的，但长期它可能排除了某个假设，是有益的。所以information gain并不等于exploration value


GSPO

问：

$$ \boxed{ \text{optimization unit 是什么？} } $$

回答：

$$ \text{sequence}. $$

因为 reward 是 sequence-level。

Agent Lightning

问：

$$ \boxed{ \text{Agent trajectory 的 action unit 是什么？} } $$

回答：

不能再简单认为是一整个 completion。

需要恢复：

$$ (s_t,a_t,s_{t+1}). $$
IGPO

问：

$$ \boxed{ \text{reward/credit unit 是什么？} } $$

回答：

至少应该开始考虑：

$$ \text{turn-level progress}. $$

因此整个趋势：

$$ \boxed{ \text{sequence} \rightarrow \text{turn/action} \rightarrow \text{state transition} } $$

2026年
总结为以下四大主线：

一、RL的基本单位是什么？
CAPO

CAPO 的两个 action-aligned 设计

第一个：

Action-level value / advantage

只在真正的 **action boundary** 估计：

$$ V(s_t) $$

然后：

$$ A_t= r_t+\gamma V(s_{t+1})-V(s_t) $$

这一整个 action 内：

$$ a_t=(y_{t,1},...,y_{t,L}) $$

共享这个 advantage。

第二个：

Action-aware importance ratio

不直接用每个 token 独立 ratio：

$$ \rho_{t,j} = \frac{\pi(y_{t,j})}{\pi_{\rm old}(y_{t,j})} $$

而是把整个 action 内 token ratio 聚合，并做长度校准。



$$ \boxed{ \text{Optimization granularity should align with causal interaction boundaries.} } $$

对于静态 reasoning：

$$ x\rightarrow CoT\rightarrow R $$

sequence abstraction 很合理。

对于：

$$ Agent\rightarrow Tool\rightarrow Environment $$

action boundary 很自然。

对于 multi-agent：

$$ Agent_A\rightarrow Agent_B $$

可能 message / communication turn 又变成新的 causal unit。

这比“哪个 PO 最先进”重要得多。

二、Exploration

2026年1月 Tree-GRPO ICLR

Tree-GRPO在一些agent state做分支，这样复用之前的prefix，有点类似于world model的多次rollout，但wm里是想象未来，做planning，而Tree-GRPO是真正policy执行。但Tree-GRPO并没有做出精确的判断——哪一个state才最值得branch？

2026年5月-$T^2PO$-ICML Spotlight

token 层，如果模型一直处于一种低效 uncertainty evolution，就触发额外 thinking intervention；

turn 层，如果某一步 interaction 几乎没有 exploration progress，就重新 sample。

2026年7月 IGRPO
按照IG来分配rollout，同时证明这种 information-gain-guided rollout 会诱导出一个teacher trajectory distribution

假设当前 policy 本来产生 trajectory 的分布是：

$$ \pi_\theta(\tau) $$

比如三条轨迹：

$$ \pi_\theta(\tau_1)=0.5,\quad \pi_\theta(\tau_2)=0.3,\quad \pi_\theta(\tau_3)=0.2 $$

但经过 Information Gain 评价以后发现：

$$ V(\tau_1)=0.1,\qquad V(\tau_2)=1.0,\qquad V(\tau_3)=0.5 $$

那么 IGRPO 的 adaptive rollout 会更频繁地扩展 \(\tau_2\) 所在的 branch。最终相当于产生了一个新的分布：

$$ \mu_\theta(\tau) \propto \pi_\theta(\tau)\exp(\gamma V(\tau)) $$

也就是：

$$ \boxed{ \text{teacher distribution} = \text{原 policy distribution} \times \text{trajectory quality weighting} } $$

这里 \(\mu_\theta\) 就是所谓的 teacher trajectory distribution

APO
RL 训练久了以后，policy 的概率分布会越来越尖，很多原本合理的 reasoning/action 分支概率逐渐掉到接近 0，于是以后再也采不到这些分支。

传统做法是加 KL，让当前 policy 不要离 reference model 太远：
$$ D_{KL}(\pi_\theta\|\pi_{\rm ref}) $$
但 KL 的问题是它要求整个概率分布形状都尽量像 reference，这会和 RL 的目标冲突，因为 RL 本来就希望：
$$ \text{正确路径概率大幅增加} $$
KL 却一直说：
“不要变得太尖，要和原始模型差不多。”
所以 APO 换了一个思路：

RL 不应该要求 policy 一直像 reference model，而应该允许它在正确路径上大胆 sharpen，同时保住 reference model 中合理候选的 support，避免 exploration space 不可逆坍缩。