---
title: 强化学习
date: 2026-05-27
categories:
    - 科研
tags:
    - 强化学习
---

强化学习笔记，包括概念定义、REINFORCE、Actor-Critic、PPO 以及相关代码
<!--more-->

## 一、基本概念

强化学习中的交互过程通常写成：

$$s_t \rightarrow a_t \rightarrow r_t,\; s_{t+1}$$

其中：
- $s_t$：时刻 $t$ 的状态（state）
- $a_t$：智能体采取的动作（action）
- $r_t$：执行动作后获得的奖励（reward）
- $s_{t+1}$：环境转移后的下一状态

## 二、MDP 马尔可夫决策过程

强化学习通常使用 Markov Decision Process（MDP）来建模。

一个 MDP 可以表示为：

$$M = (S, A, P, R, \gamma)$$

### 2.1 状态空间 $S$

状态是环境在当前时刻的信息。

在机器人操作中，状态可能包括：

$$s_t = \{\text{机械臂位置},\; \text{物体位置},\; \text{相机图像},\; \text{夹爪状态}\}$$

### 2.2 动作空间 $A$

动作是智能体能够选择的操作。例如：

- 游戏中：向左、向右、跳跃
- 机械臂中：末端位姿变化、夹爪闭合程度
- 自动驾驶中：转向、加速、刹车

动作可以是：

- **离散动作**：$a_t \in \{\text{左移},\; \text{右移},\; \text{抓取}\}$
- **连续动作**：$a_t = (\Delta x,\; \Delta y,\; \Delta z,\; \Delta \theta)$

机器人控制通常更接近连续动作空间。

### 2.3 状态转移概率 $P$

环境根据当前状态和动作，决定下一状态：

$$P(s_{t+1} \mid s_t, a_t)$$

### 2.4 奖励函数 $R$

奖励衡量动作结果是否有利：

$$r_t = R(s_t, a_t, s_{t+1})$$

### 2.5 折扣因子 $\gamma$

强化学习关心的是未来累计奖励，但未来奖励通常会被折扣：

$$\gamma \in [0, 1]$$

- $\gamma = 0$：只关心眼前奖励
- $\gamma \approx 1$：非常关心长期收益

假设 $\gamma = 0.9$，三步后的 $+10$ 奖励在当前看来价值为：

$$0.9^3 \times 10 = 7.29$$

在机器人长程任务中，通常需要较大的 $\gamma$，否则智能体只会追求短期利益。

## 三、马尔可夫性质与 POMDP

MDP 中的 M 来自 Markov，即马尔可夫性质：

$$P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \cdots) = P(s_{t+1} \mid s_t, a_t)$$

直观来说：**当前状态已经包含了预测未来所需要的全部信息**，因此不需要再额外查看完整历史。

但是在具身智能或视觉任务中，单帧图像可能并不满足马尔可夫性质。这类问题更准确地说属于 **POMDP**：部分可观测马尔可夫决策过程。

此时模型获得的通常不是完整状态 $s_t$，而是观测 $o_t$：

$$o_t \neq s_t$$

具身智能中的视觉输入、语言指令、历史动作序列，很多时候都是为了帮助模型推断隐藏状态。

## 四、Return 与优化目标

### 4.1 Return（累计折扣奖励）

单步奖励 $r_t$ 太短视，所以我们定义从时刻 $t$ 开始的累计折扣奖励，称为 Return：

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$$

### 4.2 优化目标

强化学习想学习一个策略 $\pi_\theta$，让从初始状态开始的期望累计回报尽可能大：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

其中：
- $\theta$：策略网络参数
- $\tau \sim \pi_\theta$：轨迹由当前策略采样产生
- $J(\theta)$：当前策略的平均表现

所以训练目标就是：

$$\max_\theta J(\theta)$$

也就是说：**调整策略网络，让它更容易产生高奖励轨迹。**

## 五、价值函数

只知道最终奖励还不够，我们还希望知道：
- 当前这个状态有多好？
- 当前状态下采取某个动作有多好？

因此引入价值函数。

### 5.1 状态价值函数 $V^\pi(s)$

从状态 $s$ 出发，之后一直遵循策略 $\pi$，预计能够获得多少累计回报：

$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right]$$

### 5.2 动作价值函数 $Q^\pi(s, a)$

在状态 $s$ 下先执行动作 $a$，之后再遵循策略 $\pi$，预计能够获得多少累计回报：

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]$$

### 5.3 优势函数 $A^\pi(s, a)$

优势函数衡量某个动作相对于当前策略平均水平到底好多少：

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

优势函数在 Actor-Critic 和 PPO 中非常重要。

## 六、Bellman 方程

价值函数之所以重要，是因为它满足递推关系。

Return 本身可以写成：

$$G_t = r_t + \gamma G_{t+1}$$

因此状态价值函数也满足：

$$V^\pi(s_t) = \mathbb{E}_\pi \left[ r_t + \gamma V^\pi(s_{t+1}) \right]$$

这就是 **Bellman Expectation Equation** 的核心形式。

直观理解：**当前状态的价值 = 当前一步奖励 + 折扣后的下一状态价值。**

## 七、Policy Gradient

我们已经知道目标是最大化：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right]$$

其中 $R(\tau)$ 表示整条轨迹的累计奖励。

Policy Gradient 的经典形式为：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, G_t \right]$$

## 八、REINFORCE 算法

REINFORCE 是最经典、最直接的策略梯度算法。

它的特点是：**先完整采样一条或多条轨迹，再根据实际得到的累计回报更新策略。**

损失函数即奖励取负：

$$\mathcal{L} = -\log \pi_\theta(a_t \mid s_t) \, G_t$$

### 8.1 REINFORCE 的最大问题：方差很大

REINFORCE 虽然逻辑简单，但训练通常很不稳定。原因是：一次轨迹的最终奖励可能受到很多随机因素影响，而算法会把这个结果归因给整条轨迹中的动作。

一个更严重的问题：**所有动作都背锅或邀功。**

假设机器人完成任务需要 20 步。最终成功得到 $+10$，REINFORCE 会倾向于鼓励前面执行过的每一个动作。但其中可能有一些动作其实没什么作用，甚至是多余的。反过来，最终失败时，前面某些其实很合理的动作也可能一起被惩罚。

## 九、Actor-Critic

REINFORCE 中，策略网络负责执行动作，但没有一个网络专门评价当前状态。Actor-Critic 增加了一个 Critic 网络。

### 9.1 Actor（行动者）

负责产生动作：

$$\pi_\theta(a \mid s)$$

### 9.2 Critic（评论家）

负责预测状态价值：

$$V_\phi(s)$$

训练时：
- Actor 根据 Critic 判断动作是否比预期更好
- Critic 学习预测未来回报

### 9.3 Actor 的更新目标

Actor 不再简单使用 $G_t$，而使用优势：

$$A_t = G_t - V_\phi(s_t)$$

损失可以写为：

$$\mathcal{L}_{\text{actor}} = -\log \pi_\theta(a_t \mid s_t) \, A_t$$

### 9.4 Critic 的更新目标

Critic 要尽量准确预测 Return：

$$\mathcal{L}_{\text{critic}} = \bigl( V_\phi(s_t) - G_t \bigr)^2$$

也就是说：
- Actor 学习"怎么做"
- Critic 学习"当前局面有多好"

## 十、TD Error

REINFORCE 需要等一整条轨迹结束，再计算完整 Return。但是在长任务中，这会很慢。

Actor-Critic 常使用一步估计 $r_t + \gamma V(s_{t+1})$ 作为当前状态价值的学习目标。

于是定义 TD Error：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

它可以近似看作当前动作的优势：

$$A_t \approx \delta_t$$

直观来说：
- 如果进入下一状态后，实际表现比 Critic 原先预期更好，则 $\delta_t > 0$
- 如果结果比预期更差，则 $\delta_t < 0$

这使得模型不必等到完整任务结束，就能逐步学习。

## 十一、PPO（Proximal Policy Optimization）

PPO 的基本出发点是：Policy Gradient 虽然能优化策略，但一次更新太大时，策略可能突然崩掉，因此需要限制新策略偏离旧策略太多。

PPO 关注新旧策略对同一个动作的概率变化：

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

### 11.1 PPO 的裁剪目标

PPO 使用 clipping 限制变化：

$$\mathcal{L}^{\text{CLIP}} = \mathbb{E} \left[ \min \Bigl( r_t(\theta) A_t,\; \operatorname{clip}\bigl(r_t(\theta),\; 1-\epsilon,\; 1+\epsilon\bigr) A_t \Bigr) \right]$$

PPO 本质上仍然是 Policy Gradient，只是在更新策略时加入了保护机制，避免策略一步变化太大。

## 十二、核心公式总结

下面展示 PPO 训练流程中最重要的四组公式，它们与代码一一对应。

训练流程回顾：

1. Actor 输入状态 $s_t$，采样动作 $a_t$
2. Critic 同时预测 $V_{\text{old}}(s_t)$
3. 环境执行 $a_t$，返回 reward $r_t$ 与下一状态 $s_{t+1}$
4. 将 state、action、reward、value、old_log_prob 保存到 buffer
5. rollout 收集完成后，倒序计算 TD Error
6. 使用 TD Error 累积得到 GAE Advantage
7. 使用 Advantage + old Value 得到 Return，训练 Critic
8. 使用新旧策略概率比与 clipping，训练 Actor
9. 清空 buffer，用更新后的策略重新采样

### 公式一：TD Error

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### 公式二：GAE Advantage

$$\hat{A}_t = \delta_t + \gamma\lambda \, \hat{A}_{t+1}$$

### 公式三：新旧策略概率比

$$\mathrm{ratio}_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$

### 公式四：PPO Clipped Loss

$$\mathcal{L}_{\text{PPO}} = -\min\Bigl( \mathrm{ratio}_t \cdot \hat{A}_t,\; \operatorname{clip}\bigl(\mathrm{ratio}_t,\; 1-\epsilon,\; 1+\epsilon\bigr) \cdot \hat{A}_t \Bigr)$$

把这四个公式与下方代码一一对应起来，你就已经掌握了 PPO 最核心的 Actor-Critic 训练结构。

## 十三、PPO 代码实现

```cpp
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler

# ============================================================
# 1. 超参数配置
# ============================================================

@dataclass
class PPOConfig:
    # 环境维度
    state_dim: int
    action_dim: int

    # 强化学习超参数
    gamma: float = 0.99          # 折扣因子 γ
    gae_lambda: float = 0.95     # GAE 中的 λ
    clip_eps: float = 0.2        # PPO clipping 范围 ε

    # 损失权重
    value_coef: float = 0.5      # Critic loss 权重
    entropy_coef: float = 0.01   # 熵奖励权重，鼓励探索

    # 优化相关
    learning_rate: float = 3e-4
    update_epochs: int = 10      # 一批 rollout 重复更新多少轮
    mini_batch_size: int = 64
    max_grad_norm: float = 0.5   # 梯度裁剪

    # 数据收集相关
    rollout_steps: int = 2048    # 收集多少步后执行一次 update

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 2. Rollout Buffer
# ============================================================

class RolloutBuffer:
    """
    PPO 是 on-policy 算法。

    每轮训练流程为：
        1. 使用当前策略与环境交互；
        2. 将交互数据保存到 buffer；
        3. 使用这一批数据更新策略若干轮；
        4. 清空 buffer；
        5. 使用更新后的策略重新采样。

    注意：
    - rewards 保存的是环境即时返回的 r_t；
    - returns 与 advantages 在 rollout 收集完成后再计算；
    - old_log_probs 保存"采样时策略"对动作的概率，
      后续 PPO 计算新旧策略概率比时会使用它。
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.values = []
        self.old_log_probs = []

        self.advantages = []
        self.returns = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """
        每执行一次 env.step(action)，就调用一次 add()。

        参数对应：
            state    = s_t
            action   = a_t
            reward   = r_t，由环境返回
            done     = 当前 episode 是否结束
            value    = V_old(s_t)
            log_prob = log π_old(a_t | s_t)
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

        self.values.append(value)
        self.old_log_probs.append(log_prob)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """
        根据已经采样好的 rollout，计算：
            1. TD Error:
               δ_t = r_t + γ V(s_{t+1}) - V(s_t)

            2. GAE Advantage:
               A_t = δ_t + γλ δ_{t+1} + (γλ)^2 δ_{t+2} + ...

            3. Return / Value Target:
               R_t = A_t + V(s_t)

        这里的 last_value 表示：
            如果 rollout 最后一步还没有真正结束，
            就需要用 Critic 预测 V(s_{T+1}) 来 bootstrap。

            如果最后一步刚好 terminal，
            则 last_value = 0。
        """

        num_steps = len(self.rewards)

        self.advantages = [0.0 for _ in range(num_steps)]
        self.returns = [0.0 for _ in range(num_steps)]

        gae = 0.0

        # 从后向前计算，因为当前优势依赖未来的 TD Error
        for t in reversed(range(num_steps)):

            if t == num_steps - 1:
                # rollout 的最后一个位置
                next_value = last_value
            else:
                # 中间位置的下一状态价值，已在采样时保存
                next_value = self.values[t + 1]

            # 如果当前 transition 已经到达终止状态，
            # 那么之后没有未来收益，不应该 bootstrap。
            non_terminal = 1.0 - float(self.dones[t])

            # TD Error:
            # δ_t = r_t + γV(s_{t+1}) - V(s_t)
            td_delta = (
                self.rewards[t]
                + gamma * next_value * non_terminal
                - self.values[t]
            )

            # GAE:
            # A_t = δ_t + γλ A_{t+1}
            gae = (
                td_delta
                + gamma * gae_lambda * non_terminal * gae
            )

            self.advantages[t] = gae

            # Critic 的训练目标：
            # R_t = A_t + V(s_t)
            self.returns[t] = gae + self.values[t]

    def clear(self) -> None:
        """一次 PPO update 完成后，清空旧 rollout。"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()

        self.values.clear()
        self.old_log_probs.clear()

        self.advantages.clear()
        self.returns.clear()

    def __len__(self) -> int:
        return len(self.rewards)

# ============================================================
# 3. Actor-Critic Network
# ============================================================

class ActorCritic(nn.Module):
    """
    Actor:
        输入状态 s_t；
        输出动作分布 π(a | s)。

    Critic:
        输入状态 s_t；
        输出状态价值 V(s_t)。

    这里使用两个独立 MLP，便于理解 Actor 与 Critic 的分工。
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        # ----------------------------------------------------
        # Actor：输出每个离散动作的 logits
        # ----------------------------------------------------
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )

        # ----------------------------------------------------
        # Critic：输出一个标量 V(s)
        # ----------------------------------------------------
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        仅计算 Critic 价值：
            V(s)
        """
        return self.critic(state).squeeze(-1)

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        """
        统一完成：
            1. 根据 Actor 构造动作分布；
            2. 采样动作，或者评价给定动作；
            3. 计算 log_prob；
            4. 计算 entropy；
            5. 计算 Critic value。

        两种使用场景：

        场景 A：与环境交互时
            action=None
            网络会采样动作 a_t。

        场景 B：更新 PPO 时
            action 为 buffer 中保存的旧动作
            网络会计算"新策略下这些旧动作的概率"。
        """

        logits = self.actor(state)

        # 使用 logits 而不是手动 Softmax：
        # Categorical 内部会稳定地处理 softmax 与 log_prob。
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.get_value(state)

        return action, log_prob, entropy, value

# ============================================================
# 4. PPO Agent
# ============================================================

class PPOAgent:
    """
    核心职责：
        1. act(): 使用当前策略采样动作；
        2. value(): 估计状态价值；
        3. update(): 使用 buffer 中的数据执行 PPO 更新。
    """

    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.network = ActorCritic(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
        )

    @torch.no_grad()
    def act(self, state: np.ndarray):
        """
        与环境交互时调用。

        输入：
            state = 当前环境状态 s_t

        输出：
            action   = 从 π_old(a | s_t) 中采样得到的动作
            log_prob = log π_old(a_t | s_t)
            value    = V_old(s_t)

        由于该阶段只是采样数据，因此不需要保存计算图。
        """

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action, log_prob, _, value = self.network.get_action_and_value(
            state_tensor
        )

        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )

    @torch.no_grad()
    def value(self, state: np.ndarray) -> float:
        """
        仅预测某个状态的价值 V(s)。

        主要用于：
            rollout 结束但 episode 尚未结束时，
            估计最后一个状态之后的未来收益。
        """

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        value = self.network.get_value(state_tensor)

        return value.item()

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        使用一整个 rollout buffer 更新 Actor 与 Critic。

        更新步骤：
            1. 将 buffer 转成 tensor；
            2. 标准化 advantage；
            3. 多次 epoch 遍历这批数据；
            4. 对每个 mini-batch 计算：
                   - 新旧策略概率比 ratio
                   - clipped actor loss
                   - critic value loss
                   - entropy bonus
            5. 反向传播更新参数。
        """

        data = buffer.to_tensors(self.device)

        states = data["states"]
        actions = data["actions"]
        old_log_probs = data["old_log_probs"]
        old_values = data["old_values"]
        advantages = data["advantages"]
        returns = data["returns"]

        # ----------------------------------------------------
        # 标准化 Advantage
        # ----------------------------------------------------
        # PPO 中通常会标准化 advantage，以降低训练数值波动。
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

        num_samples = len(buffer)

        actor_losses = []
        critic_losses = []
        entropy_values = []
        approx_kls = []
        clip_fractions = []

        # ----------------------------------------------------
        # 同一批 rollout 数据更新多个 epoch
        # ----------------------------------------------------
        for epoch in range(self.config.update_epochs):

            sampler = BatchSampler(
                SubsetRandomSampler(range(num_samples)),
                batch_size=self.config.mini_batch_size,
                drop_last=False,
            )

            for batch_indices in sampler:

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]

                # ------------------------------------------------
                # 用当前正在更新的新策略，重新评价旧动作
                # ------------------------------------------------
                _, new_log_probs, entropy, new_values = (
                    self.network.get_action_and_value(
                        batch_states,
                        batch_actions,
                    )
                )

                # ------------------------------------------------
                # PPO 概率比值
                #
                # ratio =
                #     π_new(a_t | s_t) / π_old(a_t | s_t)
                #
                # 使用 log_prob 计算更稳定：
                # exp(log π_new - log π_old)
                # ------------------------------------------------
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                # ------------------------------------------------
                # PPO Actor Loss
                #
                # unclipped:
                #     ratio * advantage
                #
                # clipped:
                #     clip(ratio, 1-ε, 1+ε) * advantage
                #
                # 目标是最大化二者较小值。
                # 由于优化器执行最小化，因此前面添加负号。
                # ------------------------------------------------
                surrogate_1 = ratio * batch_advantages

                surrogate_2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_eps,
                    1.0 + self.config.clip_eps,
                ) * batch_advantages

                actor_loss = -torch.min(
                    surrogate_1,
                    surrogate_2,
                ).mean()

                # ------------------------------------------------
                # Critic Loss
                #
                # Critic 希望满足：
                #     V(s_t) ≈ Return_t
                #
                # Return_t = Advantage_t + V_old(s_t)
                # ------------------------------------------------
                critic_loss = 0.5 * (
                    new_values - batch_returns
                ).pow(2).mean()

                # ------------------------------------------------
                # Entropy Bonus
                #
                # entropy 越大，策略越不确定，探索越充分。
                # 因为总体目标是最小化 loss，
                # 所以写成 - entropy_coef * entropy。
                # ------------------------------------------------
                entropy_bonus = entropy.mean()

                total_loss = (
                    actor_loss
                    + self.config.value_coef * critic_loss
                    - self.config.entropy_coef * entropy_bonus
                )

                # ------------------------------------------------
                # 梯度更新
                # ------------------------------------------------
                self.optimizer.zero_grad()
                total_loss.backward()

                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm,
                )

                self.optimizer.step()

                # ------------------------------------------------
                # 可选：记录一些训练指标
                # ------------------------------------------------
                with torch.no_grad():

                    # 近似 KL，用于观察新旧策略是否偏离过大
                    approx_kl = (
                        (ratio - 1.0) - log_ratio
                    ).mean()

                    # 有多少比例的数据触发了 clipping
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_eps)
                        .float()
                        .mean()
                    )

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                entropy_values.append(entropy_bonus.item())
                approx_kls.append(approx_kl.item())
                clip_fractions.append(clip_fraction.item())

        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropy_values)),
            "approx_kl": float(np.mean(approx_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
        }

# ============================================================
# 5. Training Loop
# ============================================================

def train_ppo(env, config: PPOConfig, total_steps: int):
    """
    env 可以理解为类似 Gymnasium 环境：

        state, info = env.reset()

        next_state, reward, terminated, truncated, info = env.step(action)

    对强化学习而言：
        - Actor 只负责输出 action；
        - reward 由 env.step(action) 返回；
        - buffer 在拿到 reward 之后写入 transition。
    """

    agent = PPOAgent(config)
    buffer = RolloutBuffer()

    state, _ = env.reset()

    episode_return = 0.0
    episode_length = 0

    for global_step in range(1, total_steps + 1):

        # ----------------------------------------------------
        # 第一步：Actor 根据当前状态选择动作
        #
        # 得到：
        #     a_t
        #     log π_old(a_t | s_t)
        #     V_old(s_t)
        # ----------------------------------------------------
        action, old_log_prob, old_value = agent.act(state)

        # ----------------------------------------------------
        # 第二步：将动作交给环境执行
        #
        # 环境返回：
        #     s_{t+1}
        #     r_t
        #     是否终止
        #
        # 关键点：
        #     reward 不是 Actor 或 Critic 输出的，
        #     而是环境执行动作后产生的反馈。
        # ----------------------------------------------------
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # ----------------------------------------------------
        # 第三步：将当前 transition 存入 buffer
        #
        # 此时一个 transition 的数据已经完整：
        #
        #     s_t
        #     a_t
        #     r_t
        #     done_t
        #     V_old(s_t)
        #     log π_old(a_t | s_t)
        # ----------------------------------------------------
        buffer.add(
            state=state,
            action=action,
            reward=reward,
            done=done,
            value=old_value,
            log_prob=old_log_prob,
        )

        episode_return += reward
        episode_length += 1

        # 环境推进到下一状态
        state = next_state

        # ----------------------------------------------------
        # 第四步：如果 episode 结束，重置环境
        # ----------------------------------------------------
        if done:
            print(
                f"step={global_step:7d} | "
                f"episode_return={episode_return:8.2f} | "
                f"episode_length={episode_length:4d}"
            )

            state, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        # ----------------------------------------------------
        # 第五步：收集满一批 rollout 后，执行 PPO 更新
        # ----------------------------------------------------
        if len(buffer) >= config.rollout_steps:

            # ------------------------------------------------
            # rollout 最后一步的 bootstrap value
            #
            # 情况 A：最后一个 transition 到达 terminal
            #         后面没有收益，所以 last_value = 0。
            #
            # 情况 B：只是因为 rollout_steps 满了而暂停采样，
            #         当前 episode 仍在继续，
            #         则用 Critic 估计 V(s_{T+1})。
            # ------------------------------------------------
            if done:
                last_value = 0.0
            else:
                last_value = agent.value(state)

            # ------------------------------------------------
            # 第六步：根据已有 reward 与 value 计算：
            #         - TD Error
            #         - Advantage
            #         - Return
            # ------------------------------------------------
            buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )

            # ------------------------------------------------
            # 第七步：PPO 更新 Actor 与 Critic
            # ------------------------------------------------
            metrics = agent.update(buffer)

            print(
                f"[update] step={global_step:7d} | "
                f"actor_loss={metrics['actor_loss']:+.4f} | "
                f"critic_loss={metrics['critic_loss']:.4f} | "
                f"entropy={metrics['entropy']:.4f} | "
                f"approx_kl={metrics['approx_kl']:.6f} | "
                f"clip_frac={metrics['clip_fraction']:.3f}"
            )

            # ------------------------------------------------
            # 第八步：清空旧数据
            #
            # PPO 是 on-policy 算法。
            # 当前策略已经被更新，因此旧 rollout 不应长期重复使用。
            # ------------------------------------------------
            buffer.clear()

    return agent
```
