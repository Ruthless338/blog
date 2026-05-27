---
title: VQ-VAE与codebook
date: 2026-03-21
categories:
    - 科研
tags:
    - 生成式AI
    - 数字人
---


现在生成式 AI（特别是视觉和动作生成领域）有一个大趋势：**把连续的物理信号（如pixel、3D关节坐标）变成像“文字”一样的离散符号（Token）**，然后再用大语言模型（如 Transformer/GPT）来生成它们。

VQ-VAE 和 Codebook 就是完成这第一步“文字化（Tokenization）”的魔法。

<!--more-->
---

### 一、 什么是 Codebook 和 VQ-VAE？

#### 1. 形象比喻：“字典”
在一些生成任务中，数据是离散的，比如数字人生成，将离散特征直接转化为对应的3D顶点坐标其实是比较困难的。
*   **VQ-VAE （离散空间）**：编写了一本 **《动作指导手册》**，里面收录了 1024 (dim) 个标准的微动作（比如：第 15 号动作是“微笑着眨眼”，第 58 号动作是“张嘴大笑”）。这本手册，就是 **Codebook（码本）**。
    *   当你看到一段真实的动作时，你不再记录坐标，而是把它拆解，记下编号：“15, 58, 230, 11...”。这个将连贯动作变成一系列编号的过程，就是 **Vector Quantization（向量量化）**。

#### 2. VQ-VAE 的三大组件
VQ-VAE (Vector Quantized Variational Autoencoder) 是一个完整的神经网络架构，主要包含三部分：
1.  **编码器 (Encoder)**：把高维、复杂的原始数据（比如 3D 坐标），压缩成一串紧凑的“连续特征向量”。
2.  **量化器 (Quantizer) + 码本 (Codebook)**：**这是 VQ 独有的核心！** 模型拿着 Encoder 算出来的连续向量，去 Codebook（字典）里找**距离最接近**的那一个标准向量。然后，**用这个标准向量替换掉原来的向量**。此时，数据就变成了 Codebook 里的索引号（ID / Token）。
3.  **解码器 (Decoder)**：拿着这串标准向量，努力还原出最初的原始 3D 舞蹈动作。

**为什么必须用 VQ-VAE？为什么不直接生成坐标？**
如果让 Transformer 直接生成 3D 连续坐标（小数），模型为了降低误差，往往会预测一个“平均值”，导致生成的动作**极其平滑、缺乏爆发力、软绵绵的（称为 Over-smoothing）**。而 VQ-VAE 把动作变成了像词汇一样的“离散选项”（要么是动作A，要么是动作B），Transformer 只需要做“多项选择题”，这大大降低了生成难度，动作也更干脆、真实。

---

简单的示例代码实现如下：
```python
class VQVAE(torch.nn.Module):
    def __init__(self,codebook_size):
        super().__init__()
        self.encoder=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=4,stride=2,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4,stride=2,padding=1),
            torch.nn.ReLU()
        ) # (B,C=32,H=7,W=7)
        self.codebook=torch.nn.Parameter(torch.randn(codebook_size,32)) # (CODEBOOK_SIZE,32)
        self.decoder=torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16,1,kernel_size=4,stride=2,padding=1),
            torch.nn.Sigmoid()
        ) # (B,C=1,H=28,W=28)
    
    def encode(self,x):
        # 图像压缩
        ze=self.encoder(x) # ze=(B,C=32,H=7,W=7)
        # VQ-VAE量化
        ze_extended=ze.unsqueeze(1) # (B,1,C=32,H=7,W=7)
        codebook_extended=self.codebook.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # (1,CODEBOOK_SIZE,C=32,H=1,W=1)
        dist=(ze_extended-codebook_extended)**2
        dist=dist.sum(dim=2)
        code_idx=dist.argmin(1) # 取最邻近codebook下标, shape=(B,H=7,W=7) 
        return ze,code_idx
    
    def forward(self,x): # x: (B,C=1,H=28,W=28)
        # 图像压缩&离散编码
        ze,code_idx=self.encode(x)
        # 离线编码转稠密码本向量
        zq=self.codebook[code_idx] # 取codebook的embedding, shape=(B,H=7,W=7,C=32)
        zq=zq.permute(0,3,1,2) # zq=(B,C=32,H=7,W=7)
        # 图像解压
        x_recon=self.decoder(ze+(zq-ze).detach()) # x_recon=(B,C=1,H=28,W=28)
        return x_recon,ze,zq
```

### 二、 DuetGen: Music Driven Two-Person Dance Generation via Hierarchical Masked Modeling

在双人舞任务中，如果直接套用普通的 VQ-VAE 会遇到大麻烦：双人舞既有**宏观的全局移动**（比如两人绕场走位），又有**极其微小的局部互动**（比如两人指尖相触）。一个普通的 Codebook 容量有限，很难同时记住“大动作”和“小细节”。

因此，DuetGen 提出了 **Hierarchical VQ-VAE（分层双人动作量化模型）**。

#### 1. DuetGen 的“两本字典”（分层 Codebook）
为了解决大小动作的冲突，作者设计了两层 Codebook，就像是给了 Transformer 一套“主谓宾”的大纲，和一套“形容词”的细节：

*   **顶层字典（Top-level Codebook, $C_{top}$）**：
    *   **作用**：负责捕捉**全局动作语义（大动作）**，比如“两人正在携手向前走”、“两人正在旋转”。
    *   **特点**：时间分辨率低（即经过了多次降采样，可能是每 8 帧才提取一个 Token）。它的视野很宏观。
*   **底层字典（Bottom-level Codebook, $C_{bot}$）**：
    *   **作用**：负责捕捉补充的**细粒度细节（小动作）**，比如在旋转时，“两人脚踝的具体弯曲角度”或“两人手臂的具体相对位置”。
    *   **特点**：时间分辨率高（降采样较少，比如每 4 帧提取一个 Token）。

#### 2. DuetGen 训练 VQ-VAE 的完整流程（对应论文 Fig. 2 左侧）
1.  **输入（Unified Representation）**：模型拿到一段双人舞，每一帧是一个 536 维的大向量（包含了 A 和 B 的关节旋转、相对位置等）。
2.  **编码压缩**：经过底层编码器 $E_B$ 和顶层编码器 $E_T$，提取出特征。
3.  **顶层量化**：特征在**顶层字典 ($C_{top}$)** 里找到最接近的编号，变成顶层 Tokens（例如 `[12, 45, 99...]`）。
4.  **底层量化**：结合顶层的特征，在**底层字典 ($C_{bot}$)** 里找到最接近的编号，变成底层 Tokens（例如 `[512, 23, 1024, 88...]`）。
5.  **解码重建**：解码器 $D$ 拿到这两种 Tokens，把它们拼在一起，**并且（重点来了）还输入了当前的音乐特征**，最终努力还原出那 536 维的 3D 舞蹈坐标。

#### 4. VQ-VAE 训练好之后用来干嘛？
一旦这个分层的 VQ-VAE 训练完毕，**它的编码器和解码器就被冻结（参数固定）了**。
此时，复杂的 3D 双人舞数据，在模型眼里就彻底变成了两行简单的整数序列（Tokens）。

接下来的任务就变得极其简单、符合当今大模型的玩法了：
**这就变成了一个 NLP（自然语言处理）任务！**
论文 Method 3.3 中的 Transformer 就像是一个填词游戏的高手，它听着音乐，预测出顶层的整数（Token），然后再根据音乐和顶层整数，预测出底层的整数。最后把这些整数扔给冻结的 VQ-VAE 解码器，完美的双人 3D 舞蹈就诞生了。

这种多层codebook去建模coarse与fine的method也是一种主流方案，如
ARTalk: Speech-Driven 3D Head Animation via Autoregressive Model中甚至采用了4层以上的codebook