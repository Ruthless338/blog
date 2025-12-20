Stable Diffusion + Text Inversion完整工作流程：
1. Text Inversion(TI)训练: 有监督训练，text_encoder一般是Transformer，输入为prompt+示例图，输出为文本嵌入向量，再经过扩散模型（预训练冻结的SD）生成图，对比示例图完成训练
2. 逆扩散：生成一个纯随机高斯噪声，选择一个prompt经过TI生成文本向量，经过t步去噪，每步去噪输入为上一步潜噪声+文本向量+步数t，经过U-Net得到大小不变的预测噪声，最后经过scheduler得到新潜变量
3. VAE解码：逆扩散得到的潜变量经过预训练的VAE decoder得到最终的生成图

Flow Matching：
不同于Diffusion Model，FM步数少，每一步使用UNet去预测速度场


1. MRGen: Segmentation Data Engine for Underrepresented MRI Modalities
Diffusion Model+text-guided + mask-conditioned生成，主要用于分割

2. Latent Drifting in Diffusion Models for Counterfactual Medical Image Synthesis 2025CVPR
使用Latent Drifting(LD)为预训练的Latent Diffusion Model(LDM)的反向扩散去噪阶段的均值增加校准，实现Counterfactual Medical Image Synthesis

3. Enhance Image Classification via Inter-Class Image Mixup with Diffusion Model  2024CVPR
Diff-Mix: 使用Textual Inversion(TI) + DB（用于微调U-Net）微调Stable Diffusion（SD），之后对采样图像（全训练集的跨类图像）进行前向加噪，再反向去噪（使用目标类TI）完成图像生成（跨类翻译） -> faithfulness and diversity

4. BEYOND OBJECTS: CONTEXTUAL SYNTHETIC DATA GENERATION FOR FINE-GRAINED CLASSIFICATION
BOB: 微调T2I构建真实图像的“background-pose”，再完全依赖“class-background-pose”生成图像

5. Inversion Circle Interpolation: Diffusion-based Image Augmentation for Data-scarce Classification 2025CVPR
Diff-II: category learning（微调U-Net） + 随机采样（两张同类别）DDIM反转>Inversion Circle interpolatoin + 提示词coarse-fine去噪 -> faithfulness and diversity
DDIM反转：从干净图像反推高噪声潜向量，不同于DM加噪的正向加随机噪声破坏图像

6. DreamDA: Generative Data Augmentation with Diffusion Models
DreamDA: 在SD+TI基础上，在逆扩散的每一步，给 U-Net 的瓶颈层特征加高斯噪声

7. Generating Images of Rare Concepts Using Pre-trained Diffusion Models  2024AAAI
针对rare concept和类级概念生成，TI聚焦示例级生成，而论文提出的SeedSelect（随机噪声作为参数，反向传播确定固定文本向量的有效噪声区域）效果更好，由于论文要求对SD无微调，所以不使用TI

8. EFFECTIVE DATA AUGMENTATION WITH DIFFUSION MODELS ICLR 2024
早期的TI+SD生成

9. Synthetic Data Augmentation using Pre-trained Diffusion Models for Long-tailed Food Image Classification 2025CVPR
在TI+SD的基础上，通过CADS动态退火为正提示文本向量加噪，通过CCFG+正负prompt融合噪声（拉进同类、推远异类）

10. AUGMENTED CONDITIONING IS ENOUGH FOR EFFECTIVE TRAINING IMAGE GENERATION
采样为两张同类图像，通过CutMix + Mixup +Dropout方法增强图像，不使用TI微调SD，针对长尾/少样本数据集

11. DALDA: Data Augmentation Leveraging Diffusion Model and LLM with Adaptive Guidance Scaling
使用LLM为每张图生成多样的prompt，同类采样prompt和示例图，再经过CLIP->IP-Adapter(解耦交叉注意力，融合特征)->Diffusion Model->VAE decoder，其中所有模型都是冻结的

12. Interpolating between Images with Diffusion Models
采样为固定的两帧，主要目的是生成平滑过渡而非faithfulness+diversity，两帧示例图加噪之后球面线性插值+TI(Text Embedding)+Pose Guidance 输入到 DM

13. Flow Matching for Medical Image Synthesis: Bridging the Gap Between Speed and Quality  MICCAI2025
使用Flash Attention优化医学图像，主要模型为FM，输入为纯高斯噪声+独热编码+mask（可选），输出为生成图

14. Contrastive Flow Matching
在Flow Matching中加入对比损失，即随机采样batch中的其他流做loss

15. FlowAR: Scale-wise Autoregressive Image Generation Meets Flow Matching
FlowAR: 使用VAE构造多尺度latent -> 自回归Transformer生成多尺度语义条件 -> Spatial-adaLN融合语义条件和latent，输入FM，最后通过VAE输出生成图像

16. Metric Flow Matching for Smooth Interpolations on the Data Manifold NeurlPS2024
MFM: 插值函数从FM的线性插值改为非线性插值，目的是构建符合数据流形的概率路径，开销高
