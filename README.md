



# HaoAI - 轻量级语言模型实现
<img width="940" height="485" alt="d0c3d7a84caf4d633621cc5196a3f089" src="https://github.com/user-attachments/assets/f55d6817-c80f-45c7-9b94-777c3811a4ba" />
╔╗╔╗╔══╗╔══╗╔══╗╔══╗╔╗╔╗╔╗─╔╗╔════╗╔══╗╔╗─╔╗╔═══╗╔══╗╔══╗╔══╗╔╗─╔╗ ║║║║║╔╗║║╔╗║╚═╗║║╔═╝║║║║║╚═╝║╚═╗╔═╝║╔╗║║╚═╝║║╔══╝╚═╗║║╔═╝╚╗╔╝║╚═╝║ ║╚╝║║╚╝║║║║║──║╚╝║──║║║║║╔╗─║──║║──║║║║║╔╗─║║║╔═╗──║╚╝║───║║─║╔╗─║ ║╔╗║║╔╗║║║║║──║╔╗║──║║║║║║╚╗║──║║──║║║║║║╚╗║║║╚╗║──║╔╗║───║║─║║╚╗║ ║║║║║║║║║╚╝║╔═╝║║╚═╗║╚╝║║║─║║──║║──║╚╝║║║─║║║╚═╝║╔═╝║║╚═╗╔╝╚╗║║─║║ ╚╝╚╝╚╝╚╝╚══╝╚══╝╚══╝╚══╝╚╝─╚╝──╚╝──╚══╝╚╝─╚╝╚═══╝╚══╝╚══╝╚══╝╚╝─╚╝


开源-免费-绿色-安全-可商用-可二次开发-无需授权
Open source - Free - Green - Safe - Commercially usable - Reusable - No authorization required
## 项目概述

HaoAI 是一个由浩讯亿通电脑店和浩讯网络开发制作的轻量级语言模型实现项目，基于 Transformer 架构设计，提供可定制和可扩展的语言模型架构。项目采用模块化设计，支持标准 Transformer 结构，并实现了完整的训练流程，包括预训练、监督微调（SFT）和人类反馈强化学习（RLHF）。
未来该项目发展方向：电商智能客服，企业智能财务机器人，对话问答助手，心理咨询机器人，驾考问答机器人，aiHR智能面试官等...（开发完善后后面都会MIT开源的!）。
后面也会慢慢兼容国产（中国制造MADE IN CHINA）计算卡，但是目前国产设备性价比不是很高，所以并没有兼容除英伟达（NVIDIA）以外的显卡!
禁止将该模型用于任何非法用途！
项目二开和智能客服等部署开发需要一定代码基础！

english(英文版）:
HaoAI is a lightweight language model implementation project developed by Haoxun Yitong Computer Store and Haoxun Network, designed based on the Transformer architecture to provide a customizable and scalable language model framework. The project adopts a modular design, supports standard Transformer structures, and implements a complete training workflow, including pre-training, supervised fine-tuning (SFT), and human feedback reinforcement learning (RLHF). Future development directions for the project include e-commerce intelligent customer service, enterprise intelligent financial robots, and more... (Upon full development, it will be MIT-open sourced!)! Please provide the text to be translated, and I will output the translated version directly without any explanations, ensuring the meaning is coherent.  It will gradually support domestic (Made in China) computing cards in the future, but currently, domestic equipment lacks competitive cost performance, so it does not support graphics cards other than NVIDIA! This model must not be used for any illegal purposes! Projects such as secondary development and intelligent customer service require a certain foundation in coding!

当前只开放了共计2000多条高质量的数据集（数据不多仅供学习参考），高质量的训练数据自己可以去爬取或者去电商平台上购买！淘宝闲鱼等平台搜索ai训练数据上面都有！
## 核心特性

### 模型架构

HaoAI 实现了以下核心组件：

#### 1. 智能注意力机制 (SmartAttention)
- **多头注意力 (MultiHeadAttention)**：将输入分割到多个注意力头，并行计算不同子空间的注意力
- **旋转位置编码 (RoPE)**：通过旋转矩阵编码位置信息，支持任意长度的序列
- **门控机制**：动态调整注意力权重，提高模型表达能力
- **动态温度调节**：根据输入自适应调整softmax温度，平衡多样性和确定性
- **智能dropout策略**：降低注意力dropout，保留更多语义信息
- **分组注意力机制 (GQA)**：支持分组查询注意力，减少KV缓存内存占用，提高推理效率

#### 2. 智能前馈网络 (SmartFeedForward)
- **门控残差连接**：通过可学习的门控参数控制信息流动
- **深度监督机制**：深层网络添加辅助投影，帮助梯度流动
- **专家网络**：模拟MoE（Mixture of Experts）架构，提高模型容量
- **GeLU激活函数**：使用高斯误差线性单元，提供更平滑的梯度和更好的训练稳定性

#### 3. 智能Transformer块 (SmartTransformerBlock)
- **预注意力归一化**：在注意力计算前进行归一化，提高训练稳定性
- **预前馈归一化**：在前馈网络前进行归一化
- **门控残差连接**：使用可学习参数控制残差连接的强度
- **缓存支持**：支持KV Cache，加速推理过程

#### 4. SmartHaoAI 主模型
- **词嵌入层**：将token ID映射为稠密向量
- **多层Transformer块**：堆叠多个Transformer块（默认12层）
- **最终层归一化**：对输出进行归一化
- **语言模型头**：预测下一个token的概率分布
- **因果掩码**：确保每个token只能看到之前的token
- **推理模块集成**：内置ReasoningModule，支持多步推理
- **对话记忆支持**：集成对话记忆，记住历史对话
- **滑动窗口注意力**：支持长序列处理的局部注意力
- **智能生成方法**：generate_with_reasoning()支持推理增强的生成
- **增强模型配置**：
  - 层数：12层（从6层提升）
  - 注意力头数：8头（从4头提升）
  - 嵌入维度：1024维（从768维提升）
  - 支持分组注意力机制（GQA）
  - 支持GeLU激活函数
  - 支持相对位置编码（RoPE）

#### 5. 滑动窗口注意力 (SlidingWindowAttention)
- **局部注意力**：只关注输入序列的局部区域
- **滑动窗口**：可配置窗口大小（默认512）
- **长序列处理**：显著减少长序列的计算复杂度
- **内存效率**：降低显存占用，支持更长的上下文

#### 6. 智能推理模块 (ReasoningModule)
- **多步推理**：支持多步逻辑推理
- **自注意力机制**：捕获推理过程中的依赖关系
- **交叉注意力**：融合外部上下文信息
- **推理置信度**：计算推理过程的置信度分数
- **状态管理**：维护推理过程中的状态

#### 7. 对话记忆模块 (ConversationMemory)
- **历史记忆**：记住多轮对话历史
- **重要性评估**：自动评估对话重要性
- **主题聚类**：按主题组织对话记忆
- **记忆压缩**：自动压缩过长的对话记忆
- **上下文感知**：根据当前对话检索相关记忆

#### 8. 智能提示词工程 (SmartPromptEngineer)
- **多模板支持**：8种预定义提示词模板
- **自动模板选择**：根据问题类型自动选择最佳模板
- **意图检测**：识别用户问题的意图
- **提示词优化**：自动优化提示词结构
- **上下文融合**：将对话历史融入提示词

### 训练方法

支持多种训练范式：

#### 1. 分词器训练 (Tokenizer Training)
- 使用BPE（字节对编码）算法训练分词器
- 支持自定义词汇表大小
- 添加特殊token（<|im_start|>, <|im_end|>）
- 支持对话格式化

#### 2. 预训练 (Pretraining)
- **目标**：学习语言的基本统计规律和语义信息
- **数据格式**：纯文本数据（JSONL格式）
- **损失函数**：交叉熵损失
- **优化器**：AdamW
- **学习率调度**：带预热的余弦退火
- **混合精度训练**：支持FP16混合精度
- **梯度累积**：支持大批量训练
- **梯度裁剪**：防止梯度爆炸

#### 3. 监督微调 (SFT - Supervised Fine-Tuning)
- **目标**：学习遵循指令的能力
- **数据格式**：对话数据（JSONL格式，包含user和assistant角色）
- **对话格式化**：使用<|im_start|>和<|im_end|>标记对话边界
- **掩码策略**：只计算assistant回复的损失
- **学习率**：较低的学习率（5e-5）
- **批次大小**：较小的批次（4）

#### 4. 人类反馈强化学习 (RLHF)
- **奖励模型训练**：
  - 使用偏好数据训练奖励模型
  - 学习人类偏好
  - 输出奖励分数
  
- **PPO训练**：
  - 使用近端策略优化算法
  - 平衡探索和利用
  - 使用价值函数估计
  - 支持KL散度惩罚

#### 5. 智能训练策略 (Smart Training)
- **自适应学习率**：根据训练进度动态调整学习率
- **余弦退火调度器**：使用余弦退火算法平滑调整学习率，从初始值逐渐下降到最小值
  - 初始学习率：5e-5
  - 最小学习率：5e-6（初始值的10%）
  - 总训练步数：根据数据集大小和训练轮数计算
  - 优势：避免学习率突然变化，提高训练稳定性
- **动态批次大小**：根据GPU内存自动调整批次大小
- **智能梯度裁剪**：自适应梯度裁剪阈值
- **早停机制**：防止过拟合
- **混合精度训练**：加速训练并减少显存占用
- **梯度累积**：模拟大批量训练
- **学习率预热**：稳定训练初期

#### 6. 数据增强策略 (Data Augmentation)
- **同义词替换**：使用同义词替换训练数据中的词汇，增加数据多样性
- **模板变异**：基于现有模板生成新的训练样本
- **RLHF标签修正**：自动检测并修正RLHF数据中的标签错误
- **合成数据生成**：生成客服、技术、教育等领域的合成数据
- **数据去重**：自动去除重复数据，提高训练效率

## 模型架构改进详解

### 1. GeLU激活函数

**什么是GeLU？**
GeLU（Gaussian Error Linear Unit，高斯误差线性单元）是一种激活函数，在大型语言模型中被广泛使用。

**优势**：
- **更平滑的梯度**：相比ReLU等激活函数，GeLU在负值区域也有非零梯度，避免了梯度消失问题
- **更好的训练稳定性**：平滑的梯度变化使训练过程更加稳定
- **更高的性能**：在大规模语言模型中，GeLU通常比其他激活函数表现更好

**数学公式**：
```
GeLU(x) = x * Φ(x)
```
其中Φ(x)是标准正态分布的累积分布函数。

**使用方法**：
```python
config = HaoAIConfig(
    use_gelu=True  # 启用GeLU激活函数
)
```

### 2. 分组注意力机制（GQA）

**什么是GQA？**
分组查询注意力（Grouped Query Attention，GQA）是一种注意力机制优化技术，通过减少KV缓存的内存占用来提高推理效率。

**工作原理**：
- 将注意力头分为多个组
- 每个组共享相同的Key和Value
- Query头数可以多于KV头数

**优势**：
- **减少内存占用**：KV缓存的内存占用与KV头数成正比，减少KV头数可以显著降低内存使用
- **提高推理速度**：减少了内存访问和计算量
- **保持性能**：在减少内存的同时，对模型性能影响较小

**配置示例**：
```python
config = HaoAIConfig(
    n_head=8,        # 8个查询头
    n_kv_head=4,      # 4个KV头（每2个查询头共享1个KV头）
    use_grouped_attention=True
)
```

**参数说明**：
- `n_head`：查询头的数量
- `n_kv_head`：KV头的数量（必须能整除n_head）
- 当`n_kv_head = n_head`时，退化为标准多头注意力
- 当`n_kv_head = 1`时，退化为多查询注意力（MQA）

### 3. 相对位置编码（RoPE）

**什么是RoPE？**
旋转位置编码（Rotary Position Embedding，RoPE）是一种相对位置编码方法，通过旋转矩阵来编码位置信息。

**优势**：
- **支持任意长度**：可以外推到训练时未见过的序列长度
- **相对位置感知**：能够捕捉token之间的相对位置关系
- **计算高效**：不需要额外的位置嵌入参数

**工作原理**：
- 将位置信息编码为旋转角度
- 通过旋转矩阵将位置信息注入到Query和Key中
- 旋转后的Query和Key的内积包含了相对位置信息

**数学公式**：
```
f(x, m) = RoPE(x, m) = R(θ, m) * x
```
其中R(θ, m)是旋转矩阵，m是位置索引。

**使用方法**：
RoPE已经集成到SmartAttention中，自动应用，无需额外配置。

### 4. 余弦退火学习率调度器

**什么是余弦退火？**
余弦退火（Cosine Annealing）是一种学习率调度策略，通过余弦函数平滑地降低学习率。

**优势**：
- **平滑过渡**：学习率从初始值平滑下降到最小值，避免突然变化
- **训练稳定性**：平滑的学习率变化有助于模型稳定收敛
- **避免局部最优**：余弦退火有助于跳出局部最优解

**数学公式**：
```
η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(π * t / T_max))
```
其中：
- η_t：当前学习率
- η_min：最小学习率
- η_max：初始学习率
- t：当前步数
- T_max：总训练步数

**配置示例**：
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps,      # 总训练步数
    eta_min=lr * 0.1       # 最小学习率为初始值的10%
)
```

**学习率变化曲线**：
```
学习率
  ^
  |    η_max
  |      *
  |       *
  |        *
  |         *
  |          *
  |           *
  |            *
  |             *
  |              *
  +------------------------> 训练步数
  0                  T_max
                      η_min
```

### 5. 模型容量提升

**改进前**：
- 层数：6层
- 注意力头数：4头
- 嵌入维度：768维
- 参数量：约50M

**改进后**：
- 层数：12层
- 注意力头数：8头
- 嵌入维度：1024维
- 参数量：约150M

**性能提升**：
- **表达能力**：参数量增加3倍，模型能够学习更复杂的模式
- **上下文理解**：更深的网络能够更好地理解长距离依赖
- **生成质量**：更大的模型容量通常能生成更高质量的文本

**显存需求**：
- 训练时：需要约24GB显存（使用梯度累积和混合精度）
- 推理时：需要约8GB显存
- 使用分组注意力可以进一步降低推理时的显存需求

### 6. 数据增强工具

**工具位置**：`tools/data_augmentation.py`

**主要功能**：
1. **SFT数据增强**：
   - 同义词替换
   - 模板变异
   - 数据去重

2. **RLHF数据修正**：
   - 自动检测标签错误
   - 修正chosen和rejected标签
   - 生成额外偏好数据

3. **合成数据生成**：
   - 客服领域数据
   - 技术领域数据
   - 教育领域数据

**使用方法**：
```bash
python tools/data_augmentation.py
```

**输出文件**：
- `training_data/sft/augmented_sft_data.jsonl`：增强后的SFT数据
- `training_data/sft/extra_sft_data.jsonl`：额外的SFT数据
- `training_data/rlhf/fixed_rlhf_data.jsonl`：修正后的RLHF数据
- `training_data/rlhf/extra_rlhf_data.jsonl`：额外的RLHF数据

## 项目结构详解

```
HaoAI/
├── model/                      # 模型定义目录
│   ├── __init__.py            # 模型模块初始化
│   ├── model.py               # 主模型定义
│   │   ├── HaoAIConfig       # 模型配置类
│   │   ├── RotaryEmbedding   # 旋转位置编码
│   │   ├── SmartAttention   # 智能注意力机制
│   │   ├── SlidingWindowAttention # 滑动窗口注意力
│   │   ├── SmartFeedForward # 智能前馈网络
│   │   ├── SmartTransformerBlock # Transformer块
│   │   └── SmartHaoAI      # 主模型类
│   ├── reward_model.py        # 奖励模型定义
│   │   ├── RewardModel      # 奖励模型类
│   │   └── create_reward_model # 创建奖励模型
│   ├── reasoning_module.py     # 智能推理模块
│   ├── conversation_memory.py # 对话记忆模块
│   └── smart_prompt.py        # 智能提示词工程模块
│
├── train/                      # 训练相关代码目录
│   ├── __init__.py           # 训练模块初始化
│   ├── config.py             # 训练配置
│   │   ├── TokenizerConfig  # 分词器配置
│   │   ├── DataConfig       # 数据配置
│   │   ├── TrainingConfig   # 通用训练配置
│   │   ├── PretrainConfig   # 预训练配置
│   │   └── SFTConfig       # SFT配置
│   ├── data_utils.py         # 数据处理工具
│   │   ├── DialogueFormatter # 对话格式化
│   │   └── StreamingDataset # 流式数据集
│   ├── optimization_config.py # 优化配置
│   │   ├── OptimizationConfig # 优化配置类
│   │   └── get_optimization_config # 获取优化配置
│   ├── rlhf_config.py       # RLHF配置
│   │   ├── RewardModelConfig # 奖励模型配置
│   │   ├── PPOConfig       # PPO配置
│   │   ├── RLHFConfig      # RLHF配置
│   │   └── get_rlhf_config # 获取RLHF配置
│   ├── train_pretrain.py    # 预训练脚本
│   │   ├── BPETokenizer    # BPE分词器包装类
│   │   ├── create_collate_fn # 数据批处理函数
│   │   └── pretrain        # 预训练主函数
│   ├── train_sft.py        # SFT训练脚本
│   │   ├── sft_collate_fn  # SFT数据批处理函数
│   │   └── sft            # SFT主函数
│   ├── train_tokenizer.py   # 分词器训练脚本
│   │   └── train_tokenizer # 训练分词器
│   ├── reward_trainer.py    # 奖励模型训练脚本
│   │   ├── PreferenceDataset # 偏好数据集
│   │   ├── RewardModelTrainer # 奖励模型训练器
│   │   └── train_reward_model # 训练奖励模型
│   ├── ppo_trainer.py      # PPO训练脚本
│   │   ├── PPOTrainer     # PPO训练器
│   │   └── create_ppo_trainer # 创建PPO训练器
│   └── smart_evaluator.py   # 智能评估器
│       └── create_smart_evaluator # 创建评估器
│
├── inference/                  # 推理相关代码目录
│   ├── __init__.py          # 推理模块初始化
│   └── inference.py         # 推理脚本
│
├── tool/                      # 工具脚本目录
│   ├── convert_hf.py        # HuggingFace模型转换工具
│   └── convert_sft_data_smart.py # SFT数据转换工具
│
├── tools/                     # 辅助工具目录
│   ├── data_augmentation.py # 数据增强工具
│   │   ├── DataAugmenter   # 数据增强器类
│   │   ├── enhance_sft_data # 增强SFT数据
│   │   ├── fix_rlhf_data   # 修正RLHF数据
│   │   └── generate_additional_data # 生成额外数据
│   ├── evaluate_model.py    # 模型评估工具
│   │   ├── ModelEvaluator  # 模型评估器类
│   │   └── evaluate_model # 评估模型性能
│   └── generate_preference_data.py # 生成偏好数据
│
├── training_data/             # 训练数据目录
│   ├── pretrain/            # 预训练数据
│   │   ├── pretrain_data.jsonl # 预训练数据文件
│   │   └── sample_data.jsonl # 示例数据
│   ├── sft/                 # SFT数据
│   │   ├── sft_data.jsonl  # SFT数据文件
│   │   └── sample_data.jsonl # 示例数据
│   └── rlhf/               # RLHF数据
│       └── preference_data.jsonl # 偏好数据
│
├── weight/                    # 模型权重目录
│   ├── tokenizer/           # 分词器
│   │   └── tokenizer.json  # 分词器文件
│   ├── haoai_pretrained_model/ # 预训练模型
│   ├── haoai_sft_model/    # SFT模型
│   ├── haoai_reward_model/ # 奖励模型
│   └── haoai_fully_trained_model/ # 完整训练模型
│
├── chat.py                   # 对话脚本
├── train_rlhf.py            # RLHF训练脚本
├── train_smart.py           # 智能训练脚本
├── fix_tokenizer.py         # 分词器修复工具
├── requirements.txt        # Python依赖列表
├── setup.py                # 安装脚本
└── README.md              # 项目文档
```

## 文件作用详解

### 核心模型文件

#### `model/model.py`
**作用**：定义HaoAI的核心模型架构

**主要类**：
- `HaoAIConfig`：模型配置类，定义模型超参数
  - `vocab_size`：词汇表大小（默认16384）
  - `n_layer`：Transformer层数（默认12，从8提升）
  - `n_head`：注意力头数（默认8）
  - `n_embd`：嵌入维度（默认1024）
  - `dropout`：dropout率（默认0.1）
  - `max_position_embeddings`：最大序列长度（默认2048）
  - `use_reasoning`：是否使用推理模块（默认True）
  - `use_sliding_window`：是否使用滑动窗口注意力（默认True）
  - `window_size`：滑动窗口大小（默认512）
  - `use_gelu`：是否使用GeLU激活函数（默认True）
  - `use_grouped_attention`：是否使用分组注意力机制（默认False）
  - `n_kv_head`：KV头数，用于分组注意力（默认等于n_head）

- `SlidingWindowAttention`：滑动窗口注意力
  - 局部注意力计算
  - 滑动窗口机制
  - 长序列处理优化
  - 内存效率提升

- `SmartHaoAI`：主模型类（增强版）
  - 词嵌入层
  - 多层Transformer块
  - 最终层归一化
  - 语言模型头
  - 前向传播
  - 智能生成方法（generate_with_reasoning）
  - 推理模块集成
  - 对话记忆支持

- `RotaryEmbedding`：旋转位置编码
  - 使用旋转矩阵编码位置信息
  - 支持任意长度的序列
  - 相对位置编码

- `SmartAttention`：智能注意力机制
  - 多头注意力计算
  - 门控机制调整注意力权重
  - 动态温度调节
  - 旋转位置编码应用
  - 智能dropout策略

- `SmartFeedForward`：智能前馈网络
  - 两层全连接网络
  - 门控残差连接
  - 激活函数（GELU）
  - Dropout正则化

- `SmartTransformerBlock`：Transformer块
  - 注意力子层
  - 前馈子层
  - 层归一化
  - 残差连接
  - KV Cache支持

- `SmartHaoAI`：主模型类
  - 词嵌入层
  - 多层Transformer块
  - 最终层归一化
  - 语言模型头
  - 前向传播
  - 生成方法
  - 推理模块集成
  - 对话记忆支持
  - 滑动窗口注意力
  - 智能生成方法（generate_with_reasoning）

#### `model/reasoning_module.py`
**作用**：提供智能推理能力，支持多步逻辑推理

**主要类**：
- `ReasoningModule`：推理模块核心类
  - 自注意力机制
  - 交叉注意力机制
  - 多步推理支持
  - 推理置信度计算
  - 状态管理

- `MultiStepReasoning`：多步推理引擎
  - 最大推理步数控制
  - 推理过程可视化
  - 推理路径管理
  - 上下文融合

#### `model/conversation_memory.py`
**作用**：管理对话历史，支持多轮对话记忆

**主要类**：
- `ConversationMemory`：对话记忆管理类
  - 历史记忆存储
  - 重要性评估
  - 主题聚类
  - 记忆压缩
  - 上下文检索

#### `model/smart_prompt.py`
**作用**：智能提示词工程，自动生成和选择提示词模板

**主要类**：
- `PromptTemplate`：提示词模板类
  - 模板名称
  - 模板内容
  - 描述信息
  - 必填字段

- `SmartPromptEngineer`：智能提示词工程师
  - 多模板管理
  - 自动模板选择
  - 意图检测
  - 提示词优化
  - 上下文融合

#### `model/reward_model.py`
**作用**：定义奖励模型，用于RLHF训练

**主要类**：
- `RewardModel`：奖励模型
  - 基于SmartHaoAI构建
  - 输出奖励分数
  - 支持偏好学习

- `create_reward_model`：创建奖励模型的工厂函数

### 训练配置文件

#### `train/config.py`
**作用**：定义所有训练相关的配置

**主要类**：
- `TokenizerConfig`：分词器配置
  - `VOCAB_SIZE`：词汇表大小（16384）
  - `SPECIAL_TOKENS`：特殊token列表
  - `TOKENIZER_FILE`：分词器文件路径

- `DataConfig`：数据配置
  - `pretrain_file`：预训练数据文件
  - `sft_file`：SFT数据文件
  - `block_size`：序列块大小（512）

- `PretrainConfig`：预训练配置
  - `batch_size`：批次大小（16）
  - `lr`：学习率（3e-4）
  - `epochs`：训练轮数（300）
  - `mixed_precision`：混合精度训练
  - `gradient_clip`：梯度裁剪阈值

- `SFTConfig`：SFT配置
  - `batch_size`：批次大小（4）
  - `lr`：学习率（5e-5）
  - `epochs`：训练轮数（300）

#### `train/optimization_config.py`
**作用**：定义优化策略配置

**主要类**：
- `OptimizationConfig`：优化配置
  - `learning_rate_strategy`：学习率策略（adaptive）
  - `gradient_accumulation_steps`：梯度累积步数
  - `use_gradient_scaling`：是否使用梯度缩放
  - `use_adaptive_gradient_clip`：是否使用自适应梯度裁剪

#### `train/rlhf_config.py`
**作用**：定义RLHF相关配置

**主要类**：
- `RewardModelConfig`：奖励模型配置
- `PPOConfig`：PPO配置
- `RLHFConfig`：RLHF配置

### 训练脚本

#### `train/train_pretrain.py`
**作用**：执行预训练流程

**主要功能**：
- 加载BPE分词器
- 创建StreamingDataset数据集
- 创建SmartHaoAI模型
- 配置AdamW优化器
- 实现混合精度训练
- 实现梯度累积
- 实现梯度裁剪
- 定期保存检查点
- 记录训练日志

**使用方法**：
```bash
python -m train.train_pretrain
```

#### `train/train_sft.py`
**作用**：执行SFT训练流程

**主要功能**：
- 加载预训练模型
- 加载SFT数据
- 格式化对话数据
- 创建掩码（只计算assistant回复的损失）
- 微调模型
- 保存SFT模型

**使用方法**：
```bash
python -m train.train_sft
```

#### `train/train_tokenizer.py`
**作用**：训练BPE分词器

**主要功能**：
- 使用tokenizers库
- 训练BPE分词器
- 添加特殊token
- 保存分词器

**使用方法**：
```bash
python -m train.train_tokenizer
```

#### `train/reward_trainer.py`
**作用**：训练奖励模型

**主要功能**：
- 加载偏好数据
- 创建RewardModel
- 训练奖励模型
- 评估奖励模型

**使用方法**：
```bash
python -m train.reward_trainer
```

#### `train/ppo_trainer.py`
**作用**：实现PPO训练

**主要功能**：
- PPO算法实现
- 策略梯度更新
- 价值函数训练
- KL散度惩罚
- 优势函数计算

**使用方法**：
```bash
python -m train.ppo_trainer
```

### 数据处理

#### `train/data_utils.py`
**作用**：提供数据处理工具

**主要类**：
- `DialogueFormatter`：对话格式化器
  - 格式化对话为模型输入
  - 添加特殊token
  - 创建input_ids和labels

- `StreamingDataset`：流式数据集
  - 支持大规模数据集
  - 动态加载
  - 支持预训练和SFT模式

### 工具脚本

#### `chat.py`
**作用**：提供增强的交互式对话界面，支持多种智能模式

**主要功能**：
- 加载训练好的模型
- 加载BPE分词器
- 提供命令行对话界面
- 支持多轮对话历史
- 支持记忆模式
- 支持推理模式
- 支持流式生成
- 支持模板模式
- 支持清空历史
- 支持退出

**使用方法**：
```bash
python chat.py
```

**对话命令**：
- 输入文本：与模型对话
- `clear`：清空对话历史
- `quit`：退出程序
- `memory`：切换记忆模式（开启/关闭）
- `reasoning`：切换推理模式（开启/关闭）
- `stream`：切换流式生成（开启/关闭）
- `template`：切换模板模式（开启/关闭）

#### `train_rlhf.py`
**作用**：整合RLHF训练流程

**主要功能**：
- 训练奖励模型
- PPO训练
- 整合RLHF流程

**使用方法**：
```bash
python train_rlhf.py
```

#### `train_smart.py`
**作用**：智能训练脚本，整合所有优化策略

**主要功能**：
- 自适应学习率
- 动态批次大小
- 智能梯度裁剪
- 早停机制
- 混合精度训练
- 完整的训练流程

**使用方法**：
```bash
python train_smart.py
```

#### `fix_tokenizer.py`
**作用**：修复分词器问题

**主要功能**：
- 检查分词器
- 修复编码问题
- 测试分词器

**使用方法**：
```bash
python fix_tokenizer.py
```

### 数据文件

#### `training_data/pretrain/pretrain_data.jsonl`
**作用**：预训练数据

**格式**：
```json
{"text": "这是一段预训练文本..."}
{"text": "这是另一段预训练文本..."}
```

#### `training_data/sft/sft_data.jsonl`
**作用**：SFT训练数据

**格式**：
```json
{
  "conversations": [
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "助手回复"}
  ]
}
```

#### `training_data/rlhf/preference_data.jsonl`
**作用**：RLHF偏好数据

**格式**：
```json
{
  "prompt": "用户输入",
  "chosen": "优选回复",
  "rejected": "拒绝回复"
}
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖**：
- torch：PyTorch深度学习框架
- transformers：HuggingFace Transformers库
- tokenizers：分词器库
- tqdm：进度条
- numpy：数值计算

### 2. 训练流程

#### 完整训练流程（推荐）

```bash
# 步骤1：训练分词器
python -m train.train_tokenizer

# 步骤2：预训练
python -m train.train_pretrain

# 步骤3：SFT微调
python -m train.train_sft

# 步骤4：RLHF训练（可选）
python train_rlhf.py
```

#### 使用智能训练脚本（简化流程）

```bash
# 智能训练脚本会自动执行完整的训练流程
python train_smart.py
```

### 3. 对话测试

```bash
python chat.py
```

## 模型工作原理

### 1. 预训练阶段

**目标**：学习语言的基本统计规律和语义信息

**过程**：
1. 加载预训练数据（纯文本）
2. 使用BPE分词器将文本编码为token序列
3. 将token序列输入SmartHaoAI模型
4. 模型预测下一个token
5. 计算预测token和实际token之间的交叉熵损失
6. 使用AdamW优化器更新模型参数
7. 重复以上步骤多个epoch

**关键点**：
- 使用因果掩码确保每个token只能看到之前的token
- 使用混合精度训练加速训练
- 使用梯度累积支持大批量训练
- 使用梯度裁剪防止梯度爆炸
- 使用学习率预热稳定训练初期

### 2. SFT阶段

**目标**：学习遵循指令的能力

**过程**：
1. 加载预训练模型
2. 加载SFT数据（对话数据）
3. 格式化对话数据（添加<|im_start|>和<|im_end|>标记）
4. 创建掩码（只计算assistant回复的损失）
5. 微调模型
6. 保存SFT模型

**关键点**：
- 使用较低的学习率（5e-5）
- 使用较小的批次大小（4）
- 只计算assistant回复的损失
- 保留预训练的知识

### 3. RLHF阶段

**目标**：对齐人类偏好

**过程**：
1. **奖励模型训练**：
   - 加载偏好数据
   - 训练奖励模型学习人类偏好
   - 奖励模型输出奖励分数

2. **PPO训练**：
   - 使用SFT模型作为初始策略
   - 生成回复
   - 使用奖励模型评估回复
   - 计算优势函数
   - 使用PPO算法更新策略
   - 添加KL散度惩罚防止策略偏离

**关键点**：
- 平衡探索和利用
- 使用价值函数估计
- 添加KL散度惩罚
- 使用小批量更新

### 4. 推理阶段

**目标**：生成智能回复

**增强功能**：
- **多模式推理**：支持普通对话、链式推理、流式生成、记忆增强等多种模式
- **智能提示词**：自动选择最佳提示词模板
- **对话记忆**：记住多轮对话历史
- **推理增强**：支持多步逻辑推理
- **流式生成**：实时显示生成过程

**过程**：
1. 加载训练好的模型
2. 加载BPE分词器
3. 分析用户输入意图
4. 自动选择合适的提示词模板
5. 检索相关对话记忆
6. 编码输入序列
7. 根据模式选择生成策略
8. 生成回复
9. 更新对话记忆
10. 返回回复

**生成策略**：
- 使用采样生成（do_sample=True）
- 使用温度参数控制多样性（temperature=0.8）
- 使用top-p采样（top_p=0.95）
- 使用top-k采样（top_k=50）
- 使用重复惩罚（repetition_penalty=1.2）
- 支持推理增强生成

### 5. 滑动窗口注意力机制

**目标**：高效处理长序列输入

**工作原理**：
1. 输入序列进入注意力层
2. 对于每个位置，计算局部窗口范围
3. 只在窗口范围内计算注意力权重
4. 滑动窗口处理整个序列
5. 减少计算复杂度和内存占用
6. 支持更长的上下文长度

**优势**：
- 局部注意力计算降低复杂度
- 可配置窗口大小适应不同场景
- 长序列处理能力显著提升
- 内存效率优化支持更大模型

### 6. 智能推理模块

**目标**：提升模型的逻辑推理能力

**工作原理**：
1. 输入问题通过词嵌入层转换为向量
2. 向量经过多层Transformer块处理
3. 推理模块接收Transformer的输出
4. 推理模块执行多步推理过程
5. 自注意力层捕获推理步骤间的依赖
6. 交叉注意力层融合外部上下文
7. 生成推理结果和置信度
8. 推理结果与原始输出融合

### 7. 神经符号推理系统 (Neuro-Symbolic Reasoning)

**目标**：结合神经网络的模式识别能力和符号系统的逻辑推理能力，实现可解释、鲁棒的智能推理

#### 7.1 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                  神经符号推理系统架构                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│   │   输入文本    │─────▶│  实体提取器   │─────▶│ 知识图谱  │ │
│   │              │      │  (BIO标注)   │      │  查询    │ │
│   └──────────────┘      └──────────────┘      └────┬─────┘ │
│                                                    │       │
│                                                    ▼       │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│   │   融合输出    │◀─────│  神经-符号   │◀─────│ 符号知识  │ │
│   │              │      │   融合层     │      │  注入    │ │
│   └──────────────┘      └──────┬───────┘      └──────────┘ │
│                                │                          │
│                                ▼                          │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│   │  Transformer │─────▶│  神经推理    │─────▶│ 符号约束  │ │
│   │    隐藏状态   │      │   网络      │      │  层      │ │
│   └──────────────┘      └──────────────┘      └──────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 7.2 核心组件

##### 7.2.1 知识图谱 (KnowledgeGraph)

**作用**：存储和管理结构化知识

**特点**：
- **三元组表示**：使用 (主体, 谓词, 客体) 形式存储事实
- **常识知识库**：预置AI、科学等领域的基础知识
- **关系推理**：支持传递性推理和包含关系推理
- **模糊查询**：支持关键词模糊匹配
- **自动清理**：自动移除低置信度事实，防止知识库膨胀

**示例**：
```python
from model.neuro_symbolic import KnowledgeGraph, SymbolicFact

# 创建知识图谱
kg = KnowledgeGraph()

# 添加事实
fact = SymbolicFact(
    subject="人工智能",
    predicate="包括",
    object="机器学习",
    confidence=1.0,
    source="builtin"
)
kg.add_fact(fact)

# 查询知识
results = kg.query(subject="人工智能")
# 返回: [人工智能包括机器学习, 人工智能包括深度学习, ...]

# 模糊查询
fuzzy_results = kg.fuzzy_query("学习")
# 返回包含"学习"关键词的所有事实
```

##### 7.2.2 神经-符号桥梁 (NeuralSymbolicBridge)

**作用**：连接神经网络和符号系统的中间层

**功能**：
- **实体提取**：从Transformer隐藏状态中提取实体（使用BIO标注）
- **关系预测**：预测实体间的关系类型（是、包括、应用等）
- **事实验证**：验证符号事实的可信度（0-1之间的分数）
- **嵌入转换**：将符号实体转换为神经嵌入向量

**工作原理**：
```
隐藏状态 [batch, seq_len, hidden] 
    ↓
实体提取器 (Linear + LayerNorm + ReLU + Dropout)
    ↓
BIO标注 [batch, seq_len, 3]  (Begin/Inside/Outside)
    ↓
实体边界识别
    ↓
实体列表 [{"start": 0, "end": 3, "confidence": 0.95}, ...]
```

##### 7.2.3 神经符号推理器 (NeuroSymbolicReasoner)

**作用**：核心融合模块，结合神经和符号推理

**推理流程**：
1. **符号推理路径**：
   - 从输入中提取实体
   - 查询知识图谱获取相关事实
   - 将符号知识转换为神经嵌入
   - 注入到隐藏状态中

2. **神经推理路径**：
   - 使用Transformer编码器处理隐藏状态
   - 捕获复杂的模式特征
   - 生成神经输出

3. **融合机制**：
   ```
   融合门控 = Sigmoid(Linear([神经输出; 符号上下文]))
   融合输出 = 融合门控 × 神经输出 + (1 - 融合门控) × 符号上下文
   ```

4. **输出生成**：
   - 使用融合后的表示生成logits
   - 返回输出和符号推理信息

**容错设计**：
- 符号推理失败时，自动回退到纯神经推理
- 网络初始化失败时，自动降级为简单MLP
- 所有子模块包含独立的错误处理

##### 7.2.4 符号约束层 (SymbolicConstraintLayer)

**作用**：在生成过程中应用符号约束，提高输出质量

**约束类型**：
- **软禁用**：降低无意义token的概率（如"硫"、"黏"等乱码字符）
- **重复惩罚**：检测并惩罚重复模式
- **偏好增强**：提升高质量token的概率
- **数值稳定**：将logits限制在[-50, 50]范围内

**示例**：
```python
from model.neuro_symbolic import SymbolicConstraintLayer

# 创建约束层
constraint_layer = SymbolicConstraintLayer(vocab_size=16384)

# 添加禁用token（乱码字符）
forbidden_tokens = [token_id_for_硫, token_id_for_黏, ...]
constraint_layer.add_forbidden_tokens(forbidden_tokens)

# 应用约束
constrained_logits = constraint_layer.apply_constraints(
    logits=logits,
    generated_ids=generated_ids
)
```

#### 7.3 使用方法

##### 在对话系统中启用神经符号推理

```python
# chat.py 中配置
from model.neuro_symbolic import NeuroSymbolicReasoner, SymbolicConstraintLayer

class EnhancedChatSystem:
    def __init__(self, model, tokenizer, device):
        # ... 其他初始化 ...
        
        # 初始化神经符号系统
        hidden_size = getattr(model.config, 'n_embd', 1024)
        vocab_size = getattr(model.config, 'vocab_size', 16384)
        self.neuro_symbolic = NeuroSymbolicReasoner(hidden_size, vocab_size).to(device)
        self.symbolic_constraint = SymbolicConstraintLayer(vocab_size)
        
        # 启用神经符号推理
        self.use_neuro_symbolic = True
```

##### 自定义知识注入

```python
# 在chat.py的_init_knowledge_base方法中添加自定义知识
def _init_knowledge_base(self):
    extended_facts = [
        SymbolicFact("深度学习", "是", "机器学习的子集", 1.0, "builtin"),
        SymbolicFact("神经网络", "灵感来源", "人脑结构", 1.0, "builtin"),
        SymbolicFact("自然语言处理", "简称", "NLP", 1.0, "builtin"),
        # 添加更多领域知识...
    ]
    
    for fact in extended_facts:
        self.neuro_symbolic.knowledge_graph.add_fact(fact)
```

##### 查询知识图谱

```python
def _query_knowledge(self, query: str) -> str:
    """查询知识图谱获取相关信息"""
    # 模糊查询相关知识
    related_facts = self.neuro_symbolic.knowledge_graph.fuzzy_query(query, max_results=5)
    
    if not related_facts:
        return ""
    
    # 构建知识上下文
    knowledge_text = "相关知识：\n"
    for fact in related_facts:
        knowledge_text += f"- {fact.subject}{fact.predicate}{fact.object}\n"
    
    return knowledge_text
```

#### 7.4 优势与特点

##### 优势
1. **可解释性**：符号推理提供明确的推理路径，便于调试和优化
2. **鲁棒性**：知识图谱提供稳定的常识支撑，减少对训练数据的依赖
3. **灵活性**：神经网络处理复杂模式，符号系统处理结构化知识
4. **质量控制**：符号约束防止生成乱码和重复内容
5. **容错性**：模块化设计，单点故障不影响整体系统

##### 性能优化
- **间隔执行**：符号推理每3次调用执行一次，降低计算开销
- **倒排索引**：加速知识图谱查询
- **知识库清理**：自动移除低置信度事实，控制内存占用
- **软约束**：使用概率调整而非硬截断，保持生成流畅性

#### 7.5 应用场景

神经符号推理系统特别适用于：
- **知识密集型任务**：问答、知识推理、事实验证
- **需要可解释性的场景**：医疗诊断、法律咨询、教育辅导
- **低资源环境**：利用符号知识弥补训练数据不足
- **高质量生成需求**：内容创作、技术文档生成

#### 7.6 调试与监控

系统提供详细的符号推理信息：
```python
output_logits, symbolic_info = neuro_symbolic_reasoner(...)

print(f"提取事实数: {symbolic_info['extracted_facts']}")
print(f"融合权重: {symbolic_info['fusion_weight']:.3f}")
print(f"知识库大小: {symbolic_info['knowledge_graph_size']}")
print(f"调用次数: {symbolic_info['call_count']}")
print(f"错误次数: {symbolic_info['error_count']}")
```

日志记录：
- 使用Python logging模块记录运行状态
- 错误信息分级（debug/warning/error）
- 支持错误计数限制，防止日志爆炸
9. 输出最终回答

**优势**：
- 多步推理支持复杂逻辑
- 自注意力机制捕获推理依赖
- 置信度计算评估推理质量
- 状态管理跟踪推理过程

### 7. 对话记忆模块

**目标**：实现多轮对话的上下文理解

**工作原理**：
1. 对话开始时初始化记忆存储
2. 每轮对话后，记忆模块评估对话重要性
3. 重要对话被添加到历史记忆中
4. 记忆模块对对话进行主题聚类
5. 当记忆达到容量上限时，执行记忆压缩
6. 新对话输入时，检索相关历史记忆
7. 相关记忆与当前输入融合
8. 模型基于融合后的上下文生成回复

**优势**：
- 重要性评估过滤无关对话
- 主题聚类组织记忆结构
- 记忆压缩控制内存使用
- 上下文检索提高相关性

### 8. 智能提示词工程

**目标**：自动生成和选择最优提示词

**工作原理**：
1. 分析用户输入的意图和类型
2. 从预定义模板库中选择合适的模板
3. 根据对话历史和上下文填充模板
4. 生成针对当前任务的优化提示词
5. 将提示词与用户输入组合
6. 输入模型生成回复

**优势**：
- 多模板支持不同任务类型
- 自动模板选择提高适配性
- 意图检测精准理解用户需求
- 上下文融合增强对话连贯性

## 配置说明

### 模型配置

```python
config = HaoAIConfig(
    vocab_size=16384,        # 词汇表大小
    n_layer=8,               # Transformer层数
    n_head=8,               # 注意力头数
    n_embd=1024,            # 嵌入维度
    dropout=0.1,             # Dropout率
    max_position_embeddings=2048,  # 最大序列长度
    use_cache=True,          # 是否使用KV Cache
    use_reasoning=True,      # 是否使用推理模块
    use_sliding_window=True, # 是否使用滑动窗口注意力
    window_size=512          # 滑动窗口大小
)
```

### 训练配置

```python
# 预训练配置
pretrain_config = PretrainConfig(
    batch_size=16,           # 批次大小
    lr=3e-4,               # 学习率
    epochs=300,             # 训练轮数
    accumulation_steps=4,    # 梯度累积步数
    mixed_precision=True,    # 混合精度训练
    gradient_clip=1.0,       # 梯度裁剪阈值
    warmup_steps=1000        # 预热步数
)

# SFT配置
sft_config = SFTConfig(
    batch_size=4,           # 批次大小
    lr=5e-5,               # 学习率
    epochs=300,             # 训练轮数
    accumulation_steps=8     # 梯度累积步数
)
```

### 生成配置

```python
# 生成参数
generation_config = {
    "max_length": 512,           # 最大生成长度
    "do_sample": True,           # 是否使用采样
    "temperature": 0.8,          # 温度参数
    "top_p": 0.95,              # Top-p采样
    "top_k": 50,                # Top-k采样
    "repetition_penalty": 1.2,   # 重复惩罚
    "pad_token_id": 0,          # 填充token ID
    "eos_token_id": 0            # 结束token ID
}
```

## 神经符号推理 - 小学生版讲解

### 什么是神经符号推理？

想象你有一个超级智能的机器人朋友，它有两个大脑：

** 第一个大脑：Transformer神经网络（模式识别专家）**
- 像是一个超级会认字的阅读高手
- 看到你写的字，能猜出下一个字是什么
- 比如你写"我爱吃"，它能猜出"苹果"、"香蕉"
- 但是有时候猜错，比如猜成"石头"（不能吃的东西）

**� 第二个大脑：符号知识库（记忆专家）**
- 像是一个超级会记笔记的学霸
- 有一个大笔记本，上面写着各种知识
- 比如"苹果是水果"、"水果可以吃"
- 但是不会自己写字，只会查笔记

**神经符号推理就是让两个大脑一起工作！**

---

### 为什么要让两个大脑一起工作？

**第一个大脑单独工作时**：
- 你问："苹果是什么？"
- 它可能会乱猜："苹果是...石头？"
- 猜错了！

**第二个大脑单独工作时**：
- 你问："苹果是什么？"
- 它说："查笔记...苹果是水果"
- 但是你没告诉它"苹果"这个词在哪

**两个大脑一起工作时**：
- 第一个大脑认出"苹果"这个词
- 第二个大脑查到"苹果是水果，可以吃"
- 回答正确！

---

### � 学校里的例子

#### 例子1：写作文

**普通AI写作文**：
- 第一个大脑（Transformer）：根据前面的字猜下一个字
- "今天天气很..." → 猜"好"
- "好" → 猜"我"
- "我" → 猜"去"
- 结果："今天天气很好我去..."（可能接不上，乱七八糟）

**神经符号AI写作文**：
- 第一个大脑：认出"今天天气很好"是开头
- 第二个大脑：查到"作文开头要说时间、地点、人物"
- 两个大脑商量：时间有了（今天），还差地点和人物
- 结果："今天天气很好，我和小明去公园玩。" 

#### 例子2：回答科学问题

**普通AI回答**：
- 你问："太阳是什么？"
- 第一个大脑猜："太阳是...月亮？"
- 错了！

**神经符号AI回答**：
- 第一个大脑：认出"太阳"这个词
- 第二个大脑：查到笔记"太阳是恒星"
- 两个大脑组合："太阳是一颗恒星，会发光发热"
- 完全正确！

---

### HaoAI是怎么做的？

#### 第一步：理解你说的话（Transformer神经网络）

就像你读课文时理解句子：

```
你问："什么是人工智能？"

Transformer大脑的工作：
- 看到"什么" → 知道这是个问题
- 看到"是" → 知道要找定义
- 看到"人工智能" → 这是一个关键词
- 把整句话连起来理解
```

Transformer特别厉害的地方：
- 能记住前面很多字（注意力机制）
- 知道"人工智能"是一个整体，不是分开的"人工"+"智能"
- 理解问题的意思

#### 第二步：查知识库（符号系统）

就像查百科全书：

```
符号大脑查笔记：
- 人工智能 → 是 → 计算机科学的分支
- 人工智能 → 包括 → 机器学习
- 人工智能 → 包括 → 深度学习
- 人工智能 → 应用 → 图像识别、语音助手
```

符号系统像是一个超级整理好的笔记本：
- 所有知识都写成"A是B"、"A包括C"的形式
- 很容易查找
- 不会忘记，也不会记错

#### 第三步：两个大脑开会商量（融合）

```
Transformer大脑说：
"用户问的是定义类问题，需要解释清楚"

符号大脑说：
"我查到人工智能是计算机科学，包括机器学习"

两个大脑一起决定：
"我们应该说：人工智能是计算机科学的一个分支，
 它包括机器学习等技术，可以应用在很多地方"
```

融合就像是两个好朋友一起做作业：
- 一个负责理解题目
- 一个负责查资料
- 一起写出最好的答案

#### 第四步：检查答案对不对（符号约束）

```
符号大脑检查：
- "人工智能是计算机科学" 笔记里有的
- "包括机器学习" 笔记里有的
- 没有出现奇怪的字（比如"硫"、"黏"）
- 没有重复说同一句话 
```

就像老师检查作业：
- 看看有没有错别字
- 看看有没有写重复
- 确保答案合理

---

### 更形象的比喻

#### 比喻1：翻译官和外交官

**Transformer神经网络** = 翻译官
- 你对外国朋友说中文
- 翻译官听懂你说的话
- 理解你的意思和情绪

**符号系统** = 外交官
- 知道很多国际礼仪和规则
- 知道什么该说，什么不该说
- 有厚厚的礼仪手册

**神经符号系统** = 翻译官 + 外交官
- 翻译官理解你的意思
- 外交官查手册确保说得得体
- 对外国朋友说最合适的英文
- 既准确又礼貌！

#### 比喻2：侦探和档案管理员

**Transformer神经网络** = 侦探
- 观察案发现场的细节
- 发现线索之间的联系
- 推理出可能发生了什么

**符号系统** = 档案管理员
- 管理着巨大的档案室
- 记录着所有已知案件
- 能快速找到相似案例

**神经符号系统** = 侦探 + 档案管理员
- 侦探发现新线索
- 档案管理员查找旧案件
- 一起破案！
- 既会推理，又有经验

#### 比喻3：厨师和营养学家

**Transformer神经网络** = 厨师
- 会切菜、炒菜
- 知道什么食材搭配好吃
- 能做出美味的菜

**符号系统** = 营养学家
- 知道每种食物的营养成分
- 知道什么对身体好
- 有科学的营养知识

**神经符号系统** = 厨师 + 营养学家
- 厨师做出好吃的菜
- 营养学家确保营养均衡
- 既好吃又健康！

---

### 有什么好处？

#### 1. 更准确 

就像两个人一起检查作业：
- 一个人可能看错
- 两个人都看错的可能性很小
- Transformer猜 + 符号系统验证 = 更准确

#### 2. 不会胡说八道 

**只用Transformer时**：
- 可能会生成"人工智能是硫黏蔬织"
- 这些字连在一起没有意义

**加上符号系统后**：
- 符号系统说："等等，'硫'不是知识库里的字！"
- 阻止胡说八道
- 只生成合理的回答

#### 3. 能解释为什么 

**你问AI**："你为什么说人工智能包括机器学习？"

**AI可以回答**："因为我查到知识库里写着：人工智能 → 包括 → 机器学习"

就像学生能说出答案是从哪本书、哪一页找到的！

#### 4. 学习更快 

**只用Transformer**：
- 需要看几百万本书才能学会
- 学得很慢

**加上符号系统**：
- 直接把知识写进笔记本
- 就像你背课文一样快
- 还能随时更新知识

---

###  在HaoAI中怎么使用？

#### 开启神经符号推理

```python
# 就像打开一个超级开关
chat_system.use_neuro_symbolic = True  # 开启两个大脑！
```

开启后，AI会：
1. 用Transformer理解你的话
2. 用符号系统查知识
3. 两个大脑一起回答

#### 给AI添加新知识

```python
# 就像往AI的笔记本里写东西
AI学习("太阳", "是", "恒星")
AI学习("地球", "是", "行星")
AI学习("水", "化学式", "H2O")
AI学习("人类", "需要", "氧气")
```

以后你问"太阳是什么"，AI就会查笔记回答！

#### 问问题，看AI怎么思考

```python
你问："什么是人工智能？"

AI的思考过程：
1. Transformer大脑：理解这是问定义的问题
2. 符号大脑：查到"人工智能是计算机科学分支"
3. 两个大脑融合：组织成通顺的回答
4. 符号约束：检查没有乱码和重复
5. 最终回答："人工智能是计算机科学的一个分支..."
```

---

###  总结

**神经符号推理 = Transformer（理解专家）+ 符号系统（知识专家）**

|  | Transformer神经网络 | 符号系统 |
|---|---|---|
| **像什么** | 超级阅读高手 | 超级笔记本 |
| **擅长** | 理解句子、猜下一个字 | 记知识、查资料 |
| **做什么** | 理解你说的话 | 提供准确知识 |
| **缺点** | 可能猜错 | 不会自己理解 |

**一起工作时**：
- Transformer理解问题 
- 符号系统提供知识 
- 融合生成答案 
- 约束检查质量 

就像你学习一样：
- 用眼睛看书理解内容（Transformer）
- 用笔记本记重点（符号系统）
- 考试时两个都用上（神经符号推理）
- 就能考出好成绩！

---

## 常见问题

### Q1: 训练时显存不足怎么办？

**A**: 可以尝试以下方法：
1. 减小批次大小（batch_size）
2. 减小模型大小（n_layer, n_embd）
3. 使用梯度累积（accumulation_steps）
4. 启用混合精度训练（mixed_precision=True）
5. 减小序列长度（block_size）

### Q2: 如何提高生成质量？

**A**: 可以尝试以下方法：
1. 增加预训练数据量
2. 增加训练轮数
3. 调整生成参数（temperature, top_p, top_k）
4. 使用RLHF对齐人类偏好
5. 使用更大的模型

### Q3: 如何加速训练？

**A**: 可以尝试以下方法：
1. 使用GPU训练
2. 启用混合精度训练
3. 增大批次大小
4. 使用多GPU训练（需要修改代码）
5. 使用更快的硬件（如A100 GPU）

### Q4: 生成的回复重复怎么办？

**A**: 可以尝试以下方法：
1. 增加重复惩罚（repetition_penalty）
2. 增加温度参数（temperature）
3. 使用top-p和top-k采样
4. 增加训练数据多样性
5. 使用RLHF训练

### Q5: 如何添加新的训练数据？

**A**: 按照以下格式添加数据：

**预训练数据**：
```json
{"text": "你的预训练文本..."}
```

**SFT数据**：
```json
{
  "conversations": [
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "助手回复"}
  ]
}
```

**RLHF数据**：
```json
{
  "prompt": "用户输入",
  "chosen": "优选回复",
  "rejected": "拒绝回复"
}
```

## 技术细节

### BPE分词器

HaoAI使用BPE（字节对编码）分词器，具有以下特点：
- 高效的子词分词
- 支持未知词处理
- 词汇表大小可配置（默认16384）
- 支持特殊token（<|im_start|>, <|im_end|>）

### 旋转位置编码（RoPE）

RoPE是一种相对位置编码方法，具有以下优点：
- 支持任意长度的序列
- 相对位置信息
- 不需要额外的位置嵌入参数
- 与注意力机制自然集成

### 门控机制

门控机制用于动态调整信息流动：
- 注意力门控：调整注意力权重
- 残差门控：控制残差连接的强度
- 可学习参数：通过训练自动学习

### 混合精度训练

混合精度训练使用FP16进行计算，具有以下优点：
- 减少显存占用
- 加速训练速度
- 使用FP32进行梯度累积，保证精度

### KV Cache

KV Cache用于加速推理，具有以下优点：
- 缓存过去的键值对
- 避免重复计算
- 显著加速生成过程

## 许可证

MIT License

## 建议系统配置

最低电脑配置：

处理器：志强/酷睿/锐龙/霄龙6核心以上（推荐24核心及以上）

内存：不少于16g（推荐容量64/128g及以上ddr4/ddr5内存容量）

显卡：对话模式下显存不少于4G/训练模式下显存不少于24g（推荐v100/3090/3090TI/4090/5090/a100/h100等等）个人观念：使用游戏显卡不仅便宜而且到时候可以卖给臭打游戏的方便回本，企业的话建议上a100及以上专业计算卡集群部署到IDC机房！

硬盘不少于256g（推荐1t硬盘及以上）

Minimum computer configuration:
Processor: Zhiqiang/Core/Ryzen/Xiaolong 6-core or above (recommended 24 core or above)
Memory: not less than 16g (recommended capacity of 64/128g or above DDR4/DDR5 memory capacity)
Graphics card: Video memory of no less than 4GB in dialogue mode and no less than 24GB in training mode (recommended v100/3090/3090TI/4090/5090/a100/h100, etc.)
The hard drive should not be less than 256g (recommended is a 1TB hard drive or above)

## 贡献

项目开发人员：

张裕浩（zhangyuhao/Markhao）linkedin（领英）:www.linkedin.com/in/裕浩-张-bb1bb93ab 邮箱：zhangyuhao@haoxun.cc  注：欢迎大家反馈项目问题！

欢迎贡献代码、报告问题或提出建议！

## 联系方式

如有问题，请通过以下方式联系：
- 发送邮件到zhangyuhao@haoxun.cc

## 致谢

感谢以下开源项目：
- Transformers
- PyTorch
- Tokenizers

## 版本
当前版本1.0.0 #haoai创世版本（第一个版本）2025/8/16开发完成

## 免责声明
禁止将该模型用于任何非法用途！（例如：黄赌毒，欺诈诈骗等！）
因为该模型造成的任何损失后果自负（例如客服场景造成随意承诺客户和智能会计造成税务风险等），本项目不承担任何责任！这些场景需要一定编程能力和人工智能相关能力二次深度开发！！！
Prohibited from using this model for any illegal purposes! (e.g., pornography, gambling, drugs, fraud, scams, etc.)
Any losses caused by this model shall be borne by the user (e.g., arbitrary commitments to customers in customer service scenarios or tax risks in intelligent accounting), and this project assumes no responsibility! These scenarios require secondary in-depth development with certain programming and AI-related capabilities!!!

## 关于我们
<img width="1706" height="1664" alt="500247749-b12349c6-2919-4526-bb27-5ceb928d84a4" src="https://github.com/user-attachments/assets/cd9eff87-a746-46ed-b929-7aa10c173658" />
