# HaoAI - 轻量级语言模型实现

## 项目概述

HaoAI 是一个轻量级语言模型实现项目，基于 Transformer 架构设计，提供可定制和可扩展的语言模型架构。项目采用模块化设计，支持标准 Transformer 结构，并实现了完整的训练流程，包括预训练、监督微调（SFT）和人类反馈强化学习（RLHF）。

## 核心特性

### 模型架构

HaoAI 实现了以下核心组件：

#### 1. 智能注意力机制 (SmartAttention)
- **多头注意力 (MultiHeadAttention)**：将输入分割到多个注意力头，并行计算不同子空间的注意力
- **旋转位置编码 (RoPE)**：通过旋转矩阵编码位置信息，支持任意长度的序列
- **门控机制**：动态调整注意力权重，提高模型表达能力
- **动态温度调节**：根据输入自适应调整softmax温度，平衡多样性和确定性
- **智能dropout策略**：降低注意力dropout，保留更多语义信息

#### 2. 智能前馈网络 (SmartFeedForward)
- **门控残差连接**：通过可学习的门控参数控制信息流动
- **深度监督机制**：深层网络添加辅助投影，帮助梯度流动
- **专家网络**：模拟MoE（Mixture of Experts）架构，提高模型容量

#### 3. 智能Transformer块 (SmartTransformerBlock)
- **预注意力归一化**：在注意力计算前进行归一化，提高训练稳定性
- **预前馈归一化**：在前馈网络前进行归一化
- **门控残差连接**：使用可学习参数控制残差连接的强度
- **缓存支持**：支持KV Cache，加速推理过程

#### 4. SmartHaoAI 主模型
- **词嵌入层**：将token ID映射为稠密向量
- **多层Transformer块**：堆叠多个Transformer块
- **最终层归一化**：对输出进行归一化
- **语言模型头**：预测下一个token的概率分布
- **因果掩码**：确保每个token只能看到之前的token
- **推理模块集成**：内置ReasoningModule，支持多步推理
- **对话记忆支持**：集成对话记忆，记住历史对话
- **滑动窗口注意力**：支持长序列处理的局部注意力
- **智能生成方法**：generate_with_reasoning()支持推理增强的生成

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
- **动态批次大小**：根据GPU内存自动调整批次大小
- **智能梯度裁剪**：自适应梯度裁剪阈值
- **早停机制**：防止过拟合
- **混合精度训练**：加速训练并减少显存占用
- **梯度累积**：模拟大批量训练
- **学习率预热**：稳定训练初期

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
  - `n_layer`：Transformer层数（默认8）
  - `n_head`：注意力头数（默认8）
  - `n_embd`：嵌入维度（默认1024）
  - `dropout`：dropout率（默认0.1）
  - `max_position_embeddings`：最大序列长度（默认2048）
  - `use_reasoning`：是否使用推理模块（默认True）
  - `use_sliding_window`：是否使用滑动窗口注意力（默认True）
  - `window_size`：滑动窗口大小（默认512）

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

## 贡献

欢迎贡献代码、报告问题或提出建议！

## 联系方式

如有问题，请通过以下方式联系：
- 提交Issue
- 发送邮件
- 加入讨论组

## 致谢

感谢以下开源项目：
- Transformers
- PyTorch
- Tokenizers

