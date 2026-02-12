"""
RLHF训练配置
包含奖励模型训练和PPO强化学习的配置参数
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class RewardModelConfig:
    """奖励模型配置"""
    
    # 模型架构
    hidden_size: int = 1024
    num_layers: int = 4
    dropout: float = 0.1
    
    # 训练参数
    learning_rate: float = 1e-5
    batch_size: int = 8
    epochs: int = 10
    gradient_accumulation_steps: int = 4
    
    # 数据配置
    max_sequence_length: int = 1024
    comparison_pairs_per_sample: int = 3
    
    # 损失函数
    margin: float = 0.1  # 对比损失边界
    temperature: float = 0.1  # 对比损失温度
    
    # 输出路径
    model_dir: str = "weight/haoai_reward_model/"

@dataclass
class PPOConfig:
    """PPO强化学习配置"""
    
    # PPO算法参数
    learning_rate: float = 5e-7  # 降低学习率提高稳定性
    clip_epsilon: float = 0.2
    gamma: float = 0.99  # 折扣因子
    lam: float = 0.95  # GAE参数
    
    # 训练参数
    batch_size: int = 8  # 增加batch_size
    minibatch_size: int = 2  # 增加minibatch_size
    ppo_epochs: int = 4
    max_episode_length: int = 256  # 减少序列长度提高稳定性
    
    # 价值函数训练
    value_coef: float = 0.5  # 增加价值损失系数
    entropy_coef: float = 0.01  # 熵系数
    
    # 训练步数
    total_timesteps: int = 5000  # 减少总步数
    save_frequency: int = 500  # 更频繁保存
    
    # KL散度约束
    target_kl: float = 0.01
    kl_coef: float = 0.2
    
    # 数值稳定性参数
    max_logit: float = 50.0  # 限制logit最大值
    min_prob: float = 1e-8  # 最小概率
    max_log_ratio: float = 20.0  # 最大log比率

@dataclass
class RLHFConfig:
    """RLHF整体配置"""
    
    # 阶段配置
    enable_reward_training: bool = True
    enable_ppo_training: bool = True
    
    # 模型路径
    sft_model_dir: str = "weight/haoai_sft_model/"
    reward_model_dir: str = "weight/haoai_reward_model/"
    rlhf_model_dir: str = "weight/haoai_rlhf_model/"
    
    # 数据配置
    preference_data_file: str = "training_data/rlhf/preference_data.jsonl"
    
    # 训练策略
    warmup_steps: int = 100
    eval_frequency: int = 500
    
    # 奖励模型配置
    reward_config: RewardModelConfig = field(default_factory=RewardModelConfig)
    
    # PPO配置
    ppo_config: PPOConfig = field(default_factory=PPOConfig)
    
    def get_reward_config(self) -> Dict[str, Any]:
        """获取奖励模型配置"""
        return {
            "hidden_size": self.reward_config.hidden_size,
            "num_layers": self.reward_config.num_layers,
            "learning_rate": self.reward_config.learning_rate,
            "batch_size": self.reward_config.batch_size,
            "epochs": self.reward_config.epochs
        }
    
    def get_ppo_config(self) -> Dict[str, Any]:
        """获取PPO配置"""
        return {
            "learning_rate": self.ppo_config.learning_rate,
            "clip_epsilon": self.ppo_config.clip_epsilon,
            "gamma": self.ppo_config.gamma,
            "batch_size": self.ppo_config.batch_size,
            "ppo_epochs": self.ppo_config.ppo_epochs
        }

# 预设配置
PRESET_CONFIGS = {
    "balanced": None,
    "fast": None,
    "quality": None
}

# 延迟初始化预设配置
def _init_preset_configs():
    global PRESET_CONFIGS
    if PRESET_CONFIGS["balanced"] is None:
        PRESET_CONFIGS["balanced"] = RLHFConfig()
        PRESET_CONFIGS["fast"] = RLHFConfig(
            reward_config=RewardModelConfig(
                epochs=5,
                batch_size=16
            ),
            ppo_config=PPOConfig(
                total_timesteps=5000,
                batch_size=8
            )
        )
        PRESET_CONFIGS["quality"] = RLHFConfig(
            reward_config=RewardModelConfig(
                epochs=15,
                learning_rate=5e-6
            ),
            ppo_config=PPOConfig(
                learning_rate=5e-7,
                ppo_epochs=8
            )
        )

def get_rlhf_config(preset: str = "balanced") -> RLHFConfig:
    """获取预设的RLHF配置"""
    # 延迟初始化预设配置
    _init_preset_configs()
    
    if preset in PRESET_CONFIGS:
        return PRESET_CONFIGS[preset]
    else:
        print(f"警告: RLHF预设配置 '{preset}' 不存在，使用默认配置")
        return RLHFConfig()