"""
HaoAI模型性能优化配置
包含各种智能训练策略和优化参数
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class OptimizationConfig:
    """性能优化配置类"""
    
    # 学习率优化
    learning_rate_strategy: str = "adaptive"  # "adaptive", "cosine", "linear"
    min_lr_ratio: float = 0.05  # 最小学习率比例
    warmup_ratio: float = 0.1  # warmup阶段比例
    min_learning_rate: float = 1e-6  # 最小学习率
    max_learning_rate: float = 1e-3  # 最大学习率
    
    # 梯度优化
    gradient_clip_norm: float = 5.0  # 增加梯度裁剪阈值，减少异常检测
    gradient_accumulation_steps: int = 8
    adaptive_gradient_clip: bool = True
    use_gradient_scaling: bool = True  # 是否使用梯度缩放
    use_adaptive_gradient_clip: bool = True  # 是否使用自适应梯度裁剪
    adaptive_learning_rate: bool = True  # 是否使用自适应学习率
    
    # 批次大小优化
    dynamic_batch_size: bool = True
    max_batch_size: int = 8
    min_batch_size: int = 1
    memory_threshold_high: float = 0.8  # 内存使用率阈值（高）
    memory_threshold_low: float = 0.6   # 内存使用率阈值（低）
    
    # 提前停止策略
    early_stopping_patience: int = 50  # 增加耐心值，允许更多轮次
    early_stopping_window: int = 200  # 增加窗口大小，更平滑的判断
    
    # 模型架构优化
    use_mixed_precision: bool = True
    attention_dropout_ratio: float = 0.5  # 注意力dropout比例
    activation_dropout_ratio: float = 0.3  # 激活函数dropout比例
    
    # 数据预处理优化
    dynamic_sequence_length: bool = True
    max_sequence_length: int = 2048
    sequence_length_percentile: int = 75  # 序列长度分位数策略
    
    # 性能监控
    monitoring_window_size: int = 100
    checkpoint_frequency_early: int = 50  # 早期训练检查点频率
    checkpoint_frequency_late: int = 200  # 后期训练检查点频率
    
    # 智能调度策略
    smart_scheduler: bool = True  # 是否使用智能调度器
    
    def get_scheduler_config(self, total_steps: int) -> Dict[str, Any]:
        """获取学习率调度器配置"""
        return {
            "total_steps": total_steps,
            "warmup_steps": int(total_steps * self.warmup_ratio),
            "min_lr_ratio": self.min_lr_ratio,
            "strategy": self.learning_rate_strategy
        }
    
    def get_gradient_config(self) -> Dict[str, Any]:
        """获取梯度优化配置"""
        return {
            "clip_norm": self.gradient_clip_norm,
            "accumulation_steps": self.gradient_accumulation_steps,
            "adaptive": self.adaptive_gradient_clip
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据预处理配置"""
        return {
            "dynamic_length": self.dynamic_sequence_length,
            "max_length": self.max_sequence_length,
            "percentile": self.sequence_length_percentile
        }

# 预定义的优化配置预设
PRESET_CONFIGS = {
    "balanced": OptimizationConfig(),
    "performance": OptimizationConfig(
        learning_rate_strategy="adaptive",
        min_lr_ratio=0.02,
        gradient_accumulation_steps=16,
        use_mixed_precision=True,
        dynamic_batch_size=True
    ),
    "stability": OptimizationConfig(
        learning_rate_strategy="cosine",
        min_lr_ratio=0.1,
        gradient_clip_norm=0.5,
        adaptive_gradient_clip=False,
        early_stopping_patience=20
    ),
    "fast_training": OptimizationConfig(
        learning_rate_strategy="linear",
        warmup_ratio=0.05,
        gradient_accumulation_steps=4,
        dynamic_batch_size=False,
        checkpoint_frequency_early=100,
        checkpoint_frequency_late=500
    )
}

def get_optimization_config(preset: str = "balanced") -> OptimizationConfig:
    """获取预设的优化配置"""
    if preset in PRESET_CONFIGS:
        return PRESET_CONFIGS[preset]
    else:
        print(f"警告: 预设配置 '{preset}' 不存在，使用默认配置")
        return OptimizationConfig()