"""
奖励模型架构
用于学习人类偏好，为RLHF提供奖励信号
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import PretrainedConfig, PreTrainedModel

class RewardModelConfig(PretrainedConfig):
    """奖励模型配置"""
    model_type = "haoai_reward"
    
    def __init__(
        self,
        vocab_size: int = 16384,
        hidden_size: int = 1024,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        
        super().__init__(**kwargs)

class RewardModelHead(nn.Module):
    """奖励模型头部网络"""
    
    def __init__(self, config: RewardModelConfig):
        super().__init__()
        self.config = config
        
        # 多层感知机用于奖励预测
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)  # 输出单个奖励值
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        try:
            # 使用最后一个token的隐藏状态
            last_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
            
            # 层归一化
            normalized = self.layer_norm(last_hidden)
            
            # 通过MLP得到奖励值
            reward = self.mlp(normalized)
            
            # 安全地squeeze
            if reward.dim() > 1:
                reward = reward.squeeze(-1)
            
            return reward  # [batch_size]
        except Exception as e:
            print(f"⚠️  RewardModelHead forward失败: {e}")
            # 返回默认值
            batch_size = hidden_states.size(0)
            return torch.zeros(batch_size, device=hidden_states.device)

class RewardModel(nn.Module):
    """奖励模型"""
    
    def __init__(self, config: RewardModelConfig, base_model=None):
        super().__init__()
        
        self.config = config
        
        # 使用预训练的语言模型作为基础
        if base_model is None:
            from .model import SmartHaoAI, HaoAIConfig
            
            # 创建基础模型配置
            base_config = HaoAIConfig(
                vocab_size=config.vocab_size,
                n_embd=config.hidden_size,
                n_layer=config.num_layers,
                dropout=config.dropout,
                max_position_embeddings=config.max_position_embeddings
            )
            
            self.base_model = SmartHaoAI(base_config)
        else:
            self.base_model = base_model
        
        # 冻结基础模型的参数（可选）
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 奖励头部
        self.reward_head = RewardModelHead(config)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Tuple[torch.Tensor, ...]:
        """前向传播"""
        
        # 通过基础模型获取隐藏状态
        # 直接访问基础模型的内部组件，避免调用forward方法导致递归
        hidden_states = self.base_model.embed(input_ids)
        
        if attention_mask is not None:
            # 确保attention mask的形状正确
            # attention_mask shape: (B, T) → 需要扩展为 (B, 1, 1, T)
            extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(hidden_states.dtype).min
        else:
            extended_attention_mask = None
        
        # 遍历所有Transformer块
        for i, block in enumerate(self.base_model.blocks):
            hidden_states, _ = block(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=None,
                use_cache=False
            )
        
        # 应用最终层归一化
        hidden_states = self.base_model.ln_f(hidden_states)
        
        # 通过奖励头部得到奖励值
        rewards = self.reward_head(hidden_states)
        
        if return_dict:
            return {
                "rewards": rewards,
                "hidden_states": hidden_states
            }
        else:
            return rewards
    
    def save_pretrained(self, save_directory: str):
        """保存模型"""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存模型状态
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # 保存配置
        config_dict = {
            'vocab_size': self.config.vocab_size,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'dropout': self.config.dropout,
            'max_position_embeddings': self.config.max_position_embeddings
        }
        
        with open(os.path.join(save_directory, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """加载模型"""
        import os
        import json
        
        # 加载配置
        config_file = os.path.join(model_path, "config.json")
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = RewardModelConfig(**config_dict)
        
        # 创建模型实例
        model = cls(config)
        
        # 加载模型状态
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')
        model.load_state_dict(state_dict)
        
        return model
    
    def predict_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """预测奖励值"""
        self.eval()
        with torch.no_grad():
            try:
                outputs = self.forward(input_ids, attention_mask, return_dict=True)
                rewards = outputs.get("rewards")
                
                if rewards is None:
                    print("[WARNING] 奖励模型返回None，返回默认值0.0")
                    return torch.zeros(input_ids.size(0), device=input_ids.device)
                
                return rewards
            except Exception as e:
                print(f"[WARNING] 奖励预测失败: {e}")
                return torch.zeros(input_ids.size(0), device=input_ids.device)
    
    def compare_responses(
        self,
        chosen_input_ids: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """比较两个响应的奖励值"""
        chosen_rewards = self.predict_reward(chosen_input_ids, chosen_attention_mask)
        rejected_rewards = self.predict_reward(rejected_input_ids, rejected_attention_mask)
        
        return chosen_rewards, rejected_rewards

class PreferenceLoss(nn.Module):
    """偏好损失函数"""
    
    def __init__(self, margin: float = 0.1, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor
    ) -> torch.Tensor:
        """计算偏好损失"""
        
        # 方法1: 对比损失
        diff = chosen_rewards - rejected_rewards
        
        # 使用margin的对比损失
        contrastive_loss = F.relu(self.margin - diff).mean()
        
        # 方法2: 交叉熵损失（可选）
        logits = torch.stack([chosen_rewards, rejected_rewards], dim=-1) / self.temperature
        labels = torch.zeros_like(chosen_rewards, dtype=torch.long)
        ce_loss = F.cross_entropy(logits, labels)
        
        # 组合损失
        total_loss = contrastive_loss + 0.5 * ce_loss
        
        return total_loss

def create_reward_model(
    vocab_size: int,
    hidden_size: int = 1024,
    num_layers: int = 4,
    base_model=None
) -> RewardModel:
    """创建奖励模型"""
    
    config = RewardModelConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
    return RewardModel(config, base_model)

def load_reward_model(model_path: str) -> RewardModel:
    """加载奖励模型"""
    return RewardModel.from_pretrained(model_path)

# 使用示例
if __name__ == "__main__":
    # 创建测试奖励模型
    model = create_reward_model(vocab_size=16384)
    
    # 测试前向传播
    input_ids = torch.randint(0, 16384, (2, 128))
    attention_mask = torch.ones_like(input_ids)
    
    outputs = model(input_ids, attention_mask)
    print(f"奖励值形状: {outputs['rewards'].shape}")
    print(f"奖励值: {outputs['rewards']}")