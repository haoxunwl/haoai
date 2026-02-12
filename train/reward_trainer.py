"""
奖励模型训练器
用于训练能够评估文本质量的奖励模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

from model.reward_model import RewardModel, PreferenceLoss, create_reward_model

class PreferenceDataset(Dataset):
    """偏好数据集"""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_file)
    
    def _load_data(self, data_file: str) -> List[Dict[str, Any]]:
        """加载偏好数据"""
        
        if not os.path.exists(data_file):
            print(f"警告: 偏好数据文件不存在: {data_file}")
            return []
        
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        continue
        
        print(f"加载了 {len(data)} 个偏好样本")
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # 获取prompt和响应
        prompt_text = item.get('prompt', '')
        chosen_text = item.get('chosen', '')
        rejected_text = item.get('rejected', '')
        
        # 组合prompt和响应
        chosen_full = f"{prompt_text} {chosen_text}"
        rejected_full = f"{prompt_text} {rejected_text}"
        
        # 编码文本
        chosen_ids = self.tokenizer.encode(chosen_full)[:self.max_length]
        rejected_ids = self.tokenizer.encode(rejected_full)[:self.max_length]
        
        # 创建注意力掩码
        chosen_mask = [1] * len(chosen_ids) + [0] * (self.max_length - len(chosen_ids))
        rejected_mask = [1] * len(rejected_ids) + [0] * (self.max_length - len(rejected_ids))
        
        # 填充
        chosen_ids = chosen_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(chosen_ids))
        rejected_ids = rejected_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(rejected_ids))
        
        return {
            'chosen_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'rejected_ids': torch.tensor(rejected_ids, dtype=torch.long),
            'chosen_mask': torch.tensor(chosen_mask, dtype=torch.long),
            'rejected_mask': torch.tensor(rejected_mask, dtype=torch.long),
            'chosen_text': chosen_full,
            'rejected_text': rejected_full
        }

class RewardModelTrainer:
    """奖励模型训练器"""
    
    def __init__(
        self,
        reward_model: RewardModel,
        tokenizer,
        config,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # 移动到设备
        self.reward_model.to(device)
        
        # 损失函数
        self.loss_fn = PreferenceLoss(
            margin=config.margin,
            temperature=config.temperature
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.reward_model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # 训练状态
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        
        self.reward_model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # 移动到设备
            chosen_ids = batch['chosen_ids'].to(self.device)
            rejected_ids = batch['rejected_ids'].to(self.device)
            chosen_mask = batch['chosen_mask'].to(self.device)
            rejected_mask = batch['rejected_mask'].to(self.device)
            
            # 前向传播
            chosen_outputs = self.reward_model(chosen_ids, chosen_mask, return_dict=True)
            rejected_outputs = self.reward_model(rejected_ids, rejected_mask, return_dict=True)
            
            chosen_rewards = chosen_outputs['rewards']
            rejected_rewards = rejected_outputs['rewards']
            
            # 计算损失
            loss = self.loss_fn(chosen_rewards, rejected_rewards)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}'
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'epoch': epoch,
            'average_loss': avg_loss,
            'total_batches': num_batches
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        
        self.reward_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # 准确率统计
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                chosen_ids = batch['chosen_ids'].to(self.device)
                rejected_ids = batch['rejected_ids'].to(self.device)
                chosen_mask = batch['chosen_mask'].to(self.device)
                rejected_mask = batch['rejected_mask'].to(self.device)
                
                # 前向传播
                chosen_rewards = self.reward_model.predict_reward(chosen_ids, chosen_mask)
                rejected_rewards = self.reward_model.predict_reward(rejected_ids, rejected_mask)
                
                # 计算损失
                loss = self.loss_fn(chosen_rewards, rejected_rewards)
                total_loss += loss.item()
                
                # 计算准确率
                correct = (chosen_rewards > rejected_rewards).sum().item()
                correct_predictions += correct
                total_predictions += chosen_rewards.size(0)
                
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'average_loss': avg_loss,
            'accuracy': accuracy,
            'total_samples': total_predictions
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        save_dir: str = None
    ) -> Dict[str, Any]:
        """完整训练流程"""
        
        if save_dir is None:
            save_dir = self.config.model_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        training_stats = {
            'train_losses': [],
            'eval_losses': [],
            'accuracies': [],
            'best_loss': float('inf'),
            'best_accuracy': 0.0
        }
        
        print("开始训练奖励模型...")
        
        for epoch in range(1, self.config.epochs + 1):
            # 训练
            train_stats = self.train_epoch(train_dataloader, epoch)
            training_stats['train_losses'].append(train_stats['average_loss'])
            
            # 评估
            if eval_dataloader is not None:
                eval_stats = self.evaluate(eval_dataloader)
                training_stats['eval_losses'].append(eval_stats['average_loss'])
                training_stats['accuracies'].append(eval_stats['accuracy'])
                
                print(f"Epoch {epoch}: "
                      f"Train Loss: {train_stats['average_loss']:.4f}, "
                      f"Eval Loss: {eval_stats['average_loss']:.4f}, "
                      f"Accuracy: {eval_stats['accuracy']:.4f}")
                
                # 保存最佳模型
                if eval_stats['average_loss'] < training_stats['best_loss']:
                    training_stats['best_loss'] = eval_stats['average_loss']
                    training_stats['best_accuracy'] = eval_stats['accuracy']
                    self.save_model(os.path.join(save_dir, "best_model"))
                    print(f"[SUCCESS] 新的最佳模型已保存 (损失: {eval_stats['average_loss']:.4f}, 准确率: {eval_stats['accuracy']:.4f})")
            else:
                print(f"Epoch {epoch}: Train Loss: {train_stats['average_loss']:.4f}")
            
            # 保存检查点
            if epoch % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}")
                self.save_model(checkpoint_path)
                print(f"[SAVE] 检查点已保存: {checkpoint_path}")
        
        # 保存最终模型
        self.save_model(os.path.join(save_dir, "final_model"))
        print("[SUCCESS] 奖励模型训练完成！")
        
        return training_stats
    
    def save_model(self, path: str):
        """保存模型"""
        self.reward_model.save_pretrained(path)
        
        # 保存训练配置
        config_info = {
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'training_config': {
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'epochs': self.config.epochs
            }
        }
        
        config_file = os.path.join(path, "training_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    def load_model(self, path: str):
        """加载模型"""
        self.reward_model = RewardModel.from_pretrained(path)
        self.reward_model.to(self.device)

def create_reward_trainer(
    tokenizer,
    vocab_size: int,
    config,
    base_model=None
) -> RewardModelTrainer:
    """创建奖励模型训练器"""
    
    # 创建奖励模型
    reward_model = create_reward_model(
        vocab_size=vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        base_model=base_model
    )
    
    return RewardModelTrainer(reward_model, tokenizer, config)

def train_reward_model(
    tokenizer,
    data_file: str,
    config,
    save_dir: str = None
) -> RewardModelTrainer:
    """训练奖励模型"""
    
    # 创建数据集
    dataset = PreferenceDataset(data_file, tokenizer, config.max_sequence_length)
    
    if len(dataset) == 0:
        print("[ERROR] 没有找到训练数据，请检查数据文件路径")
        return None
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # 创建训练器
    trainer = create_reward_trainer(tokenizer, tokenizer.vocab_size, config)
    
    # 开始训练
    trainer.train(dataloader, save_dir=save_dir)
    
    return trainer

# 使用示例
if __name__ == "__main__":
    # 测试代码
    from train.config import TokenizerConfig
    from train.rlhf_config import RewardModelConfig
    from train.train_pretrain import BPETokenizer
    
    # 创建测试配置
    config = RewardModelConfig()
    
    # 创建分词器（需要先有分词器文件）
    # tokenizer = BPETokenizer("weight/tokenizer/tokenizer.json")
    
    # 训练奖励模型
    # trainer = train_reward_model(tokenizer, "training_data/rlhf/preference_data.jsonl", config)