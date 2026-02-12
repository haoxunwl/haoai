"""
智能模型评估器
用于在训练过程中监控模型性能和质量
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

class SmartEvaluator:
    """智能模型评估器"""
    
    def __init__(self, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = tokenizer
        self.device = device
        
        # 评估指标
        self.metrics_history = defaultdict(list)
        self.best_metrics = {}
        
        # 测试样本
        self.test_prompts = [
            "你好，",
            "今天天气",
            "人工智能",
            "机器学习",
            "深度学习",
            "自然语言处理",
            "计算机科学",
            "编程语言",
            "神经网络",
            "大数据"
        ]
    
    def evaluate_model(self, model, dataloader, num_batches: int = 10) -> Dict[str, float]:
        """评估模型性能"""
        model.eval()
        
        metrics = {
            'loss': 0.0,
            'perplexity': 0.0,
            'accuracy': 0.0,
            'token_accuracy': 0.0,
            'sequence_count': 0
        }
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss
                logits = outputs.logits
                
                # 计算困惑度
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                perplexity = torch.exp(loss)
                
                # 计算准确率
                predictions = torch.argmax(shift_logits, dim=-1)
                mask = shift_labels != -100
                
                if mask.sum() > 0:
                    accuracy = (predictions[mask] == shift_labels[mask]).float().mean()
                    metrics['accuracy'] += accuracy.item()
                    metrics['token_accuracy'] += accuracy.item()
                
                metrics['loss'] += loss.item()
                metrics['perplexity'] += perplexity.item()
                metrics['sequence_count'] += 1
        
        # 计算平均值
        if metrics['sequence_count'] > 0:
            for key in ['loss', 'perplexity', 'accuracy', 'token_accuracy']:
                metrics[key] /= metrics['sequence_count']
        
        model.train()
        return metrics
    
    def generate_samples(self, model, max_length: int = 50) -> List[Dict[str, str]]:
        """生成测试样本"""
        model.eval()
        samples = []
        
        with torch.no_grad():
            for prompt in self.test_prompts:
                # 编码输入
                input_ids = self.tokenizer.encode(prompt)
                input_tensor = torch.tensor([input_ids], device=self.device)
                
                # 生成文本
                generated = model.generate(
                    input_tensor,
                    max_length=len(input_ids) + max_length,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # 解码输出
                generated_text = self.tokenizer.decode(generated[0].tolist())
                
                samples.append({
                    'prompt': prompt,
                    'generated': generated_text,
                    'length': len(generated[0])
                })
        
        model.train()
        return samples
    
    def calculate_quality_score(self, samples: List[Dict[str, str]]) -> float:
        """计算生成文本质量分数"""
        if not samples:
            return 0.0
        
        scores = []
        
        for sample in samples:
            prompt = sample['prompt']
            generated = sample['generated']
            
            # 1. 相关性评分（生成文本是否与提示相关）
            relevance_score = self._calculate_relevance(prompt, generated)
            
            # 2. 流畅性评分（基于文本长度和重复性）
            fluency_score = self._calculate_fluency(generated)
            
            # 3. 多样性评分（避免重复模式）
            diversity_score = self._calculate_diversity(generated)
            
            # 综合评分
            quality_score = (relevance_score + fluency_score + diversity_score) / 3
            scores.append(quality_score)
        
        return sum(scores) / len(scores)
    
    def _calculate_relevance(self, prompt: str, generated: str) -> float:
        """计算生成文本与提示的相关性"""
        # 简单的关键词匹配
        prompt_words = set(prompt.lower().split())
        generated_words = set(generated.lower().split())
        
        if not prompt_words:
            return 0.5  # 默认分数
        
        overlap = len(prompt_words.intersection(generated_words))
        return min(overlap / len(prompt_words), 1.0)
    
    def _calculate_fluency(self, text: str) -> float:
        """计算文本流畅性"""
        # 基于句子长度和重复性
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.3
        
        # 检查句子长度分布
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        if not sentence_lengths:
            return 0.3
        
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        
        # 理想句子长度在5-20个词之间
        if 5 <= avg_length <= 20:
            length_score = 1.0
        else:
            length_score = max(0.1, 1.0 - abs(avg_length - 12.5) / 12.5)
        
        # 检查重复性
        words = text.lower().split()
        if len(words) < 5:
            return length_score * 0.5
        
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)
        
        return (length_score + diversity_ratio) / 2
    
    def _calculate_diversity(self, text: str) -> float:
        """计算文本多样性"""
        words = text.lower().split()
        if len(words) < 3:
            return 0.1
        
        # 计算词汇多样性
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)
        
        # 检查重复模式
        if len(words) > 10:
            # 检查是否有连续重复的词
            consecutive_repeats = 0
            for i in range(1, len(words)):
                if words[i] == words[i-1]:
                    consecutive_repeats += 1
            
            repeat_penalty = max(0, 1.0 - consecutive_repeats / len(words) * 10)
            diversity_ratio *= repeat_penalty
        
        return min(diversity_ratio, 1.0)
    
    def update_metrics_history(self, metrics: Dict[str, float], global_step: int):
        """更新指标历史"""
        for key, value in metrics.items():
            self.metrics_history[key].append((global_step, value))
            
            # 更新最佳指标
            if key not in self.best_metrics or value < self.best_metrics[key]:
                self.best_metrics[key] = value
    
    def get_training_progress(self) -> Dict[str, Any]:
        """获取训练进度分析"""
        if not self.metrics_history.get('loss'):
            return {}
        
        recent_losses = [x[1] for x in self.metrics_history['loss'][-20:]]
        
        progress = {
            'current_loss': recent_losses[-1] if recent_losses else 0,
            'loss_trend': self._calculate_trend(recent_losses),
            'best_loss': self.best_metrics.get('loss', float('inf')),
            'improvement_rate': self._calculate_improvement_rate(),
            'training_stability': self._calculate_stability()
        }
        
        return progress
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 3:
            return "unknown"
        
        recent = values[-5:] if len(values) >= 5 else values
        if len(recent) < 2:
            return "unknown"
        
        # 计算斜率
        x = list(range(len(recent)))
        y = recent
        
        # 简单线性回归
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope < -0.01:
            return "improving"
        elif slope > 0.01:
            return "worsening"
        else:
            return "stable"
    
    def _calculate_improvement_rate(self) -> float:
        """计算改进速率"""
        losses = [x[1] for x in self.metrics_history['loss']]
        if len(losses) < 2:
            return 0.0
        
        # 计算平均改进速率
        improvements = []
        for i in range(1, len(losses)):
            if losses[i-1] > 0:
                improvement = (losses[i-1] - losses[i]) / losses[i-1]
                improvements.append(improvement)
        
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def _calculate_stability(self) -> float:
        """计算训练稳定性"""
        losses = [x[1] for x in self.metrics_history['loss']]
        if len(losses) < 3:
            return 1.0
        
        # 计算损失的标准差
        mean_loss = sum(losses) / len(losses)
        variance = sum((x - mean_loss) ** 2 for x in losses) / len(losses)
        std_dev = variance ** 0.5
        
        # 稳定性分数（标准差越小越稳定）
        if mean_loss == 0:
            return 1.0
        
        stability = 1.0 - min(std_dev / mean_loss, 1.0)
        return max(stability, 0.0)

# 使用示例
def create_smart_evaluator(tokenizer, device=None):
    """创建智能评估器"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return SmartEvaluator(tokenizer, device)