"""
智能优化训练脚本 - HaoAI模型
整合了所有性能优化和智能训练策略
修复版本：2025.04.07
- 修正 LambdaLR 导入错误
- 强制词汇表大小一致
- 输入ID边界裁剪
- 混合精度训练修复
- 梯度/损失有效性验证
- 适配 IterableDataset 验证方式
"""

import torch
import torch.nn as nn
import os
import time
import math
import json
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR  # 修正：原为 LambdaHR
import sys

# ---------- 调试建议 ----------
# 如需精确定位CUDA错误，取消下面注释
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
# -----------------------------

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("警告: tqdm库未安装，将使用简单进度显示")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import SmartHaoAI, HaoAIConfig
from train.config import SFTConfig, DataConfig, TokenizerConfig
from train.data_utils import StreamingDataset
from train.train_pretrain import BPETokenizer
from train.optimization_config import get_optimization_config
from train.smart_evaluator import create_smart_evaluator


class SmartTrainer:
    """智能训练器（修复版）"""

    def __init__(self, config_preset="balanced"):
        self.opt_config = get_optimization_config(config_preset)
        self.sft_cfg = SFTConfig()
        self.d_cfg = DataConfig()
        self.t_cfg = TokenizerConfig()

        # 应用优化配置
        self._apply_optimization_config()

        # 训练状态
        self.global_step = 0
        self.best_loss = float('inf')
        self.evaluator = None
        self.loss_history = []

        # 混合精度缩放器（固定实例）
        self.scaler = torch.amp.GradScaler('cuda') if self.opt_config.use_mixed_precision else None

    def _apply_optimization_config(self):
        """应用优化配置"""
        self.sft_cfg.accumulation_steps = self.opt_config.gradient_accumulation_steps
        self.d_cfg.block_size = self.opt_config.max_sequence_length
        print(f"[INFO] 使用优化配置预设: {self.opt_config}")

    # ------------------------------------------------------------------
    # 梯度处理工具
    # ------------------------------------------------------------------
    def adaptive_gradient_clip(self, parameters):
        """自适应梯度裁剪（兼容混合精度）"""
        if self.opt_config.use_mixed_precision and self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        if hasattr(torch.nn.utils, 'clip_grad_norm_'):
            total_norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_norm=self.opt_config.gradient_clip_norm
            )
            return total_norm
        else:
            # 备用方法
            total_norm = 0.0
            for p in parameters:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            clip_coef = self.opt_config.gradient_clip_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in parameters:
                    if p.grad is not None:
                        p.grad.data.mul_(clip_coef)
            return torch.tensor(total_norm)

    def adaptive_lr_adjustment(self, current_lr, grad_norm, monitor):
        """自适应学习率调整"""
        if grad_norm > 1.0:
            new_lr = current_lr * 0.8
        elif grad_norm < 0.01:
            new_lr = current_lr * 1.2
        else:
            new_lr = current_lr

        if len(monitor.loss_window) >= 10:
            recent_loss = sum(monitor.loss_window[-5:]) / 5
            if recent_loss > monitor.best_loss * 1.1:
                new_lr = new_lr * 0.9

        new_lr = max(new_lr, self.opt_config.min_learning_rate)
        new_lr = min(new_lr, self.opt_config.max_learning_rate)
        return new_lr

    def smart_scheduler_step(self, scheduler, loss, grad_norm):
        """智能调度器步骤"""
        if len(self.loss_history) >= 10:
            recent_loss = sum(self.loss_history[-5:]) / 5
            if recent_loss > self.best_loss * 1.05:
                scheduler.step()
            else:
                scheduler.step()
        else:
            scheduler.step()

    # ------------------------------------------------------------------
    # 训练环境设置
    # ------------------------------------------------------------------
    def setup_training(self):
        """设置训练环境"""
        print("\n" + "=" * 50)
        print("开始智能训练HaoAI模型（修复版）")
        print("=" * 50)

        os.makedirs(self.sft_cfg.sft_model_dir, exist_ok=True)

        tokenizer = self._load_tokenizer()
        if tokenizer is None:
            return None

        dataset, dataloader = self._create_dataset(tokenizer)
        model = self._create_model(tokenizer)
        optimizer, scheduler = self._create_optimizer_scheduler(model, len(dataloader))

        self.evaluator = create_smart_evaluator(tokenizer, self.sft_cfg.device)

        return {
            'tokenizer': tokenizer,
            'dataset': dataset,
            'dataloader': dataloader,
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler
        }

    def _load_tokenizer(self):
        """加载分词器"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = script_dir

        tokenizer_file = "weight/tokenizer/tokenizer.json"
        tokenizer_file = os.path.normpath(os.path.join(project_root, tokenizer_file))

        if not os.path.exists(tokenizer_file):
            print(f"分词器文件不存在: {tokenizer_file}")
            possible_paths = [
                "weight/tokenizer/tokenizer.json",
                "weight/tokenizer.json",
                "tokenizer.json"
            ]
            for path in possible_paths:
                abs_path = os.path.normpath(os.path.join(project_root, path))
                if os.path.exists(abs_path):
                    tokenizer_file = abs_path
                    break
            else:
                print("未找到分词器文件，请先运行分词器训练")
                return None

        print(f"加载分词器: {tokenizer_file}")
        tokenizer = BPETokenizer(tokenizer_file)
        return tokenizer

    def _create_dataset(self, tokenizer):
        """创建数据集，并验证样本合法性（适配 IterableDataset）"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = script_dir

        self.d_cfg.pretrain_file = os.path.join(project_root, self.d_cfg.pretrain_file)
        self.d_cfg.sft_file = os.path.join(project_root, self.d_cfg.sft_file)

        dataset = StreamingDataset(self.d_cfg, tokenizer, mode='sft')

        # ---- 验证数据集样本是否合法（不依赖 __getitem__）----
        try:
            sample = next(iter(dataset))
            if 'input_ids' in sample:
                ids = sample['input_ids']
                # 确保ids是torch.Tensor
                if not isinstance(ids, torch.Tensor):
                    ids = torch.tensor(ids)
                vocab_size = tokenizer.vocab_size
                if ids.max() >= vocab_size or ids.min() < 0:
                    raise ValueError(
                        f"数据集包含非法token ID! vocab_size={vocab_size}, "
                        f"min_id={ids.min()}, max_id={ids.max()}"
                    )
                print(f" 数据集验证通过，vocab_size={vocab_size}")
        except StopIteration:
            print("⚠️ 数据集为空，跳过验证")

        # 智能批处理函数
        def smart_collate_fn(batch):
            return self._smart_collate(batch, tokenizer)

        dataloader = DataLoader(
            dataset,
            batch_size=self.sft_cfg.batch_size,
            num_workers=0,
            pin_memory=True,
            collate_fn=smart_collate_fn
        )

        print(f"数据集大小: {len(dataset)}")  # 若为IterableDataset可能不支持len，会抛异常，但StreamingDataset可能实现了__len__
        print(f"批次数量: {len(dataloader)}")
        return dataset, dataloader

    def _smart_collate(self, batch, tokenizer):
        """智能批处理函数，强制裁剪越界token ID"""
        input_ids_list = [item['input_ids'] for item in batch]
        labels_list = [item['labels'] for item in batch]

        # 边界裁剪
        vocab_size = tokenizer.vocab_size
        for i, ids in enumerate(input_ids_list):
            if not isinstance(ids, torch.Tensor):
                ids = torch.tensor(ids)
            if ids.max() >= vocab_size or ids.min() < 0:
                print(f"⚠️ 批次中存在越界token ID: max={ids.max()}, min={ids.min()}, 已裁剪")
                input_ids_list[i] = torch.clamp(ids, 0, vocab_size - 1)

        # 动态最大长度选择
        seq_lengths = [len(seq) for seq in input_ids_list]
        if max(seq_lengths) / min(seq_lengths) < 2.0:
            max_len = min(max(seq_lengths), self.opt_config.max_sequence_length)
        else:
            max_len = min(int(np.percentile(seq_lengths, self.opt_config.sequence_length_percentile)),
                          self.opt_config.max_sequence_length)

        batch_size = len(batch)
        padded_input_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_masks = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, (input_ids, labels) in enumerate(zip(input_ids_list, labels_list)):
            seq_len = min(len(input_ids), max_len)

            padded_input_ids[i, :seq_len] = input_ids[:seq_len]
            padded_labels[i, :seq_len] = labels[:seq_len]
            attention_masks[i, :seq_len] = 1

            # 智能截断保留EOS（可选）
            if seq_len < len(input_ids):
                remaining_tokens = input_ids[seq_len:]
                eos_mask = (remaining_tokens == tokenizer.eos_token_id)
                eos_positions = torch.nonzero(eos_mask, as_tuple=True)[0]
                if len(eos_positions) > 0:
                    eos_pos = eos_positions[0].item() + seq_len
                    if eos_pos - seq_len < 50:
                        new_seq_len = min(eos_pos + 1, max_len)
                        if new_seq_len > seq_len:
                            padded_input_ids[i, seq_len:new_seq_len] = input_ids[seq_len:new_seq_len]
                            padded_labels[i, seq_len:new_seq_len] = labels[seq_len:new_seq_len]
                            attention_masks[i, seq_len:new_seq_len] = 1
                            seq_len = new_seq_len

        return {
            'input_ids': padded_input_ids,
            'labels': padded_labels,
            'attention_mask': attention_masks
        }

    def _create_model(self, tokenizer):
        """创建/加载模型，强制词汇表大小一致"""
        config = HaoAIConfig(
            vocab_size=tokenizer.vocab_size,
            n_layer=8,
            n_head=8,
            n_embd=1024,
            max_position_embeddings=4096,
            dropout=0.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_reasoning=False,
            use_sliding_window=False
        )

        model = None

        # 尝试加载已训练的SFT模型
        if os.path.exists(self.sft_cfg.sft_model_dir):
            try:
                model = SmartHaoAI.from_pretrained(self.sft_cfg.sft_model_dir)
                print(" SFT模型加载成功")
            except Exception as e:
                print(f" SFT模型加载失败: {e}")

        # 尝试加载预训练模型
        if model is None and os.path.exists(self.sft_cfg.pretrain_model_dir):
            try:
                model = SmartHaoAI.from_pretrained(self.sft_cfg.pretrain_model_dir)
                print(" 预训练模型加载成功")
            except Exception as e:
                print(f" 预训练模型加载失败: {e}")

        # 若加载失败，创建新模型
        if model is None:
            print(" 创建新的模型实例")
            model = SmartHaoAI(config)
        else:
            # 检查词汇表一致性并调整
            if model.config.vocab_size != tokenizer.vocab_size:
                print(f" 模型词汇表({model.config.vocab_size})与tokenizer({tokenizer.vocab_size})不一致，强制修正")
                model.config.vocab_size = tokenizer.vocab_size
                # 重新初始化嵌入层和输出层
                model.embed = nn.Embedding(tokenizer.vocab_size, model.config.n_embd)
                model.lm_head = nn.Linear(model.config.n_embd, tokenizer.vocab_size, bias=False)
                # 可选：权重重绑定
                if model.config.tie_word_embeddings:
                    model.lm_head.weight = model.embed.weight
                print("模型词汇表已修正，嵌入层重新初始化")

        model = model.to(self.sft_cfg.device)
        for module in model.modules():
            if hasattr(module, 'to'):
                module.to(self.sft_cfg.device)

        return model

    def _create_optimizer_scheduler(self, model, dataloader_length):
        """创建优化器和调度器"""
        optimizer = AdamW(
            model.parameters(),
            lr=self.sft_cfg.lr,
            weight_decay=0.01,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        self.optimizer = optimizer

        total_steps = dataloader_length * self.sft_cfg.epochs // self.sft_cfg.accumulation_steps

        if self.opt_config.learning_rate_strategy == "adaptive":
            scheduler = self._create_adaptive_scheduler(optimizer, total_steps)
        else:
            scheduler = self._create_cosine_scheduler(optimizer, total_steps)

        print(f"总训练步数: {total_steps}")
        return optimizer, scheduler

    def _create_adaptive_scheduler(self, optimizer, total_steps):
        """自适应调度器"""
        def lr_lambda(step):
            if step < total_steps * self.opt_config.warmup_ratio:
                return self.opt_config.min_lr_ratio + (1 - self.opt_config.min_lr_ratio) * \
                       (step / (total_steps * self.opt_config.warmup_ratio))
            elif step < total_steps * 0.7:
                return 1.0
            else:
                progress = (step - total_steps * 0.7) / (total_steps * 0.3)
                return self.opt_config.min_lr_ratio + (1 - self.opt_config.min_lr_ratio) * \
                       0.5 * (1 + math.cos(math.pi * progress))
        return LambdaLR(optimizer, lr_lambda)

    def _create_cosine_scheduler(self, optimizer, total_steps):
        """余弦调度器"""
        warmup_steps = int(total_steps * self.opt_config.warmup_ratio)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 0.5 * 2.0 * progress)))
        return LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # 训练主循环
    # ------------------------------------------------------------------
    def train(self):
        """执行训练"""
        training_setup = self.setup_training()
        if training_setup is None:
            print("训练设置失败")
            return

        tokenizer = training_setup['tokenizer']
        dataloader = training_setup['dataloader']
        model = training_setup['model']
        optimizer = training_setup['optimizer']
        scheduler = training_setup['scheduler']
        self.optimizer = optimizer

        print("\n开始智能训练...")
        print("=" * 50)

        class SmartTrainingMonitor:
            def __init__(self, opt_config, window_size=100):
                self.opt_config = opt_config
                self.loss_window = []
                self.grad_norm_window = []
                self.lr_window = []
                self.window_size = window_size
                self.best_loss = float('inf')
                self.patience = 0
                self.learning_plateau = 0
                self.gradient_instability = 0

            def update(self, loss, grad_norm=None, lr=None):
                self.loss_window.append(loss)
                if len(self.loss_window) > self.window_size:
                    self.loss_window.pop(0)
                if grad_norm is not None:
                    self.grad_norm_window.append(grad_norm)
                    if len(self.grad_norm_window) > self.window_size:
                        self.grad_norm_window.pop(0)
                if lr is not None:
                    self.lr_window.append(lr)
                    if len(self.lr_window) > self.window_size:
                        self.lr_window.pop(0)

                if len(self.loss_window) >= self.window_size // 2:
                    recent_loss = sum(self.loss_window[-self.window_size//2:]) / (self.window_size//2)
                    if abs(recent_loss - self.best_loss) < 1e-6:
                        self.learning_plateau += 1
                    else:
                        self.learning_plateau = 0
                    if len(self.grad_norm_window) >= 10:
                        recent_grads = self.grad_norm_window[-10:]
                        grad_variance = np.var(recent_grads) if len(recent_grads) > 1 else 0
                        if grad_variance > 100:
                            self.gradient_instability += 1
                        else:
                            self.gradient_instability = 0
                    if recent_loss < self.best_loss:
                        self.best_loss = recent_loss
                        self.patience = 0
                    else:
                        self.patience += 1

            def should_stop(self):
                if self.patience >= self.opt_config.early_stopping_patience * 2:
                    return True
                if self.learning_plateau >= 100 and self.patience >= 50:
                    return True
                return False

            def get_optimization_suggestions(self):
                suggestions = []
                if len(self.loss_window) >= 10:
                    recent_loss = np.mean(self.loss_window[-10:])
                    if recent_loss > self.best_loss * 1.5:
                        suggestions.append("损失上升，建议降低学习率")
                    if len(self.grad_norm_window) >= 10:
                        avg_grad_norm = np.mean(self.grad_norm_window[-10:])
                        if avg_grad_norm > 100:
                            suggestions.append("梯度爆炸，建议减小学习率或增加梯度裁剪")
                        elif avg_grad_norm < 0.1:
                            suggestions.append("梯度消失，建议增加学习率或减少梯度裁剪")
                return suggestions

        monitor = SmartTrainingMonitor(self.opt_config)
        model.train()

        for epoch in range(self.sft_cfg.epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.time()

            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.sft_cfg.epochs}') if TQDM_AVAILABLE else dataloader
            if not TQDM_AVAILABLE:
                print(f"Epoch {epoch+1}/{self.sft_cfg.epochs}:")

            for batch_idx, batch in enumerate(pbar):
                input_ids = batch['input_ids'].to(self.sft_cfg.device)
                labels = batch['labels'].to(self.sft_cfg.device)
                attention_mask = batch['attention_mask'].to(self.sft_cfg.device)

                try:
                    if self.opt_config.use_mixed_precision:
                        with torch.amp.autocast('cuda'):
                            outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                            loss = outputs.loss / self.sft_cfg.accumulation_steps
                    else:
                        outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                        loss = outputs.loss / self.sft_cfg.accumulation_steps
                except RuntimeError as e:
                    print(f"前向传播错误: {e}")
                    print(f"input_ids shape: {input_ids.shape}, max_id: {input_ids.max().item()}, min_id: {input_ids.min().item()}")
                    raise

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"无效损失值: {loss.item():.4f}，跳过此批次")
                    optimizer.zero_grad()
                    continue

                if self.opt_config.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % self.sft_cfg.accumulation_steps == 0:
                    grad_norm = self.adaptive_gradient_clip(model.parameters())

                    if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 1e6:
                        print(f"梯度异常 (norm={grad_norm:.2e})，跳过此更新步")
                        optimizer.zero_grad()
                        if self.opt_config.use_mixed_precision:
                            self.scaler.update()
                        continue

                    current_lr = scheduler.get_last_lr()[0]
                    if self.opt_config.adaptive_learning_rate:
                        new_lr = self.adaptive_lr_adjustment(current_lr, grad_norm, monitor)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr

                    if self.opt_config.use_mixed_precision:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()

                    if self.opt_config.smart_scheduler:
                        self.smart_scheduler_step(scheduler, loss, grad_norm)
                    else:
                        scheduler.step()

                    optimizer.zero_grad()
                    self.global_step += 1

                    current_loss = loss.item() * self.sft_cfg.accumulation_steps
                    monitor.update(current_loss,
                                   grad_norm.item() if grad_norm is not None else None,
                                   scheduler.get_last_lr()[0])
                    self.loss_history.append(current_loss)
                    if len(self.loss_history) > 100:
                        self.loss_history.pop(0)

                    if self.global_step % 50 == 0:
                        smooth_loss = sum(monitor.loss_window) / len(monitor.loss_window) if monitor.loss_window else 0
                        print(f"\n步数: {self.global_step}, 平滑损失: {smooth_loss:.4f}, 梯度范数: {grad_norm:.4f}")

                    if self.global_step % self.opt_config.checkpoint_frequency_late == 0 or \
                       (self.global_step < 1000 and self.global_step % self.opt_config.checkpoint_frequency_early == 0):
                        self._save_checkpoint(model, tokenizer, self.global_step)

                    if monitor.should_stop():
                        print("检测到性能停滞，提前停止训练")
                        break

                epoch_loss += loss.item() * self.sft_cfg.accumulation_steps
                num_batches += 1

                if TQDM_AVAILABLE:
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}, 耗时: {epoch_time:.2f}秒")

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self._save_model(model, tokenizer)
                print(f"新的最佳模型已保存 (损失: {self.best_loss:.4f})")

            if (epoch + 1) % 10 == 0 and self.evaluator:
                metrics = self.evaluator.evaluate_model(model, dataloader, 5)
                self.evaluator.update_metrics_history(metrics, self.global_step)
                progress = self.evaluator.get_training_progress()
                print(f"训练进度分析: {progress}")

            if monitor.should_stop():
                break

        self._save_model(model, tokenizer)
        self._save_training_summary(model.config)
        print("智能训练完成！")

    # ------------------------------------------------------------------
    # 保存与日志
    # ------------------------------------------------------------------
    def _save_checkpoint(self, model, tokenizer, step):
        checkpoint_path = os.path.join(self.sft_cfg.sft_model_dir, f"checkpoint_step_{step}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path, safe_serialization=False)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"检查点已保存至: {checkpoint_path}")

    def _save_model(self, model, tokenizer):
        model.save_pretrained(self.sft_cfg.sft_model_dir, safe_serialization=False)
        tokenizer.save_pretrained(self.sft_cfg.sft_model_dir)
        print(f"模型已保存至: {self.sft_cfg.sft_model_dir}")

    def _save_training_summary(self, config):
        summary = {
            "total_steps": self.global_step,
            "best_loss": self.best_loss,
            "optimization_config": {
                "preset": "balanced",
                "learning_rate_strategy": self.opt_config.learning_rate_strategy,
                "gradient_accumulation_steps": self.opt_config.gradient_accumulation_steps
            },
            "model_config": {
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "vocab_size": config.vocab_size
            }
        }
        summary_file = os.path.join(self.sft_cfg.sft_model_dir, "smart_training_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"训练总结已保存至: {summary_file}")


def main():
    trainer = SmartTrainer(config_preset="balanced")
    trainer.train()


if __name__ == "__main__":
    main()