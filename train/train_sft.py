import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os
import time
from typing import Dict, Any, List

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import SmartHaoAI, HaoAIConfig
from train.config import SFTConfig, DataConfig, TokenizerConfig
from train.data_utils import StreamingDataset
from train.train_pretrain import BPETokenizer

def sft_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    SFT数据批处理函数，处理不同长度的序列
    """
    input_ids_list = [item['input_ids'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    
    # 找到批次中的最大长度
    max_len = max(len(seq) for seq in input_ids_list)
    
    # 填充所有序列到最大长度
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    
    for input_ids, labels in zip(input_ids_list, labels_list):
        # 计算需要填充的长度
        pad_len = max_len - len(input_ids)
        
        # 填充input_ids和labels
        padded_input = torch.cat([
            input_ids,
            torch.full((pad_len,), input_ids[-1].item(), dtype=torch.long)  # 使用最后一个token填充
        ])
        padded_label = torch.cat([
            labels,
            torch.full((pad_len,), -100, dtype=torch.long)  # 填充部分标签设为-100（忽略损失）
        ])
        
        # 创建attention mask
        attention_mask = torch.cat([
            torch.ones(len(input_ids), dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long)
        ])
        
        padded_input_ids.append(padded_input)
        padded_labels.append(padded_label)
        attention_masks.append(attention_mask)
    
    # 堆叠所有张量
    return {
        'input_ids': torch.stack(padded_input_ids),
        'labels': torch.stack(padded_labels),
        'attention_mask': torch.stack(attention_masks)
    }

def sft_train() -> None:
    print("=== SFT训练开始 ===")
    try:
        sft_cfg = SFTConfig()
        d_cfg = DataConfig()
        t_cfg = TokenizerConfig()

        os.makedirs(sft_cfg.sft_model_dir, exist_ok=True)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        print(f"项目根目录: {project_root}")
        
        # 使用BPE分词器
        tokenizer_file = os.path.join(project_root, t_cfg.TOKENIZER_FILE)
        print(f"Tokenizer文件路径: {tokenizer_file}")
        tokenizer = BPETokenizer(tokenizer_file)
        
        d_cfg.pretrain_file = os.path.join(project_root, d_cfg.pretrain_file)
        d_cfg.sft_file = os.path.join(project_root, d_cfg.sft_file)
        print(f"SFT数据文件: {d_cfg.sft_file}")
        
        effective_batch_size = sft_cfg.batch_size * sft_cfg.accumulation_steps
        print("--- 指令微调 SFT 配置 ---")
        print(f"设备: {sft_cfg.device}")
        print(f"有效批量大小: {effective_batch_size}")
        print(f"SFT模型将保存至: {sft_cfg.sft_model_dir}")
        print("-------------------------")

        dataset = StreamingDataset(d_cfg, tokenizer, mode='sft')
        print(f"数据集大小: {len(dataset)}")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=sft_cfg.batch_size, 
            num_workers=0,
            pin_memory=True,
            collate_fn=sft_collate_fn
        )
        
        print(f"数据加载器创建成功，批次数量: {len(dataloader)}")

        config = HaoAIConfig(
            vocab_size=tokenizer.vocab_size,
            n_layer=6,
            n_head=4,
            n_embd=768,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        
        model = SmartHaoAI(config)
        
        if os.path.exists(sft_cfg.pretrain_model_dir):
            print(f"加载预训练模型: {sft_cfg.pretrain_model_dir}")
            model = SmartHaoAI.from_pretrained(sft_cfg.pretrain_model_dir)
        
        model.to(sft_cfg.device)
        
        optimizer = AdamW(model.parameters(), lr=sft_cfg.lr)
        
        total_steps = len(dataloader) * sft_cfg.epochs
        
        model.train()
        global_step = 0
        
        print("开始SFT训练...")
        
        for epoch in range(sft_cfg.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(sft_cfg.device)
                labels = batch['labels'].to(sft_cfg.device)
                attention_mask = batch['attention_mask'].to(sft_cfg.device)
                
                outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs.loss / sft_cfg.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % sft_cfg.accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                
                epoch_loss += loss.item() * sft_cfg.accumulation_steps
                num_batches += 1
                
                if batch_idx % sft_cfg.print_every == 0:
                    current_loss = loss.item() * sft_cfg.accumulation_steps
                    print(f"Epoch {epoch+1}/{sft_cfg.epochs}, Batch {batch_idx}, Loss: {current_loss:.4f}")
                
                if global_step % sft_cfg.save_every == 0:
                    model.save_pretrained(sft_cfg.sft_model_dir, safe_serialization=False)
                    tokenizer.save_pretrained(sft_cfg.sft_model_dir)
                    print(f"模型已保存至: {sft_cfg.sft_model_dir}")
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch+1} 完成, 平均损失: {avg_epoch_loss:.4f}")
        
        model.save_pretrained(sft_cfg.sft_model_dir, safe_serialization=False)
        tokenizer.save_pretrained(sft_cfg.sft_model_dir)
        print(f"最终模型已保存至: {sft_cfg.sft_model_dir}")
    
    except Exception as e:
        print(f"SFT训练过程中发生错误: {e}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()

if __name__ == "__main__":
    sft_train()