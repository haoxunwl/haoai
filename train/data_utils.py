import torch
import os
import json
import itertools
from glob import glob
from torch.utils.data import IterableDataset, get_worker_info
from transformers import PreTrainedTokenizerFast
from typing import List, Dict, Tuple, Iterator, Optional
from collections import deque

from .config import DataConfig

class DialogueFormatter:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.im_start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.im_start_token_id is None or self.im_end_token_id is None:
            raise ValueError("Tokenizer缺少必要的特殊token: <|im_start|> or <|im_end|>")
        self.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    def format_conversation(self, conversations: List[Dict[str, str]]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        full_token_ids = []
        
        all_msg_tokens = []
        for msg in conversations:
            role, content = msg.get("role", ""), msg.get("content", "")
            prefix = f"<|im_start|>{role}\n"
            suffix = "<|im_end|>"
            
            prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            content_tokens = self.tokenizer.encode(content, add_special_tokens=False)
            suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
            
            msg_tokens = {
                "role": role,
                "prefix": prefix_tokens,
                "content": content_tokens,
                "suffix": suffix_tokens
            }
            all_msg_tokens.append(msg_tokens)
            
            full_token_ids.extend(prefix_tokens)
            full_token_ids.extend(content_tokens)
            full_token_ids.extend(suffix_tokens)
        
        if len(full_token_ids) > self.block_size:
            return None
        
        input_ids = torch.tensor(full_token_ids, dtype=torch.long)
        labels = input_ids.clone()
        
        return input_ids, labels

class StreamingDataset(IterableDataset):
    def __init__(self, data_cfg: DataConfig, tokenizer: PreTrainedTokenizerFast, mode: str = "pretrain"):
        self.data_cfg = data_cfg
        self.tokenizer = tokenizer
        self.mode = mode
        self.formatter = DialogueFormatter(tokenizer, data_cfg.block_size)
        
        if mode == "pretrain":
            self.data_file = data_cfg.pretrain_file
        elif mode == "sft":
            self.data_file = data_cfg.sft_file
        else:
            raise ValueError(f"未知模式: {mode}")
        
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"数据文件不存在: {self.data_file}")
        
        # 估计数据集大小（用于进度条显示）
        self._estimated_size = self._estimate_dataset_size()
    
    def _estimate_dataset_size(self) -> int:
        """估计数据集中的样本数量"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            
            # 直接使用实际行数，避免低估数据集大小
            # 实际有效样本会在数据加载时动态计算
            return line_count
        except:
            # 如果无法估计，返回一个默认值
            return 1000
    
    def __len__(self) -> int:
        """返回估计的数据集大小"""
        return self._estimated_size

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            if worker_info is not None:
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                lines = lines[worker_id::num_workers]
            
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    
                    if self.mode == "pretrain":
                        text = data.get("text", "")
                        if not text:
                            continue
                        
                        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                        if len(token_ids) > self.data_cfg.block_size:
                            continue
                        
                        input_ids = torch.tensor(token_ids, dtype=torch.long)
                        labels = input_ids.clone()
                        
                        yield {"input_ids": input_ids, "labels": labels}
                        
                    elif self.mode == "sft":
                        # 支持两种数据格式：
                        # 1. 传统的conversations格式
                        # 2. 简单的prompt-response格式（用于客服数据）
                        conversations = data.get("conversations", [])
                        prompt = data.get("prompt", "")
                        response = data.get("response", "")
                        
                        if conversations:
                            # 处理传统的conversations格式
                            result = self.formatter.format_conversation(conversations)
                            if result is None:
                                continue
                            
                            input_ids, labels = result
                            yield {"input_ids": input_ids, "labels": labels}
                        elif prompt and response:
                            # 处理简单的prompt-response格式（客服数据）
                            # 将prompt和response转换为conversations格式
                            conversations = [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": response}
                            ]
                            
                            result = self.formatter.format_conversation(conversations)
                            if result is None:
                                continue
                            
                            input_ids, labels = result
                            yield {"input_ids": input_ids, "labels": labels}
                        
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    continue