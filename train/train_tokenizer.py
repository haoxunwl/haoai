from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import json
import os
from glob import glob
from typing import List

from .config import TokenizerConfig

def train_tokenizer():
    cfg = TokenizerConfig()
    
    os.makedirs(cfg.TOKENIZER_DIR, exist_ok=True)
    
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=cfg.VOCAB_SIZE,
        special_tokens=cfg.SPECIAL_TOKENS,
        min_frequency=2
    )
    
    text_files = []
    for pattern in cfg.FILES_PATTERNS:
        if os.path.exists(pattern):
            if pattern.endswith('.jsonl'):
                text_files.append(pattern)
            else:
                text_files.extend(glob(pattern))
    
    if not text_files:
        print("未找到训练文件，请检查文件路径")
        return
    
    print(f"找到 {len(text_files)} 个训练文件")
    
    def read_jsonl_files():
        for file_path in text_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'text' in data:
                            yield data['text']
                        elif 'conversations' in data:
                            for conv in data['conversations']:
                                if 'content' in conv:
                                    yield conv['content']
                    except json.JSONDecodeError:
                        continue
    
    def batch_iterator(batch_size=1000):
        batch = []
        for text in read_jsonl_files():
            batch.append(text)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    
    print("开始训练分词器...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # 简化的后处理器，只添加基本的特殊token
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[
            ("<|im_start|>", tokenizer.token_to_id("<|im_start|>")),
            ("<|im_end|>", tokenizer.token_to_id("<|im_end|>")),
        ]
    )
    
    tokenizer.save(cfg.TOKENIZER_FILE)
    print(f"分词器训练完成，已保存至: {cfg.TOKENIZER_FILE}")
    
    vocab_size = tokenizer.get_vocab_size()
    print(f"词汇表大小: {vocab_size}")

if __name__ == "__main__":
    train_tokenizer()