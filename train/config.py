import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class TokenizerConfig:
    FILES_PATTERNS: List[str] = field(default_factory=lambda: [
        "training_data/pretrain/pretrain_data.jsonl",
        "training_data/sft/sft_data.jsonl"
    ])
    CACHE_FILE: str = "training_data/tokenizer_cache.txt"
    VOCAB_SIZE: int = 16384
    SPECIAL_TOKENS: List[str] = field(default_factory=lambda: [
        "<|endoftext|>", 
        "<|im_start|>", 
        "<|im_end|>"
    ])
    TOKENIZER_DIR: str = "weight/tokenizer/"
    TOKENIZER_FILE: str = "weight/tokenizer/tokenizer.json"

@dataclass
class DataConfig:
    pretrain_file: str = "training_data/pretrain/pretrain_data.jsonl"
    sft_file: str = "training_data/sft/customer_service_data.jsonl"
    block_size: int = 512
    text_buffer_size: int = 4000

@dataclass
class TrainingConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    epochs: int = 300  # 从50轮提高到300轮以获得更充分训练
    accumulation_steps: int = 4
    print_every: int = 10
    save_every: int = 50  # 增加保存间隔，减少频繁保存

@dataclass
class PretrainConfig(TrainingConfig):
    batch_size: int = 16
    lr: float = 3e-4
    pretrain_model_dir: str = "weight/haoai_pretrained_model/"
    checkpoint_file: str = "weight/haoai_pretrained_model/pretrain_checkpoint.pt"
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    warmup_steps: int = 1000

@dataclass
class SFTConfig(TrainingConfig):
    batch_size: int = 4
    lr: float = 5e-5  # 提高学习率以加速收敛
    pretrain_model_dir: str = "weight/haoai_pretrained_model/"
    sft_model_dir: str = "weight/haoai_sft_model/"
    checkpoint_file: str = "weight/haoai_sft_model/sft_checkpoint.pt"