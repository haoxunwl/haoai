import os
import sys
import math
import json
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import SmartHaoAI, HaoAIConfig
from train.config import PretrainConfig, DataConfig, TokenizerConfig
from train.data_utils import StreamingDataset


# ============================
# Tokenizer Wrapper
# ============================

class BPETokenizer:
    def __init__(self, tokenizer_file):
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.vocab_size = self.tokenizer.get_vocab_size()

        # 使用实际的特殊token
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|endoftext|>"
        self.bos_token = "<|endoftext|>"

        # 安全地获取token ID
        self.eos_token_id = self.tokenizer.token_to_id(self.eos_token) or 0
        self.pad_token_id = self.eos_token_id
        self.bos_token_id = self.eos_token_id

    def encode(self, text, add_special_tokens=True):
        # 对于自定义tokenizer，add_special_tokens参数被忽略
        # 因为我们的tokenizer默认不添加特殊token
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def convert_tokens_to_ids(self, token):
        """兼容Hugging Face tokenizer接口"""
        return self.tokenizer.token_to_id(token)

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
                "bos_token_id": self.bos_token_id
            }, f, ensure_ascii=False, indent=2)


# ============================
# Collate Function
# ============================

def create_collate_fn(tokenizer):
    def collate_fn(batch):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]

        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )

        labels = input_ids.clone()
        labels[input_ids == tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "labels": labels
        }
    return collate_fn


# ============================
# Pretrain Entry
# ============================

def pretrain():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pretrain_cfg = PretrainConfig()
    data_cfg = DataConfig()
    tok_cfg = TokenizerConfig()

    device = "cuda"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    tokenizer = BPETokenizer(
        os.path.join(project_root, tok_cfg.TOKENIZER_FILE)
    )

    dataset = StreamingDataset(data_cfg, tokenizer, mode="pretrain")

    dataloader = DataLoader(
        dataset,
        batch_size=pretrain_cfg.batch_size,
        collate_fn=create_collate_fn(tokenizer),
        pin_memory=True,
        num_workers=0
    )

    model = SmartHaoAI(
        HaoAIConfig(
            vocab_size=tokenizer.vocab_size,
            n_layer=12,
            n_head=12,
            n_embd=768,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    ).to(device)

    # 3090 Ti 核心优化
    model.gradient_checkpointing_enable()

    optimizer = AdamW(
        model.parameters(),
        lr=pretrain_cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    scaler = torch.cuda.amp.GradScaler()

    MAX_BATCHES_PER_EPOCH = 1000
    total_steps = MAX_BATCHES_PER_EPOCH * pretrain_cfg.epochs
    warmup_steps = min(pretrain_cfg.warmup_steps * 3, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.7 * (1 + math.cos(math.pi * progress)) + 0.3
        return cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    global_step = 0

    for epoch in range(pretrain_cfg.epochs):
        progress = tqdm(
            enumerate(dataloader),
            total=MAX_BATCHES_PER_EPOCH,
            desc=f"Epoch {epoch+1}/{pretrain_cfg.epochs}"
        )

        for step, batch in progress:
            if step >= MAX_BATCHES_PER_EPOCH:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                loss = model(input_ids=input_ids, labels=labels).loss
                loss = loss / pretrain_cfg.accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % pretrain_cfg.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            progress.set_postfix({
                "loss": f"{loss.item() * pretrain_cfg.accumulation_steps:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "step": global_step
            })

        if (epoch + 1) % 1 == 0:
            model.save_pretrained(pretrain_cfg.pretrain_model_dir)
            tokenizer.save_pretrained(pretrain_cfg.pretrain_model_dir)

    model.save_pretrained(pretrain_cfg.pretrain_model_dir)
    tokenizer.save_pretrained(pretrain_cfg.pretrain_model_dir)


if __name__ == "__main__":
    pretrain()
