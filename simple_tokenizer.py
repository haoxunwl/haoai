# 独立的分词器模块
from tokenizers import Tokenizer
import os
import json

class SimpleBPETokenizer:
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

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

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