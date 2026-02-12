import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


# =========================
# Config
# =========================
class HaoAIConfig(PretrainedConfig):
    model_type = "haoai"

    def __init__(
        self,
        vocab_size=16384,
        n_layer=8,
        n_head=8,
        n_embd=1024,
        dropout=0.1,
        max_position_embeddings=2048,
        use_cache=True,
        tie_word_embeddings=False,
        use_reasoning=True,
        use_sliding_window=True,
        window_size=512,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_reasoning = use_reasoning
        self.use_sliding_window = use_sliding_window
        self.window_size = window_size


class SlidingWindowAttention(nn.Module):
    def __init__(self, config: HaoAIConfig, window_size: int = 512):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.window_size = window_size

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / math.sqrt(self.head_dim)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past_len = past_key_value[0].size(2) if past_key_value else 0
        cos, sin = self.rope(T, past_len, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        causal_mask = torch.tril(
            torch.ones(T, k.size(2), device=x.device)
        ).unsqueeze(0).unsqueeze(0)

        sliding_mask = torch.triu(
            torch.ones(T, k.size(2), device=x.device),
            diagonal=-self.window_size
        ).unsqueeze(0).unsqueeze(0)

        combined_mask = causal_mask * sliding_mask
        attn = attn.masked_fill(combined_mask == 0, torch.finfo(attn.dtype).min)

        if attention_mask is not None:
            # 确保attention mask的形状正确
            # attn shape: (B, n_head, T, T)
            # attention_mask shape: (B, T) → 需要扩展为 (B, 1, T, T)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, self.n_head, -1, -1)
            attention_mask = attention_mask.to(dtype=attn.dtype)
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.o_proj(out)

        present = (k, v) if use_cache else None
        return out, present


class SmartAttention(nn.Module):
    def __init__(self, config: HaoAIConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / math.sqrt(self.head_dim)

        self.rope = RotaryEmbedding(self.head_dim)
        
        self.gate = nn.Linear(config.n_embd, config.n_embd)

    def forward(
        self,
        x,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past_len = past_key_value[0].size(2) if past_key_value else 0
        cos, sin = self.rope(T, past_len, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        causal_mask = torch.tril(
            torch.ones(T, k.size(2), device=x.device)
        ).unsqueeze(0).unsqueeze(0)

        attn = attn.masked_fill(causal_mask == 0, torch.finfo(attn.dtype).min)

        if attention_mask is not None:
            # 确保attention mask的形状正确
            # attn shape: (B, n_head, T, T)
            # attention_mask shape: (B, T) → 需要扩展为 (B, 1, T, T)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, self.n_head, -1, -1)
            attention_mask = attention_mask.to(dtype=attn.dtype)
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        
        gate_weight = torch.sigmoid(self.gate(x))
        out = gate_weight * out + (1 - gate_weight) * x
        
        out = self.o_proj(out)

        present = (k, v) if use_cache else None
        return out, present





# =========================
# Rotary Embedding
# =========================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        # 延迟初始化inv_freq，直到第一次调用forward方法
        self.dim = dim
        self.base = base
        self.inv_freq = None

    def forward(self, seq_len, offset=0, device=None, dtype=None):
        # 延迟初始化inv_freq
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        
        # 确保inv_freq在正确的设备上
        if device is not None and self.inv_freq.device != device:
            self.inv_freq = self.inv_freq.to(device)
            
        t = torch.arange(offset, offset + seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    return torch.cat([-x[..., x.size(-1)//2:], x[..., :x.size(-1)//2]], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# =========================
# Attention
# =========================
class SmartAttention(nn.Module):
    def __init__(self, config: HaoAIConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / math.sqrt(self.head_dim)

        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
    ):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past_len = past_key_value[0].size(2) if past_key_value else 0
        cos, sin = self.rope(T, past_len, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        causal_mask = torch.tril(
            torch.ones(T, k.size(2), device=x.device)
        ).unsqueeze(0).unsqueeze(0)

        attn = attn.masked_fill(causal_mask == 0, torch.finfo(attn.dtype).min)

        if attention_mask is not None:
            # 确保attention mask的形状正确
            # attn shape: (B, n_head, T, T)
            # attention_mask shape: (B, T) → 需要扩展为 (B, 1, T, T)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, self.n_head, -1, -1)
            attention_mask = attention_mask.to(dtype=attn.dtype)
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.o_proj(out)

        present = (k, v) if use_cache else None
        return out, present


# =========================
# MLP
# =========================
class FeedForward(nn.Module):
    def __init__(self, config: HaoAIConfig):
        super().__init__()
        hidden = 4 * config.n_embd
        self.fc1 = nn.Linear(config.n_embd, hidden)
        self.fc2 = nn.Linear(hidden, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.silu(self.fc1(x))))


# =========================
# Block
# =========================
class TransformerBlock(nn.Module):
    def __init__(self, config, use_sliding_window=False, window_size=512):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        if use_sliding_window:
            self.attn = SlidingWindowAttention(config, window_size)
        else:
            self.attn = SmartAttention(config)
            
        self.mlp = FeedForward(config)

    def forward(self, x, attention_mask=None, past_key_value=None, use_cache=False):
        a, present = self.attn(
            self.ln1(x),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, present


# =========================
# Model
# =========================
class SmartHaoAI(PreTrainedModel):
    config_class = HaoAIConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

        self.post_init()
        
        self.use_reasoning = getattr(config, 'use_reasoning', False)
        self.use_sliding_window = getattr(config, 'use_sliding_window', False)
        self.window_size = getattr(config, 'window_size', 512)
        
        if self.use_reasoning:
            from model.reasoning_module import ReasoningModule
            self.reasoning_module = ReasoningModule(config.n_embd)
        
        self.conversation_memory = None
        self.memory_states = {}

    def set_conversation_memory(self, memory_module):
        self.conversation_memory = memory_module

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        reasoning_context=None,
        **kwargs
    ):
        B, T = input_ids.shape
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        x = self.embed(input_ids)
        presents = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values else None
            x, present = block(x, attention_mask, past, use_cache)
            if use_cache:
                presents.append(present)

        if self.use_reasoning and self.reasoning_module is not None:
            try:
                # 检查reasoning_context是否为None
                if reasoning_context is None:
                    pass
                else:
                    # 确保reasoning context的序列长度与当前输入匹配
                    reasoning_context = reasoning_context[:, :x.size(1), :]
                    x, reasoning_scores = self.reasoning_module(x, reasoning_context)
            except Exception as e:
                pass
        
        if self.conversation_memory is not None and self.training is False:
            x, memory_weights = self.conversation_memory(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=presents,
        )

    def generate(
        self,
        input_ids,
        max_length=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=None,
        eos_token_id=None,
        **kwargs
    ):
        return self.generate_with_reasoning(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

    def generate_with_reasoning(
        self,
        input_ids,
        max_length=256,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=None,
        eos_token_id=None,
        reasoning_steps=3,
        **kwargs
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        generated = input_ids.clone()
        past_key_values = None
        reasoning_context = None

        with torch.no_grad():
            for step in range(max_length):
                if past_key_values is None:
                    model_inputs = {"input_ids": generated}
                else:
                    model_inputs = {
                        "input_ids": generated[:, -1:],
                        "past_key_values": past_key_values,
                        "use_cache": True
                    }
                
                if self.use_reasoning and reasoning_context is not None:
                    model_inputs["reasoning_context"] = reasoning_context

                outputs = self(**model_inputs)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                if self.use_reasoning:
                    reasoning_context = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None

                logits = logits / temperature
                
                if repetition_penalty > 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            logits[i, token_id] /= repetition_penalty

                if do_sample:
                    logits = self._top_k_top_p_filtering(logits, top_k, top_p)
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

                if (next_token == eos_token_id).all():
                    break

        return generated

    def _top_k_top_p_filtering(self, logits, top_k, top_p):
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1].unsqueeze(1)
            logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value
