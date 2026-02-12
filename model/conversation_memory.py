"""
多轮对话记忆模块 - 让模型记住对话历史
"""

import torch
import torch.nn as nn
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

class ConversationMemory:
    """对话记忆管理器"""
    
    def __init__(self, max_memory_size: int = 100, memory_compression: bool = True):
        self.max_memory_size = max_memory_size
        self.memory_compression = memory_compression
        
        # 对话历史存储
        self.conversation_history = deque(maxlen=max_memory_size)
        
        # 记忆重要性评分
        self.importance_scores = {}
        
        # 主题聚类
        self.topic_clusters = {}
        
        # 记忆压缩器
        if memory_compression:
            self.compressor = MemoryCompressor()
    
    def add_conversation(self, user_input: str, model_response: str, 
                        timestamp: Optional[float] = None, 
                        importance: float = 0.5) -> int:
        """添加对话到记忆"""
        
        if timestamp is None:
            import time
            timestamp = time.time()
        
        conversation_id = len(self.conversation_history)
        
        conversation = {
            'id': conversation_id,
            'user_input': user_input,
            'model_response': model_response,
            'timestamp': timestamp,
            'importance': importance,
            'topic': self._extract_topic(user_input + model_response)
        }
        
        # 添加到历史
        self.conversation_history.append(conversation)
        
        # 更新重要性评分
        self.importance_scores[conversation_id] = importance
        
        # 更新主题聚类
        self._update_topic_clusters(conversation)
        
        return conversation_id
    
    def get_recent_conversations(self, n: int = 5) -> List[Dict[str, Any]]:
        """获取最近的对话"""
        return list(self.conversation_history)[-n:]
    
    def get_important_conversations(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """获取重要的对话"""
        important_conversations = []
        for conv in self.conversation_history:
            if self.importance_scores.get(conv['id'], 0) >= threshold:
                important_conversations.append(conv)
        return important_conversations
    
    def get_topic_conversations(self, topic: str) -> List[Dict[str, Any]]:
        """获取特定主题的对话"""
        if topic in self.topic_clusters:
            conversation_ids = self.topic_clusters[topic]
            return [conv for conv in self.conversation_history if conv['id'] in conversation_ids]
        return []
    
    def update_importance(self, conversation_id: int, new_importance: float):
        """更新对话重要性"""
        if conversation_id in self.importance_scores:
            self.importance_scores[conversation_id] = new_importance
    
    def compress_memory(self) -> str:
        """压缩记忆"""
        if self.memory_compression:
            return self.compressor.compress(list(self.conversation_history))
        else:
            return json.dumps(list(self.conversation_history))
    
    def _extract_topic(self, text: str) -> str:
        """提取对话主题"""
        # 简单的关键词提取
        keywords = ['人工智能', '机器学习', '深度学习', '编程', '技术', 
                   '科学', '数学', '语言', '文化', '历史']
        
        for keyword in keywords:
            if keyword in text:
                return keyword
        
        return '其他'
    
    def _update_topic_clusters(self, conversation: Dict[str, Any]):
        """更新主题聚类"""
        topic = conversation['topic']
        if topic not in self.topic_clusters:
            self.topic_clusters[topic] = []
        
        self.topic_clusters[topic].append(conversation['id'])

class MemoryCompressor:
    """记忆压缩器"""
    
    def __init__(self):
        self.compression_ratio = 0.7
    
    def compress(self, conversations: List[Dict[str, Any]]) -> str:
        """压缩对话记忆"""
        compressed_conversations = []
        
        for conv in conversations:
            # 压缩对话内容
            compressed_conv = {
                'id': conv['id'],
                'user_input': self._compress_text(conv['user_input']),
                'model_response': self._compress_text(conv['model_response']),
                'timestamp': conv['timestamp'],
                'importance': conv['importance'],
                'topic': conv['topic']
            }
            compressed_conversations.append(compressed_conv)
        
        return json.dumps(compressed_conversations)
    
    def _compress_text(self, text: str) -> str:
        """压缩文本"""
        # 简单的文本压缩：保留关键信息
        words = text.split()
        if len(words) > 20:  # 如果文本过长，进行压缩
            # 保留前10个和后10个词
            compressed_words = words[:10] + ['...'] + words[-10:]
            return ' '.join(compressed_words)
        return text

class ContextAwareMemory(nn.Module):
    """上下文感知记忆模块"""
    
    def __init__(self, hidden_size: int, memory_size: int = 100):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        # 记忆存储
        self.memory_keys = nn.Parameter(torch.zeros(memory_size, hidden_size))
        self.memory_values = nn.Parameter(torch.zeros(memory_size, hidden_size))
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 记忆门控
        self.memory_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # 记忆指针
        self.current_memory_index = 0
    
    def forward(self, hidden_states: torch.Tensor, 
                query: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        if query is None:
            query = hidden_states
        
        # 从记忆库中检索相关信息
        memory_scores = torch.matmul(query, self.memory_keys.T) / (hidden_size ** 0.5)
        memory_weights = F.softmax(memory_scores, dim=-1)
        
        # 加权记忆值
        retrieved_memory = torch.matmul(memory_weights, self.memory_values)
        
        # 门控融合
        gate_weights = self.memory_gate(torch.cat([hidden_states, retrieved_memory], dim=-1))
        enhanced_states = gate_weights * retrieved_memory + (1 - gate_weights) * hidden_states
        
        return enhanced_states, memory_weights
    
    def update_memory(self, new_key: torch.Tensor, new_value: torch.Tensor):
        """更新记忆"""
        # 使用最近最少使用策略更新记忆
        with torch.no_grad():
            self.memory_keys[self.current_memory_index] = new_key
            self.memory_values[self.current_memory_index] = new_value
            
            # 更新指针
            self.current_memory_index = (self.current_memory_index + 1) % self.memory_size

class MultiTurnDialogueModel:
    """多轮对话模型"""
    
    def __init__(self, base_model, tokenizer, max_context_length: int = 2048):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        
        # 对话记忆
        self.conversation_memory = ConversationMemory()
        
        # 上下文感知记忆
        if hasattr(base_model, 'config'):
            hidden_size = base_model.config.n_embd
            self.context_memory = ContextAwareMemory(hidden_size)
        
        # 对话历史
        self.dialogue_history = []
    
    def generate_response(self, user_input: str, 
                         context: Optional[str] = None,
                         use_memory: bool = True) -> Dict[str, Any]:
        """生成响应"""
        
        # 构建对话上下文
        full_context = self._build_context(user_input, context, use_memory)
        
        # 编码输入
        inputs = self.tokenizer(full_context, return_tensors='pt', 
                               max_length=self.max_context_length, 
                               truncation=True, padding=True)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码响应
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取模型的实际响应（去除上下文）
        model_response = response.replace(full_context, '').strip()
        
        # 更新对话历史
        self._update_dialogue_history(user_input, model_response)
        
        # 更新记忆
        if use_memory:
            self._update_memory(user_input, model_response)
        
        return {
            'response': model_response,
            'full_context': full_context,
            'memory_used': use_memory,
            'conversation_id': len(self.dialogue_history) - 1
        }
    
    def _build_context(self, user_input: str, context: Optional[str], 
                      use_memory: bool) -> str:
        """构建对话上下文"""
        
        context_parts = []
        
        # 添加系统提示
        system_prompt = "你是一个智能助手，能够进行多轮对话并记住对话历史。"
        context_parts.append(system_prompt)
        
        # 添加上下文
        if context:
            context_parts.append(f"上下文: {context}")
        
        # 添加记忆中的相关对话
        if use_memory:
            recent_conversations = self.conversation_memory.get_recent_conversations(3)
            if recent_conversations:
                context_parts.append("最近的对话:")
                for conv in recent_conversations:
                    context_parts.append(f"用户: {conv['user_input']}")
                    context_parts.append(f"助手: {conv['model_response']}")
        
        # 添加当前用户输入
        context_parts.append(f"用户: {user_input}")
        context_parts.append("助手:")
        
        return "\n".join(context_parts)
    
    def _update_dialogue_history(self, user_input: str, model_response: str):
        """更新对话历史"""
        dialogue_turn = {
            'user_input': user_input,
            'model_response': model_response,
            'timestamp': time.time() if 'time' in globals() else 0
        }
        self.dialogue_history.append(dialogue_turn)
    
    def _update_memory(self, user_input: str, model_response: str):
        """更新记忆"""
        # 计算对话重要性
        importance = self._calculate_importance(user_input, model_response)
        
        # 添加到对话记忆
        self.conversation_memory.add_conversation(
            user_input, model_response, importance=importance
        )
    
    def _calculate_importance(self, user_input: str, model_response: str) -> float:
        """计算对话重要性"""
        # 基于对话长度的简单重要性计算
        input_length = len(user_input)
        response_length = len(model_response)
        
        total_length = input_length + response_length
        
        # 长度越长，重要性越高（但有限制）
        importance = min(total_length / 500, 1.0)
        
        # 如果包含特定关键词，增加重要性
        important_keywords = ['重要', '关键', '记住', '下次', '以后']
        for keyword in important_keywords:
            if keyword in user_input or keyword in model_response:
                importance = min(importance + 0.3, 1.0)
        
        return importance
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """获取对话摘要"""
        return {
            'total_turns': len(self.dialogue_history),
            'recent_conversations': self.conversation_memory.get_recent_conversations(5),
            'important_conversations': self.conversation_memory.get_important_conversations(0.7),
            'topics': list(self.conversation_memory.topic_clusters.keys())
        }
    
    def clear_memory(self):
        """清空记忆"""
        self.conversation_memory = ConversationMemory()
        self.dialogue_history = []

# 使用示例
def create_multi_turn_model(base_model, tokenizer):
    """创建多轮对话模型"""
    return MultiTurnDialogueModel(base_model, tokenizer)