#!/usr/bin/env python3
"""
HaoAI对话界面
使用SmartHaoAI模型进行对话
"""

import torch
import os
import sys
from typing import List, Dict, Any
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import SmartHaoAI, HaoAIConfig
from tokenizers import Tokenizer
from model.conversation_memory import ConversationMemory, MultiTurnDialogueModel
from model.reasoning_module import MultiStepReasoning
from model.smart_prompt import SmartPromptEngineer, create_smart_prompt_engineer

class BPETokenizer:
    def __init__(self, tokenizer_file):
        self.tokenizer = Tokenizer.from_file(tokenizer_file)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.eos_token_id = self.tokenizer.token_to_id("")
        self.pad_token_id = self.eos_token_id
        self.eos_token = ""
        self.pad_token = self.eos_token
        self.bos_token = self.eos_token
        self.bos_token_id = self.eos_token_id
        
    def encode(self, text, add_special_tokens=False):
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

def format_conversation(messages: List[Dict[str, str]], include_memory: bool = False, memory_context: str = "") -> str:
    formatted = ""
    
    if include_memory and memory_context:
        formatted += f"历史对话:\n{memory_context}\n"
    
    for msg in messages:
        formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return formatted

class EnhancedChatSystem:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.conversation_memory = ConversationMemory(max_memory_size=100)
        self.multi_turn_model = MultiTurnDialogueModel(model, tokenizer)
        self.prompt_engineer = create_smart_prompt_engineer()
        
        if hasattr(model, 'use_reasoning') and model.use_reasoning:
            self.reasoning_engine = MultiStepReasoning(model, tokenizer, max_steps=5)
        else:
            self.reasoning_engine = None
        
        self.conversation_history = []
        self.use_memory = True
        self.use_reasoning = False
        self.streaming_mode = False
        self.auto_template = True
        
    def generate_response(self, user_input: str) -> str:
        if self.use_reasoning and self.reasoning_engine:
            return self._generate_with_reasoning(user_input)
        elif self.streaming_mode:
            return self._generate_streaming(user_input)
        else:
            return self._generate_normal(user_input)
    
    def _generate_normal(self, user_input: str) -> str:
        if self.auto_template:
            self.prompt_engineer.auto_select_template(user_input)
        
        context = self._build_context(user_input)
        optimized_prompt = self.prompt_engineer.optimize_prompt(user_input, context)
        
        input_ids = self.tokenizer.encode(optimized_prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_tensor)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_tensor,
                attention_mask=attention_mask,
                max_length=len(input_ids) + 200,
                do_sample=True,
                temperature=0.3,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][len(input_ids):]
        response = self.tokenizer.decode(generated_ids.tolist())
        
        if not response:
            response = "我收到了您的消息，但需要更多上下文来生成有意义的回复。"
        
        self._update_history(user_input, response)
        return response
    
    def _generate_with_reasoning(self, user_input: str) -> str:
        thoughts = []
        current_prompt = user_input
        
        for i in range(3):
            thought_prompt = f"思考步骤 {i+1}: {current_prompt}"
            thought = self._single_generate(thought_prompt, max_new_tokens=128)
            thoughts.append(thought)
            
            if any(keyword in thought for keyword in ["结论", "答案", "结果"]):
                break
            
            current_prompt = f"{current_prompt}\n思考: {thought}\n继续思考..."
        
        final_prompt = f"{user_input}\n思考过程:\n" + "\n".join(thoughts) + "\n最终回答:"
        final_response = self._single_generate(final_prompt, max_new_tokens=256)
        
        self._update_history(user_input, final_response)
        return final_response
    
    def _generate_streaming(self, user_input: str) -> str:
        context = self._build_context(user_input)
        input_ids = self.tokenizer.encode(context, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        generated_ids = input_tensor.clone()
        response_tokens = []
        
        print(" ", end="", flush=True)
        
        with torch.no_grad():
            for _ in range(200):
                outputs = self.model(
                    generated_ids,
                    use_cache=True,
                    past_key_values=None if len(response_tokens) == 0 else outputs.past_key_values
                )
                
                logits = outputs.logits[:, -1, :] / 0.8
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                response_tokens.append(next_token.item())
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                partial_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                print(f"\r {partial_text}", end="", flush=True)
        
        print()
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        self._update_history(user_input, response)
        return response
    
    def _single_generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_tensor)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_tensor,
                attention_mask=attention_mask,
                max_length=len(input_ids) + max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_ids = outputs[0][len(input_ids):]
        response = self.tokenizer.decode(generated_ids.tolist())
        return response
    
    def _build_context(self, user_input: str) -> str:
        context_parts = []
        
        if self.use_memory:
            recent_conversations = self.conversation_memory.get_recent_conversations(3)
            if recent_conversations:
                context_parts.append("历史对话:")
                for conv in recent_conversations:
                    context_parts.append(f"用户: {conv['user_input']}")
                    context_parts.append(f"助手: {conv['model_response']}")
        
        context_parts.append(f"用户: {user_input}")
        context_parts.append("助手:")
        
        return "\n".join(context_parts)
    
    def _update_history(self, user_input: str, model_response: str):
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": model_response})
        
        if self.use_memory:
            importance = self._calculate_importance(user_input, model_response)
            self.conversation_memory.add_conversation(user_input, model_response, importance=importance)
        
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def _calculate_importance(self, user_input: str, model_response: str) -> float:
        importance = min((len(user_input) + len(model_response)) / 500, 1.0)
        
        important_keywords = ['重要', '关键', '记住', '下次', '以后', '学习', '理解']
        for keyword in important_keywords:
            if keyword in user_input or keyword in model_response:
                importance = min(importance + 0.3, 1.0)
        
        return importance
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        return {
            'total_turns': len(self.conversation_history) // 2,
            'recent_conversations': self.conversation_memory.get_recent_conversations(5),
            'important_conversations': self.conversation_memory.get_important_conversations(0.7),
            'topics': list(self.conversation_memory.topic_clusters.keys())
        }
    
    def clear_memory(self):
        self.conversation_memory = ConversationMemory()
        self.conversation_history = []

def chat_with_haoai():
    print("[CHAT] HaoAI对话系统启动中...")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = "weight/haoai_fully_trained_model"
    tokenizer_dir = "weight/tokenizer"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    
    print(f"[INFO] 设备: {device}")
    print(f"[INFO] 模型目录: {model_dir}")
    print(f"[INFO] 分词器目录: {tokenizer_dir}")
    
    if not os.path.exists(model_dir):
        print(f"[ERROR] 模型目录不存在: {model_dir}")
        print("[INFO] 尝试使用SFT模型目录...")
        model_dir = "weight/haoai_sft_model"
        
        if not os.path.exists(model_dir):
            print(f"[ERROR] SFT模型目录也不存在: {model_dir}")
            return
        
    print("[SUCCESS] 模型目录存在")
    
    print("[INFO] 加载分词器...")
    tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
    
    if not os.path.exists(tokenizer_file):
        print(f"[ERROR] 分词器文件不存在: {tokenizer_file}")
        return
    
    print(f"[INFO] 分词器文件: {tokenizer_file}")
    
    tokenizer = BPETokenizer(tokenizer_file)
    
    print(f"[INFO] 词汇表大小: {tokenizer.vocab_size}")
    print(f"[INFO] EOS token ID: {tokenizer.eos_token_id}")
    
    test_text = "什么是人工智能"
    print(f"\n[TEST] 测试分词器: '{test_text}'")
    
    input_ids = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"[INFO] 编码结果: {input_ids}")
    print(f"[INFO] 编码长度: {len(input_ids)}")
    
    if len(input_ids) <= 1:
        print("[WARNING] 分词器可能有问题")
    else:
        print("[SUCCESS] 分词器编码正常")
    
    print("\n[INFO] 加载SmartHaoAI模型...")
    try:
        model = SmartHaoAI.from_pretrained(model_dir, ignore_mismatched_sizes=True)
        model.to(device)
        model.eval()
        print("[SUCCESS] 模型加载成功！")
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    chat_system = EnhancedChatSystem(model, tokenizer, device)
    
    print("=" * 50)
    print("HaoAI 增强对话系统")
    print("=" * 50)
    print("可用命令:")
    print("  help        - 查看帮助")
    print("  memory      - 切换记忆模式 (当前: 开启)")
    print("  reasoning   - 切换推理模式 (当前: 关闭)")
    print("  streaming   - 切换流式模式 (当前: 关闭)")
    print("  auto        - 切换自动模板模式 (当前: 开启)")
    print("  template    - 手动选择提示词模板")
    print("  summary     - 查看对话摘要")
    print("  clear       - 清空对话历史")
    print("  quit        - 退出程序")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n[YOU] 您: ").strip()
            
            if user_input.lower() == 'quit':
                print("[INFO] 再见！")
                break
            elif user_input.lower() == 'help':
                print("\n命令说明:")
                print("  help        - 查看帮助")
                print("  memory      - 切换记忆模式")
                print("  reasoning   - 切换推理模式")
                print("  streaming   - 切换流式模式")
                print("  auto        - 切换自动模板模式")
                print("  template    - 手动选择提示词模板")
                print("  summary     - 查看对话摘要")
                print("  clear       - 清空对话历史")
                print("  quit        - 退出程序")
                continue
            elif user_input.lower() == 'memory':
                chat_system.use_memory = not chat_system.use_memory
                status = "开启" if chat_system.use_memory else "关闭"
                print(f"[INFO] 记忆模式已{status}")
                continue
            elif user_input.lower() == 'reasoning':
                chat_system.use_reasoning = not chat_system.use_reasoning
                status = "开启" if chat_system.use_reasoning else "关闭"
                print(f"[INFO] 推理模式已{status}")
                continue
            elif user_input.lower() == 'streaming':
                chat_system.streaming_mode = not chat_system.streaming_mode
                status = "开启" if chat_system.streaming_mode else "关闭"
                print(f"[INFO] 流式模式已{status}")
                continue
            elif user_input.lower() == 'auto':
                chat_system.auto_template = not chat_system.auto_template
                status = "开启" if chat_system.auto_template else "关闭"
                print(f"[INFO] 自动模板模式已{status}")
                continue
            elif user_input.lower() == 'template':
                templates = chat_system.prompt_engineer.list_templates()
                print(f"\n可用模板: {', '.join(templates)}")
                print("当前模板:", chat_system.prompt_engineer.active_template)
                template_name = input("请输入模板名称 (或输入 'auto' 启用自动选择): ").strip()
                if template_name.lower() == 'auto':
                    chat_system.auto_template = True
                    print("[INFO] 已启用自动模板选择")
                elif template_name in templates:
                    chat_system.prompt_engineer.set_template(template_name)
                    chat_system.auto_template = False
                    print(f"[INFO] 已切换到模板: {template_name}")
                else:
                    print(f"[ERROR] 未找到模板: {template_name}")
                continue
            elif user_input.lower() == 'summary':
                summary = chat_system.get_conversation_summary()
                print("\n对话摘要:")
                print(f"  总对话轮数: {summary['total_turns']}")
                print(f"  涉及主题: {', '.join(summary['topics'])}")
                print(f"  重要对话数: {len(summary['important_conversations'])}")
                stats = chat_system.prompt_engineer.get_statistics()
                print(f"  当前模板: {stats['active_template']}")
                continue
            elif user_input.lower() == 'clear':
                chat_system.clear_memory()
                print("[INFO] 对话历史已清空")
                continue
            elif not user_input:
                continue
            
            print(" HaoAI正在思考...", end="", flush=True)
            start_time = time.time()
            
            try:
                response = chat_system.generate_response(user_input)
                
                if not chat_system.streaming_mode:
                    print(f" (耗时: {time.time() - start_time:.2f}秒)")
                    print(f"\n HaoAI: {response}\n")
                
            except Exception as e:
                print(f"\n 生成失败: {e}")
                import traceback
                traceback.print_exc()
                response = "抱歉，生成回复时出现了技术问题。"
                print(f" HaoAI: {response}\n")
                
        except KeyboardInterrupt:
            print("\n\n 对话已中断，再见！")
            break
        except Exception as e:
            print(f"\n 对话出错: {e}")
            import traceback
            traceback.print_exc()
            print("请重试...\n")

if __name__ == "__main__":
    chat_with_haoai()
