import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import SmartHaoAI, HaoAIConfig
from train.config import SFTConfig
from model.conversation_memory import ConversationMemory, MultiTurnDialogueModel
from model.reasoning_module import MultiStepReasoning
from model.smart_prompt import SmartPromptEngineer, create_smart_prompt_engineer


class SmartInferenceEngine:
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
    
    def generate_with_chain_of_thought(self, prompt: str, max_thoughts: int = 3) -> str:
        thoughts = []
        current_prompt = prompt
        
        for i in range(max_thoughts):
            thought_prompt = f"思考步骤 {i+1}: {current_prompt}"
            thought = self._generate_single(thought_prompt, max_new_tokens=128, temperature=0.8)
            thoughts.append(thought)
            
            if "结论" in thought or "答案" in thought or "结果" in thought:
                break
            
            current_prompt = f"{current_prompt}\n思考: {thought}\n继续思考..."
        
        final_prompt = f"{prompt}\n思考过程:\n" + "\n".join(thoughts) + "\n最终回答:"
        final_response = self._generate_single(final_prompt, max_new_tokens=256, temperature=0.7)
        
        return final_response
    
    def generate_streaming(self, prompt: str, callback=None) -> str:
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        generated_ids = input_ids.clone()
        response_tokens = []
        
        with torch.no_grad():
            for _ in range(256):
                outputs = self.model(
                    generated_ids,
                    use_cache=True,
                    past_key_values=None if len(response_tokens) == 0 else outputs.past_key_values
                )
                
                logits = outputs.logits[:, -1, :]
                logits = logits / 0.7
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                response_tokens.append(next_token.item())
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                if callback:
                    partial_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                    callback(partial_text)
        
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    def generate_with_memory(self, prompt: str) -> str:
        recent_history = self.conversation_memory.get_recent_conversations(3)
        
        context = ""
        if recent_history:
            context = "历史对话:\n"
            for conv in recent_history:
                context += f"用户: {conv['user_input']}\n助手: {conv['model_response']}\n"
        
        full_prompt = f"{context}\n当前问题: {prompt}\n回答:"
        
        response = self._generate_single(full_prompt, max_new_tokens=256)
        
        self.conversation_memory.add_conversation(prompt, response, importance=0.7)
        
        return response
    
    def _generate_single(self, prompt: str, max_new_tokens=256, temperature=0.7) -> str:
        context = self.conversation_memory.get_recent_conversations(2)
        context_str = ""
        if context:
            context_str = "\n".join([f"用户: {c['user_input']}\n助手: {c['model_response']}" for c in context])
        
        optimized_prompt = self.prompt_engineer.optimize_prompt(prompt, context_str)
        
        input_ids = self.tokenizer.encode(optimized_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, 'generate_with_reasoning') and self.model.use_reasoning:
                outputs = self.model.generate_with_reasoning(
                    input_ids,
                    max_length=len(input_ids[0]) + max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            else:
                outputs = self.model.generate(
                    input_ids,
                    max_length=len(input_ids[0]) + max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = response.replace(optimized_prompt, "")
        response = response.replace("<|im_start|>assistant\n", "")
        response = response.replace("<|im_end|>", "").strip()
        
        return response


try:
    from transformers import AutoConfig
    AutoConfig.register("haoai", HaoAIConfig)

    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
    _AutoModelForCausalLM.register(HaoAIConfig, SmartHaoAI)
    print("模型注册成功！")
except Exception as e:
    print(f"模型注册警告: {e}")


def run_inference():
    sft_cfg = SFTConfig()
    device = sft_cfg.device if torch.cuda.is_available() else "cpu"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "..", "train", sft_cfg.sft_model_dir)

    print(f"加载模型和分词器...")
    print(f"模型目录: {model_dir}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            device_map="auto" if device.startswith("cuda") else None,
            trust_remote_code=True
        )
        model.eval()

        print("模型加载成功！")

        if device.startswith("cuda"):
            model.half()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        inference_engine = SmartInferenceEngine(model, tokenizer, device)

        print("\n" + "="*50)
        print("HaoAI 智能推理系统")
        print("="*50)
        print("可用模式:")
        print("1. 普通对话 - 直接回答问题")
        print("2. 思考链推理 - 展示思考过程")
        print("3. 流式生成 - 逐字显示回答")
        print("4. 记忆对话 - 记住对话历史")
        print("输入 'quit' 退出, 'help' 查看帮助")
        print("="*50)

        current_mode = 1

        while True:
            print(f"\n[当前模式: {['普通对话', '思考链推理', '流式生成', '记忆对话'][current_mode-1]}]")
            user_input = input("请输入问题: ").strip()
            
            if user_input.lower() == 'quit':
                print("退出推理。")
                break
            elif user_input.lower() == 'help':
                print("\n命令说明:")
                print("  mode 1 - 切换到普通对话模式")
                print("  mode 2 - 切换到思考链推理模式")
                print("  mode 3 - 切换到流式生成模式")
                print("  mode 4 - 切换到记忆对话模式")
                print("  clear - 清空对话记忆")
                print("  quit - 退出程序")
                continue
            elif user_input.lower() == 'clear':
                inference_engine.conversation_memory = ConversationMemory()
                print("对话记忆已清空")
                continue
            elif user_input.lower().startswith('mode '):
                try:
                    mode = int(user_input.split()[1])
                    if 1 <= mode <= 4:
                        current_mode = mode
                        print(f"已切换到{['普通对话', '思考链推理', '流式生成', '记忆对话'][mode-1]}模式")
                    else:
                        print("无效的模式，请输入1-4")
                except:
                    print("模式切换失败，请输入: mode 1/2/3/4")
                continue
            elif not user_input:
                continue

            start_time = time.time()

            if current_mode == 1:
                response = inference_engine._generate_single(user_input)
                print(f"\nHaoAI: {response}")
            
            elif current_mode == 2:
                print("\n 正在思考...")
                response = inference_engine.generate_with_chain_of_thought(user_input)
                print(f"\nHaoAI: {response}")
            
            elif current_mode == 3:
                print("\n 正在生成...")
                def stream_callback(text):
                    print(f"\r生成中: {text}", end="", flush=True)
                
                response = inference_engine.generate_streaming(user_input, stream_callback)
                print(f"\n\nHaoAI: {response}")
            
            elif current_mode == 4:
                response = inference_engine.generate_with_memory(user_input)
                print(f"\nHaoAI: {response}")

            end_time = time.time()
            print(f"\n⏱  耗时: {end_time - start_time:.2f}秒")

    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_inference()
