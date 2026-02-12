import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import SmartHaoAI, HaoAIConfig

def convert_to_hf_format():
    """将训练好的模型转换为Hugging Face格式"""
    
    from train.config import SFTConfig
    sft_cfg = SFTConfig()
    
    model_dir = sft_cfg.sft_model_dir
    hf_model_dir = os.path.join(model_dir, "hf_format")
    
    os.makedirs(hf_model_dir, exist_ok=True)
    
    print(f"转换模型格式...")
    print(f"源目录: {model_dir}")
    print(f"目标目录: {hf_model_dir}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        config = HaoAIConfig.from_pretrained(model_dir)
        model = HaoAI.from_pretrained(model_dir)
        
        model.save_pretrained(hf_model_dir)
        tokenizer.save_pretrained(hf_model_dir)
        config.save_pretrained(hf_model_dir)
        
        print(f"模型已转换为Hugging Face格式并保存至: {hf_model_dir}")
        
    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    convert_to_hf_format()