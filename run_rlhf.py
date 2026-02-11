"""
直接运行RLHF训练
"""

import os
import json
import time
import random
from typing import Dict, Any, List, Optional

import torch
from tqdm import tqdm

# 自定义模块
from model.model import SmartHaoAI, HaoAIConfig
from model.reward_model import RewardModel
from simple_tokenizer import SimpleBPETokenizer as BPETokenizer
from train.rlhf_config import get_rlhf_config, RLHFConfig
from train.reward_trainer import train_reward_model
from train.ppo_trainer import create_ppo_trainer

# 训练器类
class RLHFTrainer:
    """RLHF训练器（稳定增强版）"""

    def __init__(self, config_preset: str = "balanced"):
        self.config: RLHFConfig = get_rlhf_config(config_preset)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tokenizer: Optional[BPETokenizer] = None
        self.policy_model: Optional[SmartHaoAI] = None
        self.reward_model: Optional[RewardModel] = None
        self.ppo_trainer = None

        self.current_stage = "init"
        self.training_stats: Dict[str, Any] = {}

    # ========================
    # 环境初始化
    # ========================
    def setup_environment(self) -> bool:
        print("\n[RLHF] 初始化训练环境")
        print("=" * 60)
        print(f"配置预设: {self.config}")
        print(f"运行设备: {self.device}")

        os.makedirs(self.config.rlhf_model_dir, exist_ok=True)
        os.makedirs(self.config.reward_model_dir, exist_ok=True)

        self.tokenizer = self._load_tokenizer()
        if self.tokenizer is None:
            return False

        self.policy_model = self._load_policy_model()
        if self.policy_model is None:
            return False

        # 模型已经在_load_policy_model方法中移动到正确的设备上了
        # self.policy_model.to(self.device)
        self.policy_model.train()

        print("[SUCCESS] 环境准备完成")
        return True

    def _find_project_root(self) -> str:
        return os.path.abspath(
            os.path.dirname(__file__)
        )

    def _load_tokenizer(self) -> Optional[BPETokenizer]:
        project_root = self._find_project_root()

        candidates = [
            "weight/tokenizer/tokenizer.json",
            "weight/tokenizer.json",
            "tokenizer.json"
        ]

        for rel in candidates:
            path = os.path.join(project_root, rel)
            if os.path.exists(path):
                print(f"[LOAD] tokenizer: {path}")
                return BPETokenizer(path)

        print("[ERROR] 未找到 tokenizer.json")
        return None

    def _load_policy_model(self) -> Optional[SmartHaoAI]:
        print("[LOAD] 策略模型（SFT）")

        if os.path.exists(self.config.sft_model_dir):
            try:
                # 直接加载模型到指定设备
                model = SmartHaoAI.from_pretrained(
                    self.config.sft_model_dir,
                    device_map={"": self.device}
                )
                print("[SUCCESS] 加载 SFT 模型")
                return model
            except Exception as e:
                print(f"[WARN] SFT 加载失败: {e}")
                
                # 如果加载失败，尝试创建新模型
                print("[INFO] 创建新策略模型")
                config = HaoAIConfig(
                    vocab_size=self.tokenizer.vocab_size,
                    n_layer=8,
                    n_head=8,
                    n_embd=1024
                )
                model = SmartHaoAI(config)
                model.to(self.device)
                return model

        print("[INFO] 创建新策略模型")

        config = HaoAIConfig(
            vocab_size=self.tokenizer.vocab_size,
            n_layer=8,
            n_head=8,
            n_embd=1024
        )
        model = SmartHaoAI(config)
        model.to(self.device)
        return model

    # ========================
    # 奖励模型
    # ========================
    def train_reward_model(self) -> bool:
        if not self.config.enable_reward_training:
            print("[SKIP] 奖励模型训练被禁用")
            return True

        print("\n[Stage 1] 训练奖励模型")
        print("-" * 40)

        if not os.path.exists(self.config.preference_data_file):
            print("[ERROR] 偏好数据不存在")
            return False

        trainer = train_reward_model(
            tokenizer=self.tokenizer,
            data_file=self.config.preference_data_file,
            config=self.config.reward_config,
            save_dir=self.config.reward_model_dir
        )

        if trainer is None:
            return False

        self.reward_model = trainer.reward_model.to(self.device)
        self.reward_model.eval()

        print("[SUCCESS] 奖励模型训练完成")
        return True

    def load_reward_model(self) -> bool:
        print("[LOAD] 奖励模型")

        for name in ["best_model", "final_model"]:
            path = os.path.join(self.config.reward_model_dir, name)
            if os.path.exists(path):
                self.reward_model = RewardModel.from_pretrained(path)
                self.reward_model.to(self.device).eval()
                print(f"[SUCCESS] 使用奖励模型: {name}")
                return True

        print("[ERROR] 未找到奖励模型")
        return False

    # ========================
    # PPO 训练
    # ========================
    def train_with_ppo(self) -> bool:
        print("\n[Stage 2] PPO 强化学习")
        print("-" * 40)

        if self.reward_model is None:
            if not self.load_reward_model():
                return False

        self.ppo_trainer = create_ppo_trainer(
            self.policy_model,
            self.reward_model,
            self.tokenizer,
            self.config.ppo_config,
            device=self.device
        )

        return self._run_ppo_loop()

    def _sample_prompts(self, pool: List[str], k: int) -> List[str]:
        return random.sample(pool, k=min(k, len(pool)))

    def _run_ppo_loop(self) -> bool:
        prompts_pool = [
            "请解释什么是人工智能",
            "机器学习的核心思想是什么",
            "深度学习和传统算法的区别",
            "自然语言处理的应用",
            "神经网络如何工作",
            "大模型的优势与局限",
            "强化学习的原理",
            "AI 对社会的影响",
            "如何评价一个语言模型",
            "未来 AI 的发展趋势"
        ]

        total_steps = self.config.ppo_config.total_timesteps
        save_freq = self.config.ppo_config.save_frequency

        pbar = tqdm(range(total_steps), desc="PPO Training")

        for step in pbar:
            try:
                prompts = self._sample_prompts(prompts_pool, k=4)

                rollouts = self.ppo_trainer.collect_rollouts(prompts)
                if not rollouts:
                    continue

                loss_info = self.ppo_trainer.train_step(rollouts)

                if loss_info:
                    pbar.set_postfix({
                        "reward": f"{loss_info.get('mean_reward', 0):.3f}",
                        "policy": f"{loss_info.get('policy_loss', 0):.3f}",
                        "value": f"{loss_info.get('value_loss', 0):.3f}",
                    })

                if step % save_freq == 0 and step > 0:
                    self._save_checkpoint(step)

            except Exception as e:
                print(f"[WARN] step {step} 失败: {e}")

        print("[SUCCESS] PPO 训练完成")
        return True

    # ========================
    # 保存 & 评估
    # ========================
    def _save_checkpoint(self, step: int):
        path = os.path.join(
            self.config.rlhf_model_dir,
            f"checkpoint_{step}"
        )
        os.makedirs(path, exist_ok=True)

        self.policy_model.save_pretrained(path)

        with open(os.path.join(path, "info.json"), "w", encoding="utf-8") as f:
            json.dump({
                "step": step,
                "time": time.time()
            }, f, indent=2, ensure_ascii=False)

        print(f"[SAVE] checkpoint @ step {step}")

    def train(self) -> bool:
        if not self.setup_environment():
            return False

        self.current_stage = "reward"
        if self.config.enable_reward_training:
            if not self.train_reward_model():
                return False
        else:
            if not self.load_reward_model():
                return False

        self.current_stage = "ppo"
        if self.config.enable_ppo_training:
            if not self.train_with_ppo():
                return False

        self._save_final_model()
        print("\n🎉 RLHF 训练流程完成")
        return True

    def _save_final_model(self):
        path = os.path.join(self.config.rlhf_model_dir, "final_model")
        os.makedirs(path, exist_ok=True)

        self.policy_model.save_pretrained(path)

        with open(os.path.join(path, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.time(),
                "architecture": "HaoAI-RLHF",
                "device": str(self.device)
            }, f, indent=2, ensure_ascii=False)

        print(f"[SAVE] 最终模型保存至 {path}")

# 主函数
def main():
    print("直接运行RLHF训练")
    print("=" * 50)
    
    trainer = RLHFTrainer(config_preset="balanced")
    ok = trainer.train()

    if ok:
        print("[SUCCESS] 训练成功")
    else:
        print("[FAILED] 训练失败")

# 运行主函数
if __name__ == "__main__":
    main()