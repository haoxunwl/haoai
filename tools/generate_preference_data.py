"""
生成RLHF偏好数据
用于创建奖励模型训练所需的人类偏好数据
"""

import json
import os
import random
from typing import List, Dict, Any

class PreferenceDataGenerator:
    """偏好数据生成器"""
    
    def __init__(self):
        # 示例对话模板
        self.templates = [
            {
                "prompt": "请解释一下人工智能",
                "good_responses": [
                    "人工智能是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器。它包括机器学习、自然语言处理、计算机视觉等领域。",
                    "人工智能（AI）是让机器模拟人类智能行为的技术，包括学习、推理、感知和解决问题等能力。",
                    "人工智能是通过算法和模型使计算机具备智能行为的技术，广泛应用于各个领域。"
                ],
                "bad_responses": [
                    "人工智能就是机器人。",
                    "我不太清楚，可能是某种科技吧。",
                    "人工智能就是让电脑变聪明的东西。"
                ]
            },
            {
                "prompt": "什么是机器学习",
                "good_responses": [
                    "机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下学习和改进。",
                    "机器学习是通过算法让计算机从数据中学习模式并做出预测或决策的技术。",
                    "机器学习使用统计技术让计算机系统通过经验自动改进性能。"
                ],
                "bad_responses": [
                    "机器学习就是让机器学习。",
                    "这是一种编程方法。",
                    "我不了解这个技术。"
                ]
            },
            {
                "prompt": "深度学习有哪些应用",
                "good_responses": [
                    "深度学习在图像识别、语音识别、自然语言处理、自动驾驶、医疗诊断等领域有广泛应用。",
                    "深度学习应用于计算机视觉、自然语言处理、推荐系统、游戏AI等多个领域。",
                    "深度学习的应用包括人脸识别、机器翻译、智能客服、金融风控等。"
                ],
                "bad_responses": [
                    "深度学习就是深度学习的应用。",
                    "在AI领域有应用。",
                    "很多地方都在用。"
                ]
            }
        ]
        
        # 扩展更多模板
        self._expand_templates()
    
    def _expand_templates(self):
        """扩展模板库"""
        
        additional_templates = [
            {
                "prompt": "自然语言处理是什么",
                "good_responses": [
                    "自然语言处理是人工智能的一个分支，专注于计算机与人类语言之间的交互。",
                    "NLP使计算机能够理解、解释和生成人类语言，包括文本和语音。",
                    "自然语言处理技术包括分词、词性标注、句法分析、语义理解等。"
                ],
                "bad_responses": [
                    "就是处理语言的技术。",
                    "让电脑懂人话。",
                    "一种AI技术。"
                ]
            },
            {
                "prompt": "计算机科学的重要性",
                "good_responses": [
                    "计算机科学是现代社会的基石，推动了数字化转型和科技创新。",
                    "计算机科学在科学研究、经济发展、社会进步中发挥着关键作用。",
                    "计算机科学的重要性体现在它为解决复杂问题提供了有效工具和方法。"
                ],
                "bad_responses": [
                    "计算机科学很重要。",
                    "现在是信息时代。",
                    "各行各业都需要。"
                ]
            },
            {
                "prompt": "编程语言的发展",
                "good_responses": [
                    "编程语言从机器语言、汇编语言发展到高级语言，越来越接近自然语言。",
                    "编程语言的发展趋势是更高效、更安全、更易用，支持多种编程范式。",
                    "现代编程语言强调开发效率、可维护性和跨平台能力。"
                ],
                "bad_responses": [
                    "编程语言一直在发展。",
                    "从低级到高级。",
                    "越来越先进。"
                ]
            }
        ]
        
        self.templates.extend(additional_templates)
    
    def generate_preference_pair(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """生成一个偏好对"""
        
        # 随机选择好的响应和差的响应
        good_response = random.choice(template["good_responses"])
        bad_response = random.choice(template["bad_responses"])
        
        # 随机决定是否交换顺序（避免模型学习到顺序偏好）
        if random.random() < 0.5:
            chosen = good_response
            rejected = bad_response
        else:
            chosen = bad_response
            rejected = good_response
        
        return {
            "prompt": template["prompt"],
            "chosen": chosen,
            "rejected": rejected,
            "quality": "good" if chosen == good_response else "bad"
        }
    
    def generate_dataset(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """生成完整的数据集"""
        
        dataset = []
        
        for i in range(num_samples):
            # 随机选择一个模板
            template = random.choice(self.templates)
            
            # 生成偏好对
            preference_pair = self.generate_preference_pair(template)
            
            dataset.append(preference_pair)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], file_path: str):
        """保存数据集到文件"""
        
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                # 移除quality字段（训练时不需要）
                item_to_save = {
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"]
                }
                f.write(json.dumps(item_to_save, ensure_ascii=False) + '\n')
        
        print(f"[SUCCESS] 数据集已保存到: {file_path}")
        print(f"[INFO] 样本数量: {len(dataset)}")
    
    def analyze_dataset(self, dataset: List[Dict[str, Any]]):
        """分析数据集"""
        
        good_count = sum(1 for item in dataset if item["quality"] == "good")
        bad_count = len(dataset) - good_count
        
        print(f"[INFO] 数据集分析:")
        print(f"   总样本数: {len(dataset)}")
        print(f"   优质响应: {good_count} ({good_count/len(dataset)*100:.1f}%)")
        print(f"   劣质响应: {bad_count} ({bad_count/len(dataset)*100:.1f}%)")
        
        # 统计不同提示的出现次数
        prompt_counts = {}
        for item in dataset:
            prompt = item["prompt"]
            prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
        
        print(f"   不同提示数量: {len(prompt_counts)}")
        print(f"   最频繁的提示:")
        
        for prompt, count in sorted(prompt_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     - {prompt}: {count}次")

def create_preference_data(
    output_file: str = "training_data/rlhf/preference_data.jsonl",
    num_samples: int = 1000
):
    """创建偏好数据"""
    
    print("🎯 开始生成RLHF偏好数据")
    print("=" * 50)
    
    # 创建生成器
    generator = PreferenceDataGenerator()
    
    # 生成数据集
    print(f"生成 {num_samples} 个偏好样本...")
    dataset = generator.generate_dataset(num_samples)
    
    # 分析数据集
    generator.analyze_dataset(dataset)
    
    # 保存数据集
    generator.save_dataset(dataset, output_file)
    
    print("\n🎉 偏好数据生成完成！")
    
    return dataset

def main():
    """主函数"""
    
    # 生成偏好数据
    dataset = create_preference_data(
        output_file="training_data/rlhf/preference_data.jsonl",
        num_samples=500  # 可以根据需要调整样本数量
    )
    
    # 显示一些示例
    print("\n📋 数据示例:")
    print("=" * 50)
    
    for i, item in enumerate(dataset[:3]):
        print(f"\n示例 {i+1}:")
        print(f"提示: {item['prompt']}")
        print(f"选择的响应: {item['chosen']}")
        print(f"拒绝的响应: {item['rejected']}")
        print(f"质量: {item['quality']}")

if __name__ == "__main__":
    main()