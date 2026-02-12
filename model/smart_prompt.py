"""
智能提示词工程模块
通过优化提示词来提升模型性能
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    name: str
    template: str
    description: str
    required_fields: List[str]
    optional_fields: List[str] = None

class SmartPromptEngineer:
    def __init__(self):
        self.templates = self._load_default_templates()
        self.active_template = "default"
        self.conversation_context = []
        self.user_preferences = {}
    
    def _load_default_templates(self) -> Dict[str, PromptTemplate]:
        return {
            "default": PromptTemplate(
                name="default",
                template="<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n",
                description="默认对话模板",
                required_fields=["user_input"]
            ),
            "detailed": PromptTemplate(
                name="detailed",
                template="""<|im_start|>system\n你是一个智能助手，请详细、准确地回答用户的问题。{context}<|im_end|>
<|im_start|>user\n{user_input}<|im_end|>
<|im_start|>assistant\n""",
                description="详细回答模板",
                required_fields=["user_input"],
                optional_fields=["context"]
            ),
            "creative": PromptTemplate(
                name="creative",
                template="""<|im_start|>system\n你是一个富有创造力的助手，请用生动、有趣的方式回答问题。{context}<|im_end|>
<|im_start|>user\n{user_input}<|im_end|>
<|im_start|>assistant\n""",
                description="创意回答模板",
                required_fields=["user_input"],
                optional_fields=["context"]
            ),
            "technical": PromptTemplate(
                name="technical",
                template="""<|im_start|>system\n你是一个技术专家，请用专业、准确的技术术语回答问题。如果不确定，请明确说明。{context}<|im_end|>
<|im_start|>user\n{user_input}<|im_end|>
<|im_start|>assistant\n""",
                description="技术问答模板",
                required_fields=["user_input"],
                optional_fields=["context"]
            ),
            "step_by_step": PromptTemplate(
                name="step_by_step",
                template="""<|im_start|>system\n请逐步分析问题，给出清晰的步骤和解释。{context}<|im_end|>
<|im_start|>user\n{user_input}<|im_end|>
<|im_start|>assistant\n让我逐步分析这个问题：\n""",
                description="逐步分析模板",
                required_fields=["user_input"],
                optional_fields=["context"]
            ),
            "chain_of_thought": PromptTemplate(
                name="chain_of_thought",
                template="""<|im_start|>system\n请使用思考链方式回答问题，展示你的推理过程。{context}<|im_end|>
<|im_start|>user\n{user_input}<|im_end|>
<|im_start|>assistant\n思考过程：\n""",
                description="思考链模板",
                required_fields=["user_input"],
                optional_fields=["context"]
            ),
            "code_assistant": PromptTemplate(
                name="code_assistant",
                template="""<|im_start|>system\n你是一个编程助手，请提供清晰的代码示例和解释。{context}<|im_end|>
<|im_start|>user\n{user_input}<|im_end|>
<|im_start|>assistant\n""",
                description="代码助手模板",
                required_fields=["user_input"],
                optional_fields=["context"]
            ),
            "educational": PromptTemplate(
                name="educational",
                template="""<|im_start|>system\n你是一个教育助手，请用简单易懂的方式解释概念，适合初学者学习。{context}<|im_end|>
<|im_start|>user\n{user_input}<|im_end|>
<|im_start|>assistant\n""",
                description="教育模板",
                required_fields=["user_input"],
                optional_fields=["context"]
            )
        }
    
    def set_template(self, template_name: str) -> bool:
        if template_name in self.templates:
            self.active_template = template_name
            return True
        return False
    
    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        return self.templates.get(template_name)
    
    def list_templates(self) -> List[str]:
        return list(self.templates.keys())
    
    def optimize_prompt(self, user_input: str, context: Optional[str] = None) -> str:
        template = self.templates[self.active_template]
        
        user_input = self._enhance_input(user_input)
        
        if context:
            context = self._format_context(context)
        
        formatted_prompt = template.template.format(
            user_input=user_input,
            context=context if context else ""
        )
        
        return formatted_prompt
    
    def _enhance_input(self, user_input: str) -> str:
        enhanced = user_input
        
        if len(enhanced) < 10:
            enhanced = f"请回答：{enhanced}"
        
        if not any(punctuation in enhanced for punctuation in ['？', '?', '。', '.', '！', '!']):
            if not enhanced.endswith('？') and not enhanced.endswith('?'):
                enhanced += '？'
        
        return enhanced
    
    def _format_context(self, context: str) -> str:
        if not context:
            return ""
        
        formatted = "\n相关背景：\n"
        
        if isinstance(context, list):
            for item in context[:3]:
                formatted += f"- {item}\n"
        else:
            formatted += f"{context}\n"
        
        return formatted
    
    def detect_intent(self, user_input: str) -> str:
        intent_patterns = {
            "code": [r"代码", r"编程", r"函数", r"类", r"算法", r"bug", r"调试", r"python", r"java", r"javascript"],
            "technical": [r"技术", r"原理", r"机制", r"架构", r"设计", r"实现"],
            "creative": [r"创意", r"想象", r"故事", r"诗歌", r"艺术", r"设计"],
            "educational": [r"学习", r"教学", r"解释", r"理解", r"入门", r"教程"],
            "step_by_step": [r"步骤", r"如何", r"怎么做", r"怎么", r"方法", r"流程"]
        }
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input, re.IGNORECASE):
                    return intent
        
        return "default"
    
    def auto_select_template(self, user_input: str) -> str:
        intent = self.detect_intent(user_input)
        
        template_mapping = {
            "code": "code_assistant",
            "technical": "technical",
            "creative": "creative",
            "educational": "educational",
            "step_by_step": "step_by_step"
        }
        
        selected_template = template_mapping.get(intent, "default")
        self.set_template(selected_template)
        
        return selected_template
    
    def add_conversation_context(self, role: str, content: str):
        self.conversation_context.append({"role": role, "content": content})
        
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
    
    def get_conversation_context(self) -> str:
        if not self.conversation_context:
            return ""
        
        context_parts = []
        for conv in self.conversation_context[-6:]:
            role_name = "用户" if conv["role"] == "user" else "助手"
            context_parts.append(f"{role_name}: {conv['content']}")
        
        return "\n".join(context_parts)
    
    def clear_context(self):
        self.conversation_context = []
    
    def set_preference(self, key: str, value: any):
        self.user_preferences[key] = value
    
    def get_preference(self, key: str, default=None):
        return self.user_preferences.get(key, default)
    
    def generate_system_prompt(self, user_input: str) -> str:
        intent = self.detect_intent(user_input)
        
        system_prompts = {
            "code": "你是一个专业的编程助手，擅长各种编程语言和技术问题。请提供清晰的代码示例和详细解释。",
            "technical": "你是一个技术专家，擅长解释复杂的技术概念和原理。请用准确的专业术语回答问题。",
            "creative": "你是一个富有创造力的助手，擅长创意写作和艺术表达。请用生动有趣的方式回答问题。",
            "educational": "你是一个教育专家，擅长将复杂的概念简单化。请用通俗易懂的方式解释问题，适合初学者学习。",
            "step_by_step": "你是一个逻辑清晰的助手，擅长逐步分析问题。请按照清晰的步骤回答问题，每个步骤都要有明确的解释。",
            "default": "你是一个智能助手，能够准确理解用户的问题并提供有用的回答。请保持专业、友好、有礼貌的态度。"
        }
        
        return system_prompts.get(intent, system_prompts["default"])
    
    def format_multi_turn_conversation(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        formatted = ""
        
        if system_prompt:
            formatted += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        for msg in messages:
            formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        
        return formatted
    
    def optimize_for_length(self, prompt: str, max_length: int = 2048) -> str:
        if len(prompt) <= max_length:
            return prompt
        
        tokens = prompt.split()
        if len(tokens) <= max_length:
            return prompt
        
        truncated = " ".join(tokens[:max_length])
        
        if not truncated.endswith("<|im_end|>"):
            truncated = truncated.rsplit("<|im_start|>", 1)[0]
            truncated += "<|im_end|>\n"
        
        return truncated
    
    def add_examples(self, prompt: str, examples: List[Dict[str, str]]) -> str:
        if not examples:
            return prompt
        
        examples_section = "\n示例对话：\n"
        for i, example in enumerate(examples, 1):
            examples_section += f"\n示例 {i}:\n"
            examples_section += f"用户: {example['user']}\n"
            examples_section += f"助手: {example['assistant']}\n"
        
        return prompt.replace("{context}", examples_section + "{context}")
    
    def get_statistics(self) -> Dict[str, any]:
        return {
            "active_template": self.active_template,
            "available_templates": len(self.templates),
            "conversation_context_length": len(self.conversation_context),
            "user_preferences": len(self.user_preferences)
        }

def create_smart_prompt_engineer() -> SmartPromptEngineer:
    return SmartPromptEngineer()

class PromptOptimizer:
    def __init__(self):
        self.engineer = SmartPromptEngineer()
        self.performance_history = {}
    
    def optimize_for_task(self, user_input: str, task_type: str) -> str:
        template_mapping = {
            "qa": "default",
            "coding": "code_assistant",
            "creative": "creative",
            "explanation": "educational",
            "analysis": "step_by_step",
            "reasoning": "chain_of_thought"
        }
        
        template_name = template_mapping.get(task_type, "default")
        self.engineer.set_template(template_name)
        
        return self.engineer.optimize_prompt(user_input)
    
    def record_performance(self, prompt: str, quality_score: float):
        prompt_hash = hash(prompt)
        self.performance_history[prompt_hash] = {
            "prompt": prompt,
            "quality_score": quality_score,
            "timestamp": time.time() if 'time' in globals() else 0
        }
    
    def get_best_template(self) -> str:
        if not self.performance_history:
            return "default"
        
        template_scores = {}
        for entry in self.performance_history.values():
            template = self.engineer.active_template
            if template not in template_scores:
                template_scores[template] = []
            template_scores[template].append(entry["quality_score"])
        
        best_template = "default"
        best_avg_score = 0
        
        for template, scores in template_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_template = template
        
        return best_template

def create_prompt_optimizer() -> PromptOptimizer:
    return PromptOptimizer()