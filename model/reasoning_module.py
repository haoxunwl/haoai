"""
智能推理模块 - 增强模型的逻辑推理和问题解决能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any

class ReasoningModule(nn.Module):
    def __init__(self, hidden_size: int, num_reasoning_layers: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_reasoning_layers = num_reasoning_layers
        
        self.reasoning_layers = nn.ModuleList([
            ReasoningLayer(hidden_size) for _ in range(num_reasoning_layers)
        ])
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.reasoning_state = None
        
        self.logic_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states: torch.Tensor, 
                reasoning_context: Optional[torch.Tensor] = None,
                reasoning_steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        if self.reasoning_state is None or self.reasoning_state.size(0) != batch_size:
            self.reasoning_state = hidden_states.clone().to(hidden_states.device)
        
        for step in range(reasoning_steps):
            for layer in self.reasoning_layers:
                self.reasoning_state = layer(self.reasoning_state)
            
            if reasoning_context is not None:
                attended_state, _ = self.cross_attention(
                    query=self.reasoning_state,
                    key=reasoning_context,
                    value=reasoning_context
                )
                self.reasoning_state = self.reasoning_state + attended_state
        
        fusion_gate = torch.sigmoid(
            nn.Linear(hidden_size * 2, hidden_size)(
                torch.cat([hidden_states, self.reasoning_state], dim=-1)
            )
        )
        
        enhanced_states = fusion_gate * self.reasoning_state + (1 - fusion_gate) * hidden_states
        
        reasoning_scores = self.logic_head(enhanced_states.mean(dim=1))
        
        return enhanced_states, reasoning_scores
    
    def reset_state(self):
        self.reasoning_state = None
    
    def to(self, device, *args, **kwargs):
        """Override to method to ensure reasoning_state is moved to correct device"""
        self = super().to(device, *args, **kwargs)
        if self.reasoning_state is not None:
            self.reasoning_state = self.reasoning_state.to(device)
        return self

class ReasoningLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x_norm = self.layer_norm1(x)
        attended, _ = self.self_attention(x_norm, x_norm, x_norm)
        
        gate_weights = torch.sigmoid(self.gate(torch.cat([x, attended], dim=-1)))
        x = residual + gate_weights * attended
        
        residual = x
        x_norm = self.layer_norm2(x)
        ff_out = self.feed_forward(x_norm)
        
        gate_weights = torch.sigmoid(self.gate(torch.cat([x, ff_out], dim=-1)))
        x = residual + gate_weights * ff_out
        
        return x

class LogicalReasoningHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        self.logic_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        self.reasoning_type_head = nn.Linear(hidden_size, 4)
    
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        sequence_rep = hidden_states.mean(dim=1)
        
        logic_output = self.logic_network(sequence_rep)
        
        reasoning_type = self.reasoning_type_head(sequence_rep)
        
        return {
            'logic_output': logic_output,
            'reasoning_type': reasoning_type,
            'confidence': F.softmax(logic_output, dim=-1).max(dim=-1)[0]
        }

class MultiStepReasoning:
    def __init__(self, model, tokenizer, max_steps: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        
        self.reasoning_history = []
        self.current_step = 0
    
    def reason_step(self, input_text: str, context: Optional[str] = None) -> Dict[str, Any]:
        if self.current_step >= self.max_steps:
            return {'status': 'max_steps_reached', 'result': None}
        
        inputs = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        
        context_inputs = None
        if context:
            context_inputs = self.tokenizer(context, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            if hasattr(self.model, 'reasoning_module') and self.model.reasoning_module is not None:
                context_hidden = None
                if context_inputs and 'input_ids' in context_inputs:
                    with torch.no_grad():
                        context_outputs = self.model(**context_inputs)
                        context_hidden = context_outputs.hidden_states if hasattr(context_outputs, 'hidden_states') else None
                
                reasoning_output = self.model.reasoning_module(
                    outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state,
                    context_hidden
                )
            else:
                reasoning_output = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs.logits
        
        generated_text = self.tokenizer.decode(
            outputs.logits.argmax(dim=-1)[0], 
            skip_special_tokens=True
        )
        
        reasoning_step = {
            'step': self.current_step,
            'input': input_text,
            'output': generated_text,
            'confidence': outputs.logits.softmax(dim=-1).max().item()
        }
        self.reasoning_history.append(reasoning_step)
        self.current_step += 1
        
        return {
            'status': 'success',
            'result': generated_text,
            'step': self.current_step - 1,
            'confidence': reasoning_step['confidence']
        }
    
    def multi_step_reasoning(self, initial_input: str, context: Optional[str] = None) -> Dict[str, Any]:
        self.reset()
        
        current_input = initial_input
        final_result = None
        
        for step in range(self.max_steps):
            result = self.reason_step(current_input, context)
            
            if result['status'] != 'success':
                break
                
            final_result = result
            
            if self._should_terminate(result):
                break
            
            current_input = result['result']
        
        return {
            'final_result': final_result,
            'reasoning_history': self.reasoning_history,
            'total_steps': self.current_step
        }
    
    def _should_terminate(self, result: Dict[str, Any]) -> bool:
        if result['confidence'] > 0.95:
            return True
        
        if self.current_step >= self.max_steps:
            return True
        
        if len(self.reasoning_history) >= 2:
            last_two = self.reasoning_history[-2:]
            if last_two[0]['output'] == last_two[1]['output']:
                return True
        
        return False
    
    def reset(self):
        self.reasoning_history = []
        self.current_step = 0

def create_reasoning_module(model, hidden_size: int) -> ReasoningModule:
    reasoning_module = ReasoningModule(hidden_size)
    model.reasoning_module = reasoning_module
    return reasoning_module