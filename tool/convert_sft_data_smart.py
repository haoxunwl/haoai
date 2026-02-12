#!/usr/bin/env python3
"""
SFT训练数据转换工具 - 智能版
智能解析连续问答文本，转换为标准对话格式
"""

import os
import json
import re
from typing import List, Dict, Any

def parse_qa_text_smart(content: str) -> List[Dict[str, str]]:
    """
    智能解析连续问答文本，提取问题和答案对
    
    Args:
        content: 原始文本内容
        
    Returns:
        问答对列表
    """
    qa_pairs = []
    
    # 按行处理
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 检查是否是问题行（以问号结尾或包含特定关键词）
        is_question = (
            line.endswith('？') or line.endswith('?') or 
            line.startswith('什么是') or 
            line.startswith('介绍一下') or 
            line.startswith('帮我') or 
            '有什么区别' in line
        )
        
        if is_question:
            # 提取问题部分（只保留问题本身，不包括答案）
            # 问题通常以问号结尾，或者是一个完整的疑问句
            question_match = re.match(r'^([^。！？.!]*[？?])', line)
            if question_match:
                question = question_match.group(1).strip()
                
                # 剩余部分是当前行的答案部分
                remaining_line = line[len(question):].strip()
                answer_lines = []
                
                if remaining_line:
                    answer_lines.append(remaining_line)
                
                # 收集后续行作为答案，直到遇到下一个问题
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    # 检查下一行是否是新的问题
                    next_is_question = (
                        next_line.endswith('？') or next_line.endswith('?') or 
                        next_line.startswith('什么是') or 
                        next_line.startswith('介绍一下') or 
                        next_line.startswith('帮我') or 
                        '有什么区别' in next_line
                    )
                    
                    if next_is_question:
                        break
                    answer_lines.append(next_line)
                    j += 1
                
                # 如果有答案内容，则保存问答对
                if answer_lines:
                    answer = ' '.join(answer_lines)
                    qa_pairs.append({
                        'question': question,
                        'answer': answer
                    })
                    
                    # 跳过已处理的答案行
                    i = j
                    continue
        
        i += 1
    
    return qa_pairs

def qa_to_conversation(qa_pairs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    将问答对转换为对话格式
    
    Args:
        qa_pairs: 问答对列表
        
    Returns:
        对话格式数据
    """
    conversations = []
    
    for qa in qa_pairs:
        conversation = {
            "conversations": [
                {
                    "role": "user",
                    "content": qa['question']
                },
                {
                    "role": "assistant", 
                    "content": qa['answer']
                }
            ]
        }
        conversations.append(conversation)
    
    return conversations

def convert_text_to_sft_smart(input_file: str, output_file: str) -> None:
    """
    将文本文件转换为SFT训练数据格式
    
    Args:
        input_file: 输入文本文件路径
        output_file: 输出JSONL文件路径
    """
    
    # 读取输入文件
    print(f"正在读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("正在智能解析问答对...")
    qa_pairs = parse_qa_text_smart(content)
    print(f"解析到 {len(qa_pairs)} 个问答对")
    
    # 显示解析结果
    print("\n解析结果预览:")
    for i, qa in enumerate(qa_pairs[:5]):
        print(f"\n--- 问答对 {i+1} ---")
        print(f"问题: {qa['question']}")
        print(f"答案: {qa['answer'][:100]}...")
    
    # 转换为对话格式
    print("\n正在转换为对话格式...")
    conversations = qa_to_conversation(qa_pairs)
    
    # 保存为JSONL格式
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')
    
    print(f"转换完成！共生成 {len(conversations)} 条对话数据")
    
    # 显示样本数据
    print("\n样本数据预览:")
    for i, conv in enumerate(conversations[:3]):
        print(f"\n--- 样本 {i+1} ---")
        for msg in conv['conversations']:
            role = msg['role']
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"{role}: {content}")

def main():
    """主函数"""
    
    # 文件路径
    input_file = "ai训练数据 - 副本.txt"
    output_file = "training_data/sft/sft_data.jsonl"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    try:
        convert_text_to_sft_smart(input_file, output_file)
        print("\n SFT数据转换成功完成！")
        print(f"输出文件: {output_file}")
        print("\n您现在可以使用这个文件进行SFT微调训练。")
        
    except Exception as e:
        print(f"\n 转换过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()