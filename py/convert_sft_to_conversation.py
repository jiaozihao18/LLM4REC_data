"""
将SFT训练数据转换为通用的对话格式（conversation format）

读取 all_tasks_combined.jsonl，转换为标准的对话格式并保存为 Parquet 文件。
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import polars as pl

# 统一的系统提示词
SYSTEM_PROMPT = """你是一个专业的商品推荐助手，擅长理解和使用语义ID进行商品推荐和用户偏好分析。"""


def load_jsonl(file_path: Path) -> List[Dict]:
    """
    加载JSONL格式的数据文件
    
    输入：
        file_path: JSONL文件路径
    输出：
        样本列表
    """
    samples = []
    print(f"正在加载 {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="加载数据"):
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"警告：跳过无效的JSON行: {e}")
                    continue
    print(f"成功加载 {len(samples):,} 个样本")
    return samples


def convert_to_conversation_format(samples: List[Dict]) -> List[Dict]:
    """
    将instruction/output格式转换为对话格式
    
    输入：
        samples: 原始样本列表，包含：
            - 普通任务：{"task": str, "instruction": str, "output": str, ...}
            - 任务14：{"task": "multi_turn_dialogue_recommendation", "messages": List[Dict]}
    输出：
        转换为对话格式的样本列表，格式为：
        {
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ],
            "task": str,
            ...
        }
    """
    conversations_data = []
    
    for sample in tqdm(samples, desc="转换格式"):
        task = sample.get("task", "unknown")
        
        # 任务14是多轮对话，已经包含messages字段
        if task == "multi_turn_dialogue_recommendation" and "messages" in sample:
            # 构建完整的对话，在messages前添加system message
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            # 添加原有的messages（已经是user/assistant轮次）
            conversation.extend(sample["messages"])
            
            # 构建转换后的样本（只保留核心字段）
            converted_sample = {
                "conversations": conversation,
                "task": task
            }
            
            conversations_data.append(converted_sample)
        
        # 其他任务：instruction/output格式
        elif "instruction" in sample and "output" in sample:
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["output"]},
            ]
            
            # 构建转换后的样本（只保留核心字段）
            converted_sample = {
                "conversations": conversation,
                "task": task
            }
            
            conversations_data.append(converted_sample)
        else:
            print(f"警告：跳过格式不支持的样本（task={task}）")
            continue
    
    return conversations_data


def save_conversations_to_parquet(conversations_data: List[Dict], output_path: Path):
    """
    保存对话格式的数据到Parquet文件
    
    输入：
        conversations_data: 对话格式的样本列表
        output_path: 输出文件路径
    """
    print(f"\n正在保存 {len(conversations_data):,} 个对话样本到 {output_path}...")
    
    # 转换为 DataFrame
    df = pl.DataFrame(conversations_data)
    
    # 保存为 Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    
    print(f"保存完成！")
    print(f"  - 文件路径: {output_path}")
    print(f"  - 样本数量: {len(df):,}")
    print(f"  - 列: {', '.join(df.columns)}")


def convert_combined_file(input_path: Path, output_path: Path):
    """
    转换 all_tasks_combined.jsonl 文件
    
    输入：
        input_path: 输入JSONL文件路径
        output_path: 输出Parquet文件路径
    """
    # 加载原始数据
    samples = load_jsonl(input_path)
    
    if not samples:
        print(f"警告：{input_path} 中没有有效样本")
        return None
    
    # 转换为对话格式
    conversations_data = convert_to_conversation_format(samples)
    
    if not conversations_data:
        print(f"警告：转换后没有有效样本")
        return None
    
    # 保存为 Parquet 格式
    save_conversations_to_parquet(conversations_data, output_path)
    
    return conversations_data

def main():
    """主函数：读取 all_tasks_combined.jsonl 并转换为对话格式的 Parquet 文件"""
    import argparse
    
    parser = argparse.ArgumentParser(description="将SFT数据转换为对话格式（Parquet）")
    parser.add_argument(
        "--input-file",
        type=str,
        default="/home/zihao/llm/llm4rec/data/output/sft_data/test_sequence_prediction.jsonl",
        help="输入文件路径（all_tasks_combined.jsonl）"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/home/zihao/llm/llm4rec/data/output/sft_data/test_sequence_prediction.parquet",
        help="输出文件路径（Parquet格式）"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    
    # 检查输入文件是否存在
    if not input_path.exists():
        print(f"错误：输入文件不存在 {input_path}")
        return
    
    print(f"{'='*60}")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"{'='*60}\n")
    
    # 转换文件
    conversations_data = convert_combined_file(input_path, output_path)
    
    if conversations_data:
        # 打印统计信息
        print(f"\n{'='*60}")
        print("转换完成！统计信息：")
        print(f"{'='*60}")
        
        task_counts = {}
        for conv in conversations_data:
            task = conv.get("task", "unknown")
            task_counts[task] = task_counts.get(task, 0) + 1
        
        for task, count in sorted(task_counts.items()):
            print(f"  {task}: {count:,} 个样本")
        print(f"\n总计: {len(conversations_data):,} 个对话样本")
        
    print("\n✅ 数据转换完成！")


if __name__ == "__main__":
    main()

