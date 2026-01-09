"""
将预训练数据转换为Parquet格式（包含text列）

读取 pretrain_data.jsonl，提取text字段并保存为 Parquet 文件。
"""

import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import polars as pl


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


def convert_to_text_format(samples: List[Dict]) -> List[Dict]:
    """
    将预训练数据转换为包含text字段的格式
    
    输入：
        samples: 原始样本列表，包含 {"task": str, "text": str}
    输出：
        转换后的样本列表，格式为：
        {
            "text": str,
            "task": str (可选)
        }
    """
    text_data = []
    
    for sample in tqdm(samples, desc="转换格式"):
        # 提取text字段
        text = sample.get("text", "")
        task = sample.get("task", "")
        
        if not text:
            print(f"警告：跳过没有text字段的样本（task={task}）")
            continue
        
        # 构建转换后的样本
        converted_sample = {
            "text": text
        }
        
        # 可选：保留task字段用于统计
        if task:
            converted_sample["task"] = task
        
        text_data.append(converted_sample)
    
    return text_data


def save_to_parquet(text_data: List[Dict], output_path: Path):
    """
    保存数据到Parquet文件
    
    输入：
        text_data: 包含text字段的样本列表
        output_path: 输出文件路径
    """
    print(f"\n正在保存 {len(text_data):,} 个样本到 {output_path}...")
    
    # 转换为 DataFrame
    df = pl.DataFrame(text_data)
    
    # 保存为 Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    
    print(f"保存完成！")
    print(f"  - 文件路径: {output_path}")
    print(f"  - 样本数量: {len(df):,}")
    print(f"  - 列: {', '.join(df.columns)}")
    
    # 显示前几行示例
    if len(df) > 0:
        print(f"\n前3行示例：")
        for i in range(min(3, len(df))):
            text_preview = df["text"][i][:100] + "..." if len(df["text"][i]) > 100 else df["text"][i]
            print(f"  [{i+1}] {text_preview}")


def convert_pretrain_file(input_path: Path, output_path: Path):
    """
    转换 pretrain_data.jsonl 文件
    
    输入：
        input_path: 输入JSONL文件路径
        output_path: 输出Parquet文件路径
    """
    # 加载原始数据
    samples = load_jsonl(input_path)
    
    if not samples:
        print(f"警告：{input_path} 中没有有效样本")
        return None
    
    # 转换为text格式
    text_data = convert_to_text_format(samples)
    
    if not text_data:
        print(f"警告：转换后没有有效样本")
        return None
    
    # 保存为 Parquet 格式
    save_to_parquet(text_data, output_path)
    
    return text_data


def main():
    """主函数：读取 pretrain_data.jsonl 并转换为包含text列的 Parquet 文件"""
    import argparse
    
    parser = argparse.ArgumentParser(description="将预训练数据转换为Parquet格式（包含text列）")
    parser.add_argument(
        "--input-file",
        type=str,
        default="/home/zihao/llm/llm4rec/data/output/pretrain_data/pretrain_data.jsonl",
        help="输入文件路径（pretrain_data.jsonl）"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/home/zihao/llm/llm4rec/data/output/pretrain_data/pretrain_data.parquet",
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
    text_data = convert_pretrain_file(input_path, output_path)
    
    if text_data:
        # 打印统计信息
        print(f"\n{'='*60}")
        print("转换完成！统计信息：")
        print(f"{'='*60}")
        
        # 统计task分布（如果存在）
        task_counts = {}
        for item in text_data:
            task = item.get("task", "")
            if task:
                task_counts[task] = task_counts.get(task, 0) + 1
        
        if task_counts:
            for task, count in sorted(task_counts.items()):
                print(f"  {task}: {count:,} 个样本")
        
        print(f"\n总计: {len(text_data):,} 个文本样本")
        
    print("\n✅ 数据转换完成！")


if __name__ == "__main__":
    main()

