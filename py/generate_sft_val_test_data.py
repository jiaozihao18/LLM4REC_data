"""
生成验证集和测试集数据的脚本

使用sequence_prediction任务：
- val: 使用序列的最后-2位置（倒数第二个）作为预测目标
- test: 使用序列的最后-1位置（最后一个）作为预测目标
- 最大历史长度限制为50，超过则截取最后50个
"""

import json
import random
from pathlib import Path
from typing import List, Dict

import polars as pl
from tqdm import tqdm


def load_sequences(data_dir: Path, category: str) -> pl.DataFrame:
    """
    加载用户序列数据（不截断，保留完整序列）
    
    输入：
        data_dir: 数据目录路径
        category: 商品类别
    输出：
        sequences_df: 包含用户序列（语义ID序列）的DataFrame
    """
    print("加载序列数据文件...")
    
    # 加载序列数据（完整序列，不截断）
    sequences_df = pl.read_parquet(
        data_dir / "output" / f"{category}_sequences_with_semantic_ids.parquet"
    )
    
    print(f"加载了 {len(sequences_df):,} 个用户序列")
    
    return sequences_df


def generate_sequence_prediction_samples(
    sequences_df: pl.DataFrame,
    split: str,
    max_history_len: int = 50
) -> List[Dict]:
    """
    生成序列预测任务的样本
    
    输入：
        sequences_df: 序列数据
        split: 'val' 或 'test'
        max_history_len: 最大历史长度限制
    输出：
        样本列表
    """
    print(f"\n生成{split.upper()}数据集：序列预测任务...")
    samples = []
    
    # 输入模板（与训练集保持一致）
    input_templates = [
        lambda history: f"根据以下ID序列，请预测下一个应该是什么：\n{', '.join(history)}",
        lambda history: f"历史序列如下：{', '.join(history)}，请补全下一个ID。",
        lambda history: f"已知行为ID序列：{', '.join(history)}，请推测接下来最有可能出现的ID。",
        lambda history: f"考虑到以下的行为序列：\n{', '.join(history)}\n请问下一个ID是什么？",
        lambda history: f"请根据前面的ID序列预测接下来的ID：{', '.join(history)}",
        lambda history: f"用户已经依次操作了：{', '.join(history)}，请输出可能的下一个ID。",
        lambda history: f"请补写下一个合理的行为ID，前序列为：{', '.join(history)}",
        lambda history: f"基于用户的历史行为模式（{', '.join(history)}），请预测下一个最可能出现的ID。",
        lambda history: f"用户浏览序列：{', '.join(history)}，请预测下一个商品ID。"
    ]
    
    for row in tqdm(sequences_df.iter_rows(named=True), total=len(sequences_df)):
        semantic_id_sequence = row["semantic_id_sequence"]
        
        # 根据split确定目标位置
        if split == "val":
            # val: 使用最后-2位置（倒数第二个）
            if len(semantic_id_sequence) < 3:  # 至少需要3个：历史+预测目标
                continue
            target_idx = len(semantic_id_sequence) - 2
        elif split == "test":
            # test: 使用最后-1位置（最后一个）
            if len(semantic_id_sequence) < 2:  # 至少需要2个：历史+预测目标
                continue
            target_idx = len(semantic_id_sequence) - 1
        else:
            raise ValueError(f"split必须是'val'或'test'，当前为: {split}")
        
        # 提取历史序列（目标位置之前的所有元素）
        history = semantic_id_sequence[:target_idx]
        next_sid = semantic_id_sequence[target_idx]
        
        # 限制历史长度，如果超过max_history_len，截取最后max_history_len个
        if len(history) > max_history_len:
            history = history[-max_history_len:]
        
        # 随机选择自然语言prompt
        instruction = random.choice(input_templates)(history)
        output = next_sid
        
        samples.append({
            "task": "sequence_prediction",
            "split": split,
            "instruction": instruction.strip(),
            "output": output
        })
    
    print(f"生成了 {len(samples):,} 个{split.upper()}样本")
    return samples


def save_samples(samples: List[Dict], output_path: Path):
    """保存样本到JSONL文件"""
    print(f"\n保存 {len(samples):,} 个样本到 {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    # 配置
    DATA_DIR = Path("/home/zihao/llm/llm4rec/data")
    CATEGORY = "Movies_and_TV"
    OUTPUT_DIR = DATA_DIR / "output" / "sft_data"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    MAX_HISTORY_LEN = 50  # 最大历史长度限制
    
    # 加载序列数据（完整序列，不截断）
    sequences_df = load_sequences(DATA_DIR, CATEGORY)
    
    # 生成验证集（val: 最后-2位置）
    val_samples = generate_sequence_prediction_samples(
        sequences_df, 
        split="val",
        max_history_len=MAX_HISTORY_LEN
    )
    save_samples(val_samples, OUTPUT_DIR / "val_sequence_prediction.jsonl")
    
    # 生成测试集（test: 最后-1位置）
    test_samples = generate_sequence_prediction_samples(
        sequences_df,
        split="test",
        max_history_len=MAX_HISTORY_LEN
    )
    save_samples(test_samples, OUTPUT_DIR / "test_sequence_prediction.jsonl")
    
    # 打印统计信息
    print("\n" + "="*50)
    print("数据生成完成！统计信息：")
    print("="*50)
    print(f"验证集（val）: {len(val_samples):,} 个样本")
    print(f"测试集（test）: {len(test_samples):,} 个样本")
    print(f"\n数据保存在: {OUTPUT_DIR}")
    print(f"- 验证集: {OUTPUT_DIR / 'sft_val_sequence_prediction.jsonl'}")
    print(f"- 测试集: {OUTPUT_DIR / 'sft_test_sequence_prediction.jsonl'}")


if __name__ == "__main__":
    main()

