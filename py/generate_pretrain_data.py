"""
生成预训练数据的脚本

基于任务1、2、4生成预训练数据：
- 任务1：ID → 描述（描述性文本，非问答格式）
- 任务2：描述 → ID（描述性文本，非问答格式）
- 任务4：序列预测（不使用滑窗，序列<50直接用，>50切分成多段，每段50个）
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import polars as pl
from tqdm import tqdm


def extract_fields(s: str) -> Dict[str, Optional[str]]:
    """
    从clean_combine字段提取Category和Plot
    
    输入：包含Category和Plot字段的字符串
    输出：包含category和plot键的字典
    """
    fields = ["Category", "Plot"]
    result = {}

    text = s or ""

    for i, field in enumerate(fields):
        if i < len(fields) - 1:
            next_field = fields[i + 1]
            pattern = rf"{field}:(.*?)(?=\n{next_field}:|\Z)"
        else:
            pattern = rf"{field}:(.*)"

        m = re.search(pattern, text, flags=re.DOTALL)
        value = m.group(1).strip() if m else None
        result[field.lower()] = value if value else None

    return result


def load_data(data_dir: Path, category: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    加载商品数据和用户序列数据
    
    输入：
        data_dir: 数据目录路径
        category: 商品类别
    输出：
        items_df: 包含商品信息（语义ID、标题、类别、简介）的DataFrame
        sequences_df: 包含用户序列（语义ID序列）的DataFrame
    """
    print("加载数据文件...")
    
    # 加载items数据（用于获取title）
    items_df_cold = pl.read_parquet(data_dir / "output" / f"{category}_cold_items.parquet")
    items_df_hot = pl.read_parquet(data_dir / "output" / f"{category}_hot_items.parquet")
    items_df = pl.concat([items_df_cold, items_df_hot])
    
    # 加载clean_combine数据（包含category和plot）
    clean_combine_df = pl.read_csv(data_dir / "output" / f"{category}_combine_clean.csv")
    
    # 加载语义ID映射
    semantic_ids_df = pl.read_parquet(data_dir / "output" / f"{category}_semantic_ids.parquet")
    
    # 合并所有数据：语义ID + title + clean_combine
    items_df = semantic_ids_df.join(
        items_df.select(["parent_asin", "title"]), 
        on="parent_asin", 
        how="left"
    )
    
    # 提取category和plot
    items_df = items_df.with_columns(
        pl.col("clean_combine")
        .map_elements(
            extract_fields,
            return_dtype=pl.Struct([
                pl.Field("category", pl.Utf8),
                pl.Field("plot", pl.Utf8),
            ])
        )
        .struct.unnest()
    )
    
    # 加载序列数据
    sequences_df = pl.read_parquet(
        data_dir / "output" / f"{category}_sequences_with_semantic_ids.parquet"
    )
    
    # Leave-one-out: 去掉每个序列的最后两个元素（用于预训练数据）
    def remove_last_two(seq):
        """去掉序列的最后两个元素"""
        if len(seq) <= 2:
            return []  # 如果序列长度<=2，去掉最后两个后为空
        return seq[:-2]
    
    # 应用函数去掉最后两个元素
    sequences_df = sequences_df.with_columns(
        pl.col("semantic_id_sequence")
        .map_elements(remove_last_two, return_dtype=pl.List(pl.Utf8))
    )
    
    # 过滤掉序列长度为0的行（去掉最后两个后没有数据了）
    sequences_df = sequences_df.filter(
        pl.col("semantic_id_sequence").list.len() > 0
    )
    
    print(f"加载了 {len(items_df):,} 个items")
    print(f"加载了 {len(sequences_df):,} 个用户序列（已去掉最后两个元素用于预训练）")
    
    return items_df, sequences_df


def task1_semantic_anchoring_pretrain(items_df: pl.DataFrame) -> List[Dict]:
    """
    任务1：语义锚定任务（ID → 语义描述）
    预训练版本：使用描述性文本，非问答格式
    
    输入：语义ID
    输出：描述性文本（语义ID对应的是...）
    """
    print("\n生成任务1：语义锚定任务（预训练版本）...")
    samples = []
    
    # 描述性文本模板（非问答格式）
    description_templates = [
        lambda sid, t, c, p: f"语义ID {sid} 对应的是《{t}》，这是一部{c}类型的作品。" + 
                            (f"故事讲述了{p[:150]}。" if p and len(p) > 10 else "") if t and c else
                            f"语义ID {sid} 对应的是《{t}》。" + (f"简介：{p[:150]}。" if p else "") if t else "",
        lambda sid, t, c, p: f"语义ID {sid} 代表商品《{t}》，属于{c}类别。" + 
                            (f"该作品的主要内容是：{p[:130]}。" if p and len(p) > 10 else "") if t and c else
                            f"语义ID {sid} 代表商品《{t}》。" if t else "",
        lambda sid, t, c, p: f"《{t}》的语义ID是{sid}，这是一部{c}类型的作品。" + 
                            (f"这部作品的剧情概要：{p[:120]}。" if p and len(p) > 10 else "") if t and c else
                            f"《{t}》的语义ID是{sid}。" if t else "",
        lambda sid, t, c, p: f"语义ID {sid} 标识的商品是《{t}》，分类为{c}。" + 
                            (f"该作品的故事情节如下：{p[:125]}。" if p and len(p) > 10 else "") if t and c else
                            f"语义ID {sid} 标识的商品是《{t}》。" if t else "",
        lambda sid, t, c, p: f"语义标识符{sid}对应《{t}》，这是一部{c}类型的作品。" + 
                            (f"作品简介：{p[:140]}。" if p and len(p) > 10 else "") if t and c else
                            f"语义标识符{sid}对应《{t}》。" if t else "",
    ]
    
    for row in tqdm(items_df.iter_rows(named=True), total=len(items_df)):
        semantic_id = row["semantic_id"]
        title = row.get("title", "")
        category = row.get("category", "")
        plot = row.get("plot", "")
        
        # 跳过缺失关键信息的item
        if not title or not semantic_id:
            continue
        
        # 随机选择一个描述模板
        template = random.choice(description_templates)
        text = template(semantic_id, title, category, plot).strip()
        
        if not text:
            continue
        
        samples.append({
            "task": "semantic_anchoring_pretrain",
            "text": text
        })
    
    print(f"生成了 {len(samples):,} 个任务1样本")
    return samples


def task2_semantic_recovery_pretrain(items_df: pl.DataFrame) -> List[Dict]:
    """
    任务2：语义回收任务（语义 → ID）
    预训练版本：使用描述性文本，非问答格式
    
    输入：商品描述（Title, Category, Plot）
    输出：描述性文本（...的语义ID是...）
    """
    print("\n生成任务2：语义回收任务（预训练版本）...")
    samples = []
    
    # 描述性文本模板（非问答格式）
    description_templates = [
        lambda sid, t, c, p: f"《{t}》是一部{c}类型的作品，它的语义ID是{sid}。" + 
                            (f"故事讲述了{p[:150]}。" if p and len(p) > 10 else "") if t and c else
                            f"《{t}》的语义ID是{sid}。" + (f"简介：{p[:150]}。" if p else "") if t else "",
        lambda sid, t, c, p: f"商品《{t}》属于{c}类别，对应的语义ID为{sid}。" + 
                            (f"该作品的主要内容是：{p[:130]}。" if p and len(p) > 10 else "") if t and c else
                            f"商品《{t}》对应的语义ID为{sid}。" if t else "",
        lambda sid, t, c, p: f"《{t}》的语义标识符是{sid}，这是一部{c}类型的作品。" + 
                            (f"这部作品的剧情概要：{p[:120]}。" if p and len(p) > 10 else "") if t and c else
                            f"《{t}》的语义标识符是{sid}。" if t else "",
        lambda sid, t, c, p: f"商品名称是《{t}》，类别为{c}，语义ID是{sid}。" + 
                            (f"该作品的故事情节如下：{p[:125]}。" if p and len(p) > 10 else "") if t and c else
                            f"商品名称是《{t}》，语义ID是{sid}。" if t else "",
        lambda sid, t, c, p: f"《{t}》是一部{c}类型的作品，语义ID为{sid}。" + 
                            (f"作品简介：{p[:140]}。" if p and len(p) > 10 else "") if t and c else
                            f"《{t}》的语义ID为{sid}。" if t else "",
    ]
    
    for row in tqdm(items_df.iter_rows(named=True), total=len(items_df)):
        semantic_id = row["semantic_id"]
        title = row.get("title", "")
        category = row.get("category", "")
        plot = row.get("plot", "")
        
        # 跳过缺失关键信息的item
        if not title or not semantic_id:
            continue
        
        # 随机选择一个描述模板
        template = random.choice(description_templates)
        text = template(semantic_id, title, category, plot).strip()
        
        if not text:
            continue
        
        samples.append({
            "task": "semantic_recovery_pretrain",
            "text": text
        })
    
    print(f"生成了 {len(samples):,} 个任务2样本")
    return samples


def task4_sequence_prediction_pretrain(sequences_df: pl.DataFrame, segment_size: int = 50) -> List[Dict]:
    """
    任务4：序列预测任务（预训练版本）
    不使用滑窗，如果序列小于segment_size直接用，超过segment_size就切分成多段
    
    输入：用户历史序列
    输出：描述性文本（用户最近的浏览序列是...）
    """
    print("\n生成任务4：序列预测任务（预训练版本）...")
    samples = []
    
    # 描述性文本模板（非问答格式，直接描述整个序列）
    description_templates = [
        lambda seq: f"用户最近的浏览序列是 {', '.join(seq)}。",
        lambda seq: f"用户的浏览历史是 {', '.join(seq)}。",
        lambda seq: f"用户依次浏览了以下商品：{', '.join(seq)}。",
        lambda seq: f"用户的浏览行为序列为 {', '.join(seq)}。",
        lambda seq: f"用户浏览了商品序列 {', '.join(seq)}。",
    ]
    
    for row in tqdm(sequences_df.iter_rows(named=True), total=len(sequences_df)):
        semantic_id_sequence = row["semantic_id_sequence"]
        
        # 需要至少1个item才能生成描述
        if len(semantic_id_sequence) < 1:
            continue
        
        # 如果序列长度 <= segment_size，直接使用整个序列
        if len(semantic_id_sequence) <= segment_size:
            # 使用整个序列
            seq = semantic_id_sequence
            
            # 随机选择一个描述模板
            template = random.choice(description_templates)
            text = template(seq).strip()
            
            samples.append({
                "task": "sequence_prediction_pretrain",
                "text": text
            })
        else:
            # 如果序列长度 > segment_size，切分成多段
            # 每段segment_size个元素，最后一段可能不足segment_size
            num_segments = (len(semantic_id_sequence) + segment_size - 1) // segment_size
            
            for seg_idx in range(num_segments):
                start_idx = seg_idx * segment_size
                end_idx = min(start_idx + segment_size, len(semantic_id_sequence))
                
                segment = semantic_id_sequence[start_idx:end_idx]
                
                # 如果段长度为0，跳过
                if len(segment) == 0:
                    continue
                
                # 使用整个段
                seq = segment
                
                # 随机选择一个描述模板
                template = random.choice(description_templates)
                text = template(seq).strip()
                
                samples.append({
                    "task": "sequence_prediction_pretrain",
                    "text": text
                })
    
    print(f"生成了 {len(samples):,} 个任务4样本")
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
    OUTPUT_DIR = DATA_DIR / "output" / "pretrain_data"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    SEGMENT_SIZE = 50  # 序列切分大小
    
    # 加载数据
    items_df, sequences_df = load_data(DATA_DIR, CATEGORY)
    
    # 生成预训练数据
    all_samples = []
    
    # 任务1：语义锚定（ID → 描述）
    all_samples.extend(task1_semantic_anchoring_pretrain(items_df))
    
    # 任务2：语义回收（描述 → ID）
    all_samples.extend(task2_semantic_recovery_pretrain(items_df))
    
    # 任务4：序列预测（不使用滑窗，切分处理）
    all_samples.extend(task4_sequence_prediction_pretrain(sequences_df, segment_size=SEGMENT_SIZE))
    
    # 保存数据
    save_samples(all_samples, OUTPUT_DIR / "pretrain_data.jsonl")
    
    # 打印统计信息
    print("\n" + "="*50)
    print("预训练数据生成完成！统计信息：")
    print("="*50)
    task_counts = {}
    for sample in all_samples:
        task = sample["task"]
        task_counts[task] = task_counts.get(task, 0) + 1
    
    for task, count in sorted(task_counts.items()):
        print(f"{task}: {count:,} 个样本")
    print(f"\n总计: {len(all_samples):,} 个样本")
    print(f"\n数据保存在: {OUTPUT_DIR / 'pretrain_data.jsonl'}")


if __name__ == "__main__":
    main()

