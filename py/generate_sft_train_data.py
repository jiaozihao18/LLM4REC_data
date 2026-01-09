"""
生成SFT训练数据的脚本

实现14个辅助任务来帮助LLM理解语义ID token，生成用于监督微调的训练样本
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

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
    
    # Leave-one-out: 去掉每个序列的最后两个元素（用于训练数据）
    def remove_last_two(seq):
        """去掉序列的最后两个元素"""
        if len(seq) <= 2:
            return []  # 如果序列长度<=2，去掉最后两个后为空
        return seq[:-2]
        return seq
    
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
    print(f"加载了 {len(sequences_df):,} 个用户序列（已去掉最后两个元素用于训练）")
    
    return items_df, sequences_df


import random

def task1_semantic_anchoring(items_df: pl.DataFrame) -> List[Dict]:
    """
    任务1：语义锚定任务（ID → 语义描述）
    输入：语义ID
    输出：Title, Category, Plot（自然语言风格、多样化Prompt设计）
    """
    print("\n生成任务1：语义锚定任务...")
    samples = []

    # 多套instruct prompt模板
    instruction_templates = [
        "请根据以下语义ID，给出它的相关描述：{semantic_id}",
        "已知以下语义ID：{semantic_id}，请提供其详细的信息。",
        "对于语义ID {semantic_id}，你能描述一下它指代的内容吗？",
        "请输出该语义ID对应的商品信息：{semantic_id}",
        "你收到一个语义ID：{semantic_id}。请帮我详细介绍相关内容。"
    ]

    # 输出模板：使用自然语言完整描述商品信息
    output_templates = [
        # 自然语言描述模板：使用完整句子描述商品信息
        lambda t, c, p: (
            f"{t}是一部{c}类型的作品。" + 
            (f"故事讲述了{p[:150]}。" if p and len(p) > 10 else "")
            if t and c else (
                f"{t}。" + (f"简介：{p[:150]}。" if p else "")
                if t else ""
            )
        ),
        lambda t, c, p: (
            f'这是"{t}"，一部{c}类型的作品。' + 
            (f"这部作品的剧情概要：{p[:120]}。" if p and len(p) > 10 else "")
            if t and c else f'这是"{t}"。' if t else ""
        ),
        lambda t, c, p: (
            f"{t}属于{c}类别。" + 
            (f"该作品的主要内容是：{p[:130]}。" if p and len(p) > 10 else "")
            if t and c else f"{t}。" if t else ""
        ),
        lambda t, c, p: (
            f'您查询的商品是"{t}"，这是一部{c}类型的作品。' + 
            (f"作品简介：{p[:140]}。" if p and len(p) > 10 else "")
            if t and c else f'您查询的商品是"{t}"。' if t else ""
        ),
        lambda t, c, p: (
            f"这个语义ID对应的是{t}，分类为{c}。" + 
            (f"该作品的故事情节如下：{p[:125]}。" if p and len(p) > 10 else "")
            if t and c else f"这个语义ID对应的是{t}。" if t else ""
        ),
        # 结构化格式模板：用于增强模型对格式化输出的理解
        lambda t, c, p: f"{t}（类别：{c}）" + (f"\n简介：{p[:100]}" if p else "") if t and c else f"{t}" if t else "",
        lambda t, c, p: (
            f"商品名称：{t}\n" +
            (f"类别：{c}\n" if c else "") +
            (f"内容简介：{p[:100]}" if p else "")
            if t else ""
        ),
    ]

    for row in tqdm(items_df.iter_rows(named=True), total=len(items_df)):
        semantic_id = row["semantic_id"]
        title = row.get("title", "")
        category = row.get("category", "")
        plot = row.get("plot", "")

        # 跳过缺失关键信息的item
        if not title or not semantic_id:
            continue

        # 随机选一个prompt模板
        instruction_tmpl = random.choice(instruction_templates)
        instruction = instruction_tmpl.format(semantic_id=semantic_id)

        # 随机选一个output模板并输出
        output_tmpl = random.choice(output_templates)
        output = output_tmpl(title, category, plot).strip()
        if not output:
            continue

        samples.append({
            "task": "semantic_anchoring",
            "instruction": instruction,
            "output": output
        })

    print(f"生成了 {len(samples):,} 个任务1样本")
    return samples


def task2_semantic_recovery(items_df: pl.DataFrame) -> List[Dict]:
    """
    任务2：语义回收任务（语义 → ID）
    输入：Title, Category, Plot
    输出：语义ID
    LLM自然语言风格的多prompt输入
    """
    print("\n生成任务2：语义回收任务...")
    samples = []

    # 构建多套prompt模板，更自然地描述输入信息
    input_templates = [
        lambda t, c, p: f"请根据以下商品信息返回唯一的语义ID：\n商品名：{t}\n类别：{c}\n简介：{p}" if t and c and p else
                        f"请根据以下商品信息返回唯一的语义ID：\n商品名：{t}\n类别：{c}" if t and c else
                        f"请根据以下商品信息返回唯一的语义ID：\n商品名：{t}\n简介：{p}" if t and p else
                        f"请根据以下商品信息返回唯一的语义ID：\n商品名：{t}",
        lambda t, c, p: f"以下商品介绍，请推断其语义ID：\nTitle: {t}\nCategory: {c}\nPlot: {p}" if t and c and p else
                        f"以下商品介绍，请推断其语义ID：\nTitle: {t}\nCategory: {c}" if t and c else
                        f"以下商品介绍，请推断其语义ID：\nTitle: {t}\nPlot: {p}" if t and p else
                        f"以下商品介绍，请推断其语义ID：\nTitle: {t}",
        lambda t, c, p: (
            f'商品的名称为"{t}"，类别属于"{c}"，简介为：{p}。\n请问它的语义ID是什么？' if t and c and p else
            f"商品名称：{t}，类别：{c}。\n请给出对应的语义ID。" if t and c else
            f'该商品叫做"{t}"，主要内容：{p}\n请返回其语义ID。' if t and p else
            f"商品名称：{t}\n请返回其语义ID。"
        ),
        lambda t, c, p: (
            f"已知商品的相关信息如下：\n- 名称：{t}\n- 分类：{c}\n- 简介：{p}\n请帮我找出它的语义ID。" if t and c and p else
            f"已知商品的相关信息如下：\n- 名称：{t}\n- 分类：{c}\n请帮我找出它的语义ID。" if t and c else
            f"已知商品的相关信息如下：\n- 名称：{t}\n- 简介：{p}\n请帮我找出它的语义ID。" if t and p else
            f"已知商品名称：{t}\n请给出对应的语义ID。"
        ),
        lambda t, c, p: (
            f"请根据商品信息（名称：{t}，类别：{c}，简介：{p}）返回其唯一的语义ID。" if t and c and p else
            f"请根据商品信息（名称：{t}，类别：{c}）返回其唯一的语义ID。" if t and c else
            f"请根据商品信息（名称：{t}，简介：{p}）返回其唯一的语义ID。" if t and p else
            f'请根据商品名称"{t}"返回其唯一的语义ID。'
        )
    ]

    # 输出模板：生成包含推理过程的输出
    def generate_output_with_reasoning(sid, title, category, plot):
        """
        生成带推理的输出
        
        输入：语义ID、标题、类别、简介
        输出：包含ID和推理说明的字符串
        """
        output_templates = [
            # 类型1：简单直接输出
            lambda: f"语义ID: {sid}",
            lambda: f"该商品的语义ID为：{sid}",
            # 类型2：带简单推理的输出
            lambda: f"根据商品信息，对应的语义ID是{sid}。",
            lambda: f"考虑到商品名称、类别和简介的匹配度，语义ID应为{sid}。",
            lambda: f"该商品的语义ID是{sid}，这是根据其基本信息确定的唯一标识符。",
            # 类型3：带详细推理的输出
            lambda: (
                f"根据提供的商品信息，语义ID是{sid}。" + 
                (f"这是因为该商品属于{category}类别，且商品名称为'{title}'。" if category and title else
                 f"这是因为商品名称为'{title}'。" if title else "")
            ) if title else f"语义ID是{sid}。",
            lambda: (
                f"语义ID: {sid}。该ID与商品的名称'{title}'" +
                (f"、类别'{category}'" if category else "") +
                (f"以及简介内容" if plot else "") +
                "相匹配。"
            ) if title else f"语义ID: {sid}",
        ]
        return random.choice(output_templates)()
    
    output_templates = [
        lambda sid, t, c, p: generate_output_with_reasoning(sid, t, c, p)
    ]

    for row in tqdm(items_df.iter_rows(named=True), total=len(items_df)):
        semantic_id = row["semantic_id"]
        title = row.get("title", "")
        category = row.get("category", "")
        plot = row.get("plot", "")

        # 跳过缺失关键信息的item
        if not title or not semantic_id:
            continue

        # 随机选一个input prompt模板
        input_tmpl = random.choice(input_templates)
        instruction = input_tmpl(title, category, plot).strip()

        # 随机选一个output 模板（现在需要传入完整信息）
        out_tmpl = random.choice(output_templates)
        output = out_tmpl(semantic_id, title, category, plot).strip()

        samples.append({
            "task": "semantic_recovery",
            "instruction": instruction,
            "output": output
        })

    print(f"生成了 {len(samples):,} 个任务2样本")
    return samples


def task3_sequence_semantic_understanding(
    sequences_df: pl.DataFrame, 
    items_df: pl.DataFrame
) -> List[Dict]:
    """
    任务3：序列语义理解任务（ID序列 → 自然语言）
    输入：用户历史序列（语义ID列表）
    输出：序列兴趣总结
    可以滑动窗口，统计频繁类别
    """
    print("\n生成任务3：序列语义理解任务...")
    samples = []
    
    # 构建语义ID到item信息的映射
    sid_to_item = {}
    for row in items_df.iter_rows(named=True):
        sid = row["semantic_id"]
        category = row.get("category", "")
        # 提前split并缓存
        categories_list = [c.strip() for c in category.split(",")] if category else []
        sid_to_item[sid] = {
            "title": row.get("title", ""),
            "category": category,
            "categories": categories_list,
            "plot": row.get("plot", ""),
            "has_plot": bool(row.get("plot", ""))
        }

    # 输入prompt模板
    input_templates = [
        lambda ids, items: f"已知一位用户最近浏览的商品ID序列如下：{ids}\n请分析该用户的兴趣偏好和内容类型倾向。",
        lambda ids, items: f"请根据以下用户历史商品ID列表（{ids}）推测用户主要的兴趣领域、偏好类别和内容风格。",
        lambda ids, items: f"用户近期感兴趣的内容ID列表：[{ids}]。\n请总结该用户的兴趣类型、内容偏好和浏览模式。",
        lambda ids, items: f"给定商品ID序列：{ids}\n这个序列反映了用户怎样的兴趣？请从类别、风格、主题等角度分析。",
        lambda ids, items: f"请阅读下方的商品ID序列，并生成该用户的兴趣类型描述：\n{ids}\n请分析用户的偏好特点和内容倾向。"
    ]
    # 或者可用带title描述的版本
    def gen_title_prompt(window, titles):
        if titles:
            title_str = "，".join(titles[:5])
            if len(titles) > 5:
                title_str += f"等{len(titles)}项"
            return f"一位用户最近浏览的商品包括：{title_str}。\n请归纳其感兴趣的主要类别或特点。"
        else:
            return None

    # 输出模板
    output_templates = [
        lambda cats, n_cats, is_narrative, cat_freq: f"该用户对{'、'.join(cats)}类别的内容表现出明显兴趣，浏览频率较高。" if cats else "",
        lambda cats, n_cats, is_narrative, cat_freq: f"兴趣主要集中在{'、'.join(cats)}类别，主题多样化，涵盖了{n_cats}个不同领域。" if cats and n_cats > 1 else "",
        lambda cats, n_cats, is_narrative, cat_freq: f"该用户的兴趣方向包括{n_cats}个不同类别，偏好丰富，内容类型多样。" if n_cats > 1 else "",
        lambda cats, n_cats, is_narrative, cat_freq: f"用户偏好故事性和情节推进型内容，倾向于有完整叙事线的作品。" if False else "",
        lambda cats, n_cats, is_narrative, cat_freq: f"用户主要喜欢{('、'.join(cats) if cats else '多样化分类')}，{'并倾向于剧情丰富、叙事性强的作品' if False else '且兴趣分布较单一，主要集中在少数类别' if n_cats==1 else '且主题多元，涵盖多个内容领域'}。",
        lambda cats, n_cats, is_narrative, cat_freq: f"从浏览历史来看，用户对{'、'.join(cats[:2]) if cats else '各类内容'}表现出持续关注，{'偏好有完整故事线的叙事型内容' if False else '内容偏好较为均衡'}。"
    ]

    window_size = 10
    min_window_len = 2
    min_count = 2
    MAX_WINDOWS = 2  # 限制每个用户的窗口数量

    for row in tqdm(sequences_df.iter_rows(named=True), total=len(sequences_df)):
        semantic_id_sequence = row["semantic_id_sequence"]
        seq_len = len(semantic_id_sequence)
        if seq_len < min_window_len:
            continue

        # 限制窗口数量：只采样固定数量窗口（避免全滑窗）
        possible_starts = max(1, seq_len - window_size + 1)
        num_windows = min(MAX_WINDOWS, possible_starts)
        if num_windows == 0:
            continue
        
        # 随机采样窗口起始位置（random已在文件顶部导入）
        if num_windows < possible_starts:
            start_indices = random.sample(range(possible_starts), num_windows)
        else:
            start_indices = list(range(possible_starts))

        for start_idx in start_indices:
            window = semantic_id_sequence[start_idx : start_idx + window_size]
            if len(window) < min_window_len:
                continue

            # 收集窗口内的item信息
            categories = []
            titles = []
            for sid in window:
                if sid in sid_to_item:
                    item = sid_to_item[sid]
                    categories.extend(item["categories"])
                    if item["title"]:
                        titles.append(item["title"])
            if not categories:
                continue

            # 统计分类
            category_counter = Counter(categories)
            frequent_categories = [cat for cat, cnt in category_counter.most_common(5) if cnt >= min_count]
            top_categories = frequent_categories[:3]

            # 格式化ID序列
            window_str = ", ".join(window[:5])
            if len(window) > 5:
                window_str += f", ... (共{len(window)}个)"

            # 随机选择输入模板
            if random.random() < 0.5 and titles:
                instruction = gen_title_prompt(window, titles)
                if instruction is None:
                    instruction = random.choice(input_templates)(window_str, titles)
            else:
                instruction = random.choice(input_templates)(window_str, titles)

            # 归纳总结输出
            cat_set = set(categories)
            n_cats = len(cat_set)
            is_narrative = False
            cat_freq_dict = dict(category_counter.most_common(5))
            
            # 连续尝试多个输出模板，取第一个非空
            output = ""
            for tmpl in random.sample(output_templates, len(output_templates)):
                out = tmpl(top_categories, n_cats, is_narrative, cat_freq_dict)
                if out:
                    output = out
                    break
            # 保证自然语言收尾
            if output and not output.endswith("。"):
                output += "。"

            if output and instruction:
                samples.append({
                    "task": "sequence_semantic_understanding",
                    "instruction": instruction.strip(),
                    "output": output.strip()
                })
    
    print(f"生成了 {len(samples):,} 个任务3样本")
    return samples


def task4_sequence_prediction(sequences_df: pl.DataFrame) -> List[Dict]:
    """
    任务4：序列预测任务（行为建模）
    输入：历史序列（前N-1个）
    输出：下一个语义ID
    """
    import random
    print("\n生成任务4：序列预测任务...")
    samples = []

    # 输入模板
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

        # 需要至少2个item才能做预测
        if len(semantic_id_sequence) < 2:
            continue

        # 为每个位置生成一个样本（滑动窗口，主要任务不采样）
        for i in range(1, len(semantic_id_sequence)):
            history = semantic_id_sequence[:i]
            next_sid = semantic_id_sequence[i]

            # 限制历史长度，避免输入过长
            if len(history) > 50:
                history = history[-50:]

            # 随机选择自然语言prompt
            instruction = random.choice(input_templates)(history)
            output = next_sid

            samples.append({
                "task": "sequence_prediction",
                "instruction": instruction.strip(),
                "output": output
            })

    print(f"生成了 {len(samples):,} 个任务4样本")
    return samples


def task5_mixed_space_alignment(
    items_df: pl.DataFrame, 
    num_negatives_per_positive: int = 2
) -> List[Dict]:
    """
    任务5：混合空间对齐任务（ID + 文本一致性判断）
    输入：语义ID + 描述（Title, Category, Plot）
    输出：是否匹配（是/否）
    """
    import random

    print("\n生成任务5：混合空间对齐任务...")
    samples = []
    
    # 多样化自然语言输入模板
    positive_input_templates = [
        lambda sid, title, category, plot: f"请判断语义ID {sid} 是否与下述描述相符。\nTitle: {title}\nCategory: {category}\nPlot: {plot}",
        lambda sid, title, category, plot: f"根据给定的ID（{sid}），判断以下信息是否是对应项：\n标题：{title}\n类别：{category}\n简介：{plot}",
        lambda sid, title, category, plot: f"语义ID: {sid}\n相关信息如下：\n- Title: {title}\n- Category: {category}\n- Plot: {plot}\n请问这些信息是否匹配？",
        lambda sid, title, category, plot: f"下方给出一个语义ID及其相关描述，请你判断它们是否一致。\nID: {sid}\nTitle: {title}\nCategory: {category}\nPlot: {plot}",
        lambda sid, title, category, plot: f"请结合以下ID及信息判断匹配情况。\nID：{sid}\n标题：{title}\n类别：{category}\n剧情：{plot}",
        lambda sid, title, category, plot: f"ID: {sid}\nTitle: {title}\n" + (f"Category: {category}\n" if category else "") + (f"Plot: {plot}\n" if plot else "") + "上述内容是否语义上一致？"
    ]
    negative_input_templates = [
        lambda sid, title, category, plot: f"请判断以下内容和ID {sid} 是否属于同一个实体。\nTitle: {title}\nCategory: {category}\nPlot: {plot}",
        lambda sid, title, category, plot: f"请分析该描述与语义ID {sid} 是否相匹配。\n标题：{title}\n类别：{category}\n简介：{plot}",
        lambda sid, title, category, plot: f"ID为{sid}，以下信息是否对应？\nTitle: {title}\nCategory: {category}\nPlot: {plot}",
        lambda sid, title, category, plot: f"有如下语义ID与描述：\nID：{sid}\nTitle：{title}\nCategory：{category}\nPlot：{plot}\n请问它们内容是否一致？",
        lambda sid, title, category, plot: f"请判断语义ID {sid} 的真实描述是否下列内容。\nTitle: {title}\nCategory: {category}\nPlot: {plot}"
    ]
    
    # 准备所有items的列表用于生成负样本
    all_items = list(items_df.iter_rows(named=True))
    items_by_category = {}
    
    for item in all_items:
        category = item.get("category", "")
        if category:
            # 使用第一个类别作为key
            first_cat = category.split(",")[0].strip() if category else ""
            if first_cat:
                if first_cat not in items_by_category:
                    items_by_category[first_cat] = []
                items_by_category[first_cat].append(item)
    
    for row in tqdm(items_df.iter_rows(named=True), total=len(items_df)):
        semantic_id = row["semantic_id"]
        title = row.get("title", "")
        category = row.get("category", "")
        plot = row.get("plot", "")
        
        if not title or not semantic_id:
            continue
        
        # 正样本：匹配的ID和描述，随机自然语言模板
        instruction = random.choice(positive_input_templates)(
            semantic_id, title, category if category else "", plot if plot else ""
        )
        samples.append({
            "task": "mixed_space_alignment",
            "instruction": instruction,
            "output": "是",
            "label": 1
        })
        
        # 负样本生成：使用多种策略生成负样本
        num_negatives = max(1, num_negatives_per_positive // 2)
        
        if len(all_items) > 1:
            negative_strategies = []
            
            # 策略1：只替换plot
            if plot:
                negative_item = random.choice(all_items)
                if negative_item.get("parent_asin") != row.get("parent_asin"):
                    neg_plot = negative_item.get("plot", "")
                    if neg_plot:
                        negative_strategies.append(
                            ("plot_only", lambda: (
                                semantic_id, title, category if category else "", neg_plot
                            ))
                        )
            
            # 策略2：只替换title
            negative_item = random.choice(all_items)
            if negative_item.get("parent_asin") != row.get("parent_asin"):
                neg_title = negative_item.get("title", "")
                if neg_title and neg_title != title:
                    negative_strategies.append(
                        ("title_only", lambda: (
                            semantic_id, neg_title, category if category else "", plot if plot else ""
                        ))
                    )
            
            # 策略3：只替换category
            if category:
                negative_item = random.choice(all_items)
                if negative_item.get("parent_asin") != row.get("parent_asin"):
                    neg_category = negative_item.get("category", "")
                    if neg_category and neg_category != category:
                        negative_strategies.append(
                            ("category_only", lambda: (
                                semantic_id, title, neg_category, plot if plot else ""
                            ))
                        )
            
            # 策略4：同时替换title和category（保持plot）
            if plot and category:
                negative_item = random.choice(all_items)
                if negative_item.get("parent_asin") != row.get("parent_asin"):
                    neg_title = negative_item.get("title", "")
                    neg_category = negative_item.get("category", "")
                    if neg_title and neg_category and (neg_title != title or neg_category != category):
                        negative_strategies.append(
                            ("title_category", lambda: (
                                semantic_id, neg_title, neg_category, plot
                            ))
                        )
            
            # 策略5：hard negative - 同类别的不同商品
            if category:
                first_cat = category.split(",")[0].strip() if category else ""
                if first_cat and first_cat in items_by_category:
                    pool = [item for item in items_by_category[first_cat] 
                           if item.get("parent_asin") != row.get("parent_asin")]
                    if pool:
                        hard_neg_item = random.choice(pool)
                        hard_neg_title = hard_neg_item.get("title", "")
                        hard_neg_plot = hard_neg_item.get("plot", "")
                        if hard_neg_title:
                            negative_strategies.append(
                                ("hard_negative", lambda: (
                                    semantic_id, hard_neg_title, category, hard_neg_plot if hard_neg_plot else ""
                                ))
                            )
            
            # 从可用策略中随机选择并生成负样本
            if negative_strategies:
                selected_strategies = random.sample(
                    negative_strategies, 
                    min(num_negatives, len(negative_strategies))
                )
                
                for strategy_name, strategy_func in selected_strategies:
                    try:
                        neg_sid, neg_t, neg_c, neg_p = strategy_func()
                        neg_instruction = random.choice(negative_input_templates)(
                            neg_sid, neg_t, neg_c, neg_p
                        )
                        samples.append({
                            "task": "mixed_space_alignment",
                            "instruction": neg_instruction,
                            "output": "否",
                            "label": 0,
                            "negative_strategy": strategy_name
                        })
                    except Exception:
                        continue  # 策略执行失败时跳过，继续下一个策略
    
    print(f"生成了 {len(samples):,} 个任务5样本")
    return samples

def task6_co_purchase_analysis(sequences_df: pl.DataFrame, items_df: pl.DataFrame):
    """
    任务6：共同购买预测（Co-purchase Prediction）
    
    输入：用户序列数据、商品数据
    输出：判断两个商品是否经常被用户一起购买的样本
    """
    from collections import Counter
    import random

    # 构建语义ID到item信息的映射
    sid_to_item = {}
    for row in items_df.iter_rows(named=True):
        sid = row["semantic_id"]
        sid_to_item[sid] = {
            "title": row.get("title", ""),
            "parent_asin": row.get("parent_asin", "")
        }

    # 统计商品对在用户序列中共同出现的次数
    MAX_SEQ_LEN = 20
    WINDOW_SIZE = 3  # 只统计窗口内的共现

    co_purchase_counts = Counter()
    for row in sequences_df.iter_rows(named=True):
        semantic_id_sequence = row.get("semantic_id_sequence", [])
        sids = [sid for sid in semantic_id_sequence[-MAX_SEQ_LEN:] if sid in sid_to_item]
        if len(sids) < 2:
            continue
        for i in range(len(sids)):
            for j in range(i+1, min(i+WINDOW_SIZE, len(sids))):
                pair = tuple(sorted([sids[i], sids[j]]))
                co_purchase_counts[pair] += 1

    # 共同购买阈值
    MIN_CO_PURCHASE = 3
    common_pairs = [
        (sid1, sid2, count)
        for (sid1, sid2), count in co_purchase_counts.items()
        if count >= MIN_CO_PURCHASE
    ]

    def random_title_or_sid(title: str, sid: str) -> str:
        if not title:
            return sid
        return title if random.random() < 0.5 else sid

    positive_prompts = [
        lambda t1, t2, sid1, sid2: f"{random_title_or_sid(t1,sid1)}和{random_title_or_sid(t2,sid2)}是用户经常一起买的商品吗？",
        lambda t1, t2, sid1, sid2: f"很多用户会同时购买{random_title_or_sid(t1,sid1)}和{random_title_or_sid(t2,sid2)}吗？",
        lambda t1, t2, sid1, sid2: f"请判断{random_title_or_sid(t1,sid1)}和{random_title_or_sid(t2,sid2)}是否属于常见的联合购买商品。",
        lambda t1, t2, sid1, sid2: f"在用户的购买记录中，{random_title_or_sid(t1,sid1)}和{random_title_or_sid(t2,sid2)}会频繁同时出现吗？",
        lambda t1, t2, sid1, sid2: f"{random_title_or_sid(t1,sid1)}和{random_title_or_sid(t2,sid2)}有没有可能被用户一起放进购物车？",
        lambda t1, t2, sid1, sid2: f"是否存在大量用户同时对{random_title_or_sid(t1,sid1)}和{random_title_or_sid(t2,sid2)}感兴趣并购入？"
    ]
    negative_prompts = [
        lambda t1, t2, sid1, sid2: f"{random_title_or_sid(t1,sid1)}和{random_title_or_sid(t2,sid2)}几乎不会被用户同时购买，对吗？",
        lambda t1, t2, sid1, sid2: f"这两件商品{random_title_or_sid(t1,sid1)}和{random_title_or_sid(t2,sid2)}不是很常见的搭配购买选择吧？",
        lambda t1, t2, sid1, sid2: f"在用户的购买历史中，{random_title_or_sid(t1,sid1)}和{random_title_or_sid(t2,sid2)}一起出现的情况很少，对吗？",
        lambda t1, t2, sid1, sid2: f"{random_title_or_sid(t1,sid1)}和{random_title_or_sid(t2,sid2)}经常同时被买入的说法不准确吧？",
        lambda t1, t2, sid1, sid2: f"会有很多用户把{random_title_or_sid(t1,sid1)}和{random_title_or_sid(t2,sid2)}一起买下吗？请判断。",
        lambda t1, t2, sid1, sid2: f"{random_title_or_sid(t1,sid1)}和{random_title_or_sid(t2,sid2)}通常不会被放在同一笔订单中，是吗？"
    ]

    samples = []
    # 正样本生成
    for sid1, sid2, count in common_pairs:
        item1 = sid_to_item[sid1]
        item2 = sid_to_item[sid2]
        prompt_func = random.choice(positive_prompts)
        instruction = prompt_func(item1.get('title', ''), item2.get('title', ''), sid1, sid2)
        samples.append({
            "task": "co_purchase_prediction",
            "instruction": instruction,
            "output": "是",
            "sid1": sid1,
            "sid2": sid2,
            "label": 1
        })

    # 负样本生成
    all_sids = list(sid_to_item.keys())
    negative_pairs = set()
    need_neg = len(common_pairs)
    attempts = 0
    max_attempts = need_neg * 10  # 避免极端死循环

    while len(negative_pairs) < need_neg and len(all_sids) >= 2 and attempts < max_attempts:
        sid1, sid2 = random.sample(all_sids, 2)
        pair = tuple(sorted([sid1, sid2]))
        attempts += 1
        if pair not in co_purchase_counts and pair not in negative_pairs:
            negative_pairs.add(pair)
            item1 = sid_to_item[sid1]
            item2 = sid_to_item[sid2]
            prompt_func = random.choice(negative_prompts)
            instruction = prompt_func(item1.get('title', ''), item2.get('title', ''), sid1, sid2)
            samples.append({
                "task": "co_purchase_prediction",
                "instruction": instruction,
                "output": "否",
                "sid1": sid1,
                "sid2": sid2,
                "label": 0
            })

    print(f"生成了 {len(samples):,} 个任务6（共同购买）样本")
    return samples


def task7_recommendation_reasoning(
    sequences_df: pl.DataFrame,
    items_df: pl.DataFrame
) -> List[Dict]:
    """
    任务7：推荐理由生成（Reasoning over Recommendation）
    
    输入：用户历史序列、推荐商品
    输出：推荐理由解释（包含类别匹配和plot关键词分析）
    
    使用TF-IDF提取plot关键词，预计算并缓存以提升效率
    """
    import random
    from collections import Counter
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("\n生成任务7：推荐理由生成（含 TF-IDF plot 关键词）...")
    samples = []
    
    # 创建语义ID到item信息的映射
    sid_to_item = {}
    for row in items_df.iter_rows(named=True):
        sid = row["semantic_id"]
        category = row.get("category", "")
        sid_to_item[sid] = {
            "title": row.get("title", ""),
            "category": category,
            "categories": [c.strip() for c in category.split(",")] if category else [],  # 预处理
            "plot": row.get("plot", "")
        }
    
    # 预处理：使用TF-IDF分析所有plot，提取关键词
    plots = [
        row.get("plot", "")
        for row in items_df.iter_rows(named=True)
        if row.get("plot")
    ]
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )
    if plots:
        tfidf_matrix = vectorizer.fit_transform(plots)
        feature_names = vectorizer.get_feature_names_out()
    else:
        tfidf_matrix = None
        feature_names = []

    # TF-IDF关键词抽取函数（使用稀疏矩阵）
    def extract_tfidf_keywords_sparse(plot, vectorizer, feature_names, top_k=5):
        """
        从plot文本中提取TF-IDF关键词
        
        输入：plot文本、vectorizer、特征名、top_k
        输出：关键词列表
        """
        if not plot or not hasattr(vectorizer, "transform"):
            return []
        vec = vectorizer.transform([plot])
        if vec.nnz == 0:
            return []
        scores = vec.data
        indices = vec.indices
        if len(scores) == 0:
            return []
        if len(scores) <= top_k:
            top_idx = range(len(scores))
        else:
            top_idx = scores.argsort()[-top_k:][::-1]
        return [feature_names[indices[i]] for i in top_idx]

    # 预计算并缓存：每个sid只transform一遍plot
    print("预计算 plot 的 TF-IDF 关键词...")
    sid_to_plot_keywords = {}
    sid_to_plot_counter = {}
    for row in items_df.iter_rows(named=True):
        sid = row["semantic_id"]
        plot = row.get("plot", "")
        if plot:
            kws = extract_tfidf_keywords_sparse(plot, vectorizer, feature_names, top_k=3)
            sid_to_plot_keywords[sid] = kws
            sid_to_plot_counter[sid] = Counter(kws)
        else:
            sid_to_plot_keywords[sid] = []
            sid_to_plot_counter[sid] = Counter()

    # 输入模板
    input_templates = [
        lambda history, rec, rec_title: f"用户历史浏览记录：{', '.join(history[:5])}{'...' if len(history) > 5 else ''}\n推荐商品：{rec_title} ({rec})\n请解释为什么推荐这个商品给该用户。",
        lambda history, rec, rec_title: f"根据用户的历史行为序列（{', '.join(history[:5])}{'...' if len(history) > 5 else ''}），我们推荐了 {rec_title}。请说明这个推荐的理由。",
        lambda history, rec, rec_title: f"用户之前浏览过：{', '.join(history[:5])}{'...' if len(history) > 5 else ''}\n现在推荐：{rec_title}\n请给出推荐原因。",
        lambda history, rec, rec_title: f"已知用户的历史序列：{', '.join(history[:5])}{'...' if len(history) > 5 else ''}\n推荐商品：{rec_title} ({rec})\n请解释这个推荐是基于什么考虑的？",
        lambda history, rec, rec_title: f"用户历史：{', '.join(history[:5])}{'...' if len(history) > 5 else ''}\n推荐：{rec_title}\n为什么这个商品适合该用户？请说明理由。",
        lambda history, rec, rec_title: f"基于用户浏览历史（{', '.join(history[:5])}{'...' if len(history) > 5 else ''}），推荐了 {rec_title}。请解释推荐依据。"
    ]

    # 输出模板：生成推荐理由，可包含plot主题关键词
    def generate_reasoning(
        history_cats, rec_cat, rec_title, common_cats,
        cat_freq=None, common_plot_themes=None, history_plot_counter=None, rec_plot_keywords=None
    ):
        freq_info = ""
        if cat_freq and common_cats:
            top_freq = cat_freq.get(common_cats[0], 0)
            if top_freq > 1:
                freq_info = f"（在历史中出现{top_freq}次）"

        # plot主题构建
        plot_tmpl = ""
        if isinstance(common_plot_themes, list) and common_plot_themes:
            plot_group = "、".join(common_plot_themes)
            plot_tmpl = f"在情节关键词方面，用户历史作品和该商品均涉及{plot_group}等主题，体现出相似的兴趣倾向。"
        elif rec_plot_keywords:
            plot_kws = "、".join(rec_plot_keywords)
            if plot_kws:
                plot_tmpl = f"该商品的情节关键词包括{plot_kws}。"

        # 理由生成模板：使用多种句式表达
        templates = [
            # 句式1：对比句式
            lambda: (
                f"相比其他类型，用户对{common_cats}表现出更明显的偏好{freq_info}。"
                + (f" {plot_tmpl}" if plot_tmpl else "") +
                f"因此推荐{rec_title}，它在类别和风格上都与用户兴趣高度匹配。"
            ),
            # 句式2：因果推理
            lambda: (
                f"由于用户长期关注{common_cats}类内容{freq_info}，推荐{rec_title}能够延续这种兴趣。"
                + (f" {plot_tmpl}" if plot_tmpl else "") +
                "该商品在主题和风格上与用户历史偏好一致。"
            ),
            # 句式3：数据支撑
            lambda: (
                f"在用户的历史浏览中，{common_cats}类型占比很高{freq_info}，表明用户对该类型有强烈兴趣。"
                + (f" {plot_tmpl}" if plot_tmpl else "") +
                f"推荐{rec_title}正是基于这一偏好分析。"
            ),
            # 句式4：直接推荐
            lambda: (
                f"推荐{rec_title}，因为用户历史多次浏览{common_cats}类型的内容{freq_info}，"
                + f"而{rec_title}属于类似类型。" +
                (f" {plot_tmpl}" if plot_tmpl else "") +
                "在题材和风格上与用户兴趣高度相似。"
            ),
            # 句式5：偏好分析
            lambda: (
                f"根据用户历史偏好分析，{common_cats}类别是其最常关注的内容类型{freq_info}。"
                + (f" {plot_tmpl}" if plot_tmpl else "") +
                f"推荐{rec_title}是因为它与用户历史兴趣和情节关注点高度匹配。"
            ),
            # 句式6：一致性说明
            lambda: (
                f"用户此前主要关注{common_cats}相关内容{freq_info}，"
                + f"{rec_title}在分类和风格上与历史偏好一致。" +
                (f" {plot_tmpl}" if plot_tmpl else "") +
                "因此适合推荐。"
            ),
            # 持续兴趣
            lambda: (
                f"基于用户对{common_cats}类内容的持续兴趣{freq_info}，推荐了同类的{rec_title}。"
                + (f" {plot_tmpl}" if plot_tmpl else "") +
                f"{rec_title}在主题和风格上与用户浏览模式高度契合。"
            ),
            # 兴趣关联
            lambda: (
                f"用户历史记录显示对{common_cats}类别表现明显偏好{freq_info}，{rec_title}属于同类内容。"
                + (f" {plot_tmpl}" if plot_tmpl else "") +
                "在内容与情节上均与用户兴趣关联密切。"
            ),
            # 句式8：模式匹配 - 强调符合用户模式
            lambda: (
                f"考虑到用户多次浏览{common_cats}类型的内容{freq_info}，推荐{rec_title}。"
                + (f" {plot_tmpl}" if plot_tmpl else "") +
                "该推荐符合用户的兴趣模式。"
            ),
            # 句式9：主题契合
            lambda: (
                f"从用户浏览历史来看，{common_cats}是用户最感兴趣的主题{freq_info}。"
                + (f" {plot_tmpl}" if plot_tmpl else "") +
                f"推荐{rec_title}是因为它在主题上与用户偏好高度契合。"
            ),
        ]
        return random.choice(templates)()
    
    for row in tqdm(sequences_df.iter_rows(named=True), total=len(sequences_df)):
        semantic_id_sequence = row["semantic_id_sequence"]

        if len(semantic_id_sequence) < 2:
            continue

        for i in range(1, len(semantic_id_sequence)):
            if random.random() > 0.3:
                continue

            history = semantic_id_sequence[:i]
            rec_sid = semantic_id_sequence[i]

            if len(history) > 10:
                history = history[-10:]
            if rec_sid not in sid_to_item:
                continue

            # 类别交集
            history_categories = []
            for sid in history:
                if sid in sid_to_item:
                    history_categories.extend(sid_to_item[sid].get("categories", []))
            rec_item = sid_to_item[rec_sid]
            rec_category = rec_item.get("category", "")
            rec_title = rec_item.get("title", "")

            if not rec_category or not history_categories:
                continue
            rec_cats = sid_to_item[rec_sid].get("categories", [])
            category_counter = Counter(history_categories)
            common_cats = [cat for cat in rec_cats if cat in category_counter]

            if not common_cats:
                top_cats = [cat for cat, _ in category_counter.most_common(2)]
                common_cats = top_cats[:1] if top_cats else []
            if not common_cats:
                continue

            common_cats_str = "、".join(common_cats[:2])
            cat_freq_dict = dict(category_counter.most_common(5))

            # TF-IDF plot关键词抽取（使用缓存）
            history_plot_keywords = Counter()
            for sid in set(history):
                kws = sid_to_plot_keywords.get(sid, [])
                history_plot_keywords.update(kws)
            rec_plot_keywords = sid_to_plot_keywords.get(rec_sid, [])
            common_plot_themes = [kw for kw in rec_plot_keywords if kw in history_plot_keywords]

            # 生成输入
            instruction = random.choice(input_templates)(history, rec_sid, rec_title)

            # 生成推荐理由（加入 plot 关键词、主题）
            output = generate_reasoning(
                history_categories, rec_category, rec_title, 
                common_cats_str, cat_freq_dict, 
                common_plot_themes, history_plot_keywords, rec_plot_keywords
            )

            samples.append({
                "task": "recommendation_reasoning",
                "instruction": instruction.strip(),
                "output": output
            })
    
    print(f"生成了 {len(samples):,} 个任务7样本")
    return samples


def task8_topk_candidate_recommendation(
    sequences_df: pl.DataFrame,
    items_df: pl.DataFrame,
    num_candidates: int = 5,
    max_positions: int = 3,
    max_samples: int = 200_000
) -> List[Dict]:
    """
    任务8：Top-K候选推荐（List-wise Thinking）
    
    输入：用户历史序列、商品数据
    输出：从候选列表中选择或排序的样本
    
    任务形式A：从候选列表中选择最合适的商品
    任务形式B：对候选列表按推荐优先级排序
    """
    import random

    print("\n生成任务8：Top-K候选推荐...")

    samples = []

    # 构建基础映射
    sid_to_item = {}
    all_sids = []
    for row in items_df.iter_rows(named=True):
        sid = row["semantic_id"]
        sid_to_item[sid] = {
            "title": row.get("title", ""),
            "category": row.get("category", ""),
        }
        all_sids.append(sid)
    all_sids_set = set(all_sids)

    # 文本快速映射：用于拼接
    sid_to_text = {
        sid: f"{item['title']}（{item['category']}）"
        for sid, item in sid_to_item.items()
    }

    # 按类别的 sid 池
    items_by_category = {}
    for sid, item in sid_to_item.items():
        category = item.get("category", "")
        if category:
            first_cat = category.split(",")[0].strip()
            if first_cat not in items_by_category:
                items_by_category[first_cat] = []
            items_by_category[first_cat].append(sid)

    # 选择任务模板
    selection_templates = [
        lambda history, candidates: f"用户历史序列：\n{', '.join(history)}\n\n候选商品：\n" + "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)]) + "\n\n请从候选列表中选择最合适的一个。",
        lambda history, candidates: f"根据用户历史（{', '.join(history)}），请从以下候选中选择最匹配的商品：\n" + "\n".join(candidates),
        lambda history, candidates: f"用户浏览记录：{', '.join(history)}\n候选商品列表：{', '.join(candidates)}\n请选择最符合用户偏好的商品。",
        lambda history, candidates: f"已知用户序列：{', '.join(history)}\n候选：{', '.join(candidates)}\n请输出最合适的推荐。"
    ]
    ranking_templates = [
        lambda history, candidates: f"用户历史序列：\n{', '.join(history)}\n\n候选商品：\n" + "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)]) + "\n\n请按推荐优先级排序。",
        lambda history, candidates: f"根据用户历史（{', '.join(history)}），请对以下候选商品按推荐优先级排序：\n" + "\n".join(candidates),
        lambda history, candidates: f"用户浏览记录：{', '.join(history)}\n候选：{', '.join(candidates)}\n请按从高到低的推荐顺序输出。"
    ]

    for row in tqdm(sequences_df.iter_rows(named=True), total=len(sequences_df)):
        seq = row["semantic_id_sequence"]
        if not seq or len(seq) < 2:
            continue

        positions = list(range(1, min(len(seq), 20)))
        if len(positions) > max_positions:
            positions = random.sample(positions, max_positions)

        for i in positions:
            if len(samples) >= max_samples:
                break

            # history (截断到8个)
            h_raw = seq[:i]
            if len(h_raw) > 8:
                h_raw = h_raw[-8:]
            history = [s for s in h_raw if s in sid_to_item]
            if not history or i >= len(seq):
                continue

            true_next = seq[i]
            if true_next not in sid_to_item:
                continue

            # 候选池构建
            candidates = [true_next]

            # 类别负例（不含自己）
            cat = sid_to_item[true_next].get("category", "")
            first_cat = cat.split(",")[0].strip() if cat else ""
            if first_cat:
                pool = [s for s in items_by_category.get(first_cat, []) if s != true_next]
                if pool:
                    cat_neg_num = min(2, len(pool))
                    candidates += random.sample(pool, cat_neg_num)

            # 随机负例池
            excluded = set(candidates)
            neg_pool = list(all_sids_set - excluded)
            need = num_candidates - len(candidates)
            if need > 0 and len(neg_pool) >= need:
                candidates += random.sample(neg_pool, need)

            candidates = candidates[:num_candidates]
            random.shuffle(candidates)  # 保证输出一定是混排

            # 文本映射
            history_text = [sid_to_text[s] for s in history]
            candidate_text = [sid_to_text[s] for s in candidates]

            # 50% 选择，50% 排序
            if random.random() < 0.5:
                # selection task
                instruction = random.choice(selection_templates)(history_text, candidate_text)
                output = true_next
                samples.append({
                    "task": "topk_candidate_selection",
                    "instruction": instruction.strip(),
                    "output": output,
                    "candidates": candidates
                })
            else:
                # ranking task：对候选列表排序
                sorted_candidates = [true_next] + [c for c in candidates if c != true_next]
                out_text = ", ".join(sorted_candidates)
                instruction = random.choice(ranking_templates)(history_text, candidate_text)
                samples.append({
                    "task": "topk_candidate_ranking",
                    "instruction": instruction.strip(),
                    "output": out_text,
                    "candidates": candidates
                })
        if len(samples) >= max_samples:
            break

    print(f"生成了 {len(samples):,} 个任务8样本")
    return samples


def task9_preference_qa(
    sequences_df: pl.DataFrame,
    items_df: pl.DataFrame
) -> List[Dict]:
    """
    任务9：用户偏好问答（Preference QA）
    
    输入：用户历史序列、商品数据
    输出：基于历史序列回答用户偏好相关问题的样本
    """
    import random
    from collections import Counter

    print("\n生成任务9：用户偏好问答...")
    samples = []

    # 创建语义ID到item信息的映射（保证有 categories 字段）
    sid_to_item = {}
    for row in items_df.iter_rows(named=True):
        sid = row["semantic_id"]
        category_str = row.get("category", "")
        categories = [c.strip() for c in category_str.split(",")] if category_str else []
        sid_to_item[sid] = {
            "title": row.get("title", ""),
            "category": category_str,
            "categories": categories,
            "plot": row.get("plot", "")
        }

    # 问题模板
    question_templates = [
        ("这个用户更偏好哪一类内容？", "preference_category"),
        ("用户主要对什么类型的内容感兴趣？", "preference_category"),
        ("用户兴趣是否多样？", "diversity"),
        ("用户最近偏好是否发生变化？", "preference_change"),
        ("用户是否更偏向剧情型内容？", "narrative_preference"),
        ("用户喜欢什么风格的内容？", "style_preference"),
        ("该用户的主要兴趣领域是什么？", "main_interest"),
        ("用户对哪些类别表现出明显偏好？", "preferred_categories")
    ]

    # 答案生成函数
    def generate_answer(question_type, categories, plots, titles, seq_len, cat_freq_dict=None):
        category_counter = Counter(categories)
        top_categories = [cat for cat, _ in category_counter.most_common(3) if cat]
        unique_cats = len(set([c for c in categories if c]))
        has_plots = bool(plots)

        # 使用类别频率信息（如果提供）
        if cat_freq_dict is None:
            cat_freq_dict = dict(category_counter.most_common(3))

        if question_type == "preference_category":
            if top_categories:
                top_cat = top_categories[0]
                freq = cat_freq_dict.get(top_cat, 0)
                freq_info = f"（出现{freq}次）" if freq > 1 else ""
                answers = [
                    f"用户主要偏好 {', '.join(top_categories)} 类内容{freq_info}，浏览频率较高。",
                    f"该用户更偏好 {', '.join(top_categories)} 类别{freq_info}，在历史记录中多次出现。",
                    f"用户对 {', '.join(top_categories)} 类内容表现出明显兴趣{freq_info}，是用户最常浏览的类型。",
                    f"根据历史记录分析，用户主要关注 {', '.join(top_categories)} 相关内容{freq_info}，兴趣集中在这几个类别。"
                ]
                return random.choice(answers)
            else:
                return "无法判断用户偏好的内容类别。"

        elif question_type == "diversity":
            if unique_cats >= 5:
                return "用户兴趣非常多样，涵盖了多个不同类别的内容。"
            elif unique_cats >= 3:
                return "用户兴趣较为多样，同时对多个类别都有一定关注。"
            elif unique_cats >= 1:
                return "用户兴趣相对集中，主要关注少数几个类别。"
            else:
                return "暂无足够信息判断用户兴趣多样性。"

        elif question_type == "preference_change":
            if seq_len >= 5 and unique_cats > 0:
                mid = len(categories) // 2
                first_half = set([c for c in categories[:mid] if c])
                second_half = set([c for c in categories[mid:] if c])
                union_len = len(first_half | second_half)
                if union_len == 0:
                    return "历史记录不足，无法判断用户偏好变化。"
                overlap = len(first_half & second_half) / union_len
                if overlap < 0.5:
                    return "用户最近偏好发生了明显变化，开始关注新的内容类型。"
                else:
                    return "用户偏好保持相对稳定，持续关注相似类型的内容。"
            else:
                return "由于历史记录较短，难以判断偏好变化。"

        elif question_type == "narrative_preference":
            if has_plots:
                return "用户更偏向剧情型内容，偏好有完整故事线和情节发展的作品。"
            else:
                return "用户对剧情型内容的偏好不明显。"

        elif question_type == "style_preference":
            if has_plots:
                return "用户偏好叙事性和情节驱动的作品风格。"
            else:
                return "用户对内容风格没有明显偏好。"

        elif question_type == "main_interest":
            if top_categories:
                return f"用户的主要兴趣领域是 {top_categories[0]} 类内容。"
            else:
                return "无法从历史记录中明确判断用户的主要兴趣。"

        elif question_type == "preferred_categories":
            if top_categories:
                return f"用户对 {', '.join(top_categories)} 表现出明显偏好。"
            else:
                return "用户对各类内容的偏好较为均衡。"

        return ""

    for row in tqdm(sequences_df.iter_rows(named=True), total=len(sequences_df)):
        semantic_id_sequence = row["semantic_id_sequence"]

        if len(semantic_id_sequence) < 2:
            continue

        categories = []
        plots = []
        titles = []

        for sid in semantic_id_sequence:
            if sid in sid_to_item:
                item = sid_to_item[sid]
                # 获取categories字段（是list）
                cats = item.get("categories", [])
                if cats:
                    categories.extend([c for c in cats if c])
                if item.get("plot", ""):
                    plots.append(item["plot"])
                if item.get("title", ""):
                    titles.append(item["title"])

        if not categories:
            continue

        question, q_type = random.choice(question_templates)

        sequence_str = ", ".join(semantic_id_sequence[:5])
        if len(semantic_id_sequence) > 5:
            sequence_str += f", ... (共{len(semantic_id_sequence)}个)"

        cat_counter = Counter(categories)
        cat_freq_dict = dict(cat_counter.most_common(5))

        instruction = f"用户历史序列：{sequence_str}\n\n问题：{question}"
        output = generate_answer(q_type, categories, plots, titles, len(semantic_id_sequence), cat_freq_dict)

        if output:
            samples.append({
                "task": "preference_qa",
                "instruction": instruction.strip(),
                "output": output,
            })

    print(f"生成了 {len(samples):,} 个任务9样本")
    if len(samples) == 0:
        print("⚠️ 可能原因：所有用户历史里集成的类别(Category)字段为空，或者 items_df 没有正确的 category 字段。")
    return samples


def task10_constraint_aware_recommendation(
    sequences_df: pl.DataFrame,
    items_df: pl.DataFrame
) -> List[Dict]:
    """
    任务10：约束感知推荐（Constraint-aware）
    
    输入：用户历史序列、商品数据
    输出：在约束条件下推荐商品的样本
    
    基于类别提出显式约束条件，推荐符合约束的商品
    """
    import random
    from collections import Counter

    print("\n生成任务10：约束感知推荐...")

    samples = []

    # 1. 准备基本数据结构
    sid_to_item = {}
    items_by_category = {}
    for row in items_df.iter_rows(named=True):
        sid = row["semantic_id"]
        category = row.get("category", "")
        categories_list = [c.strip() for c in category.split(",")] if category else []
        sid_to_item[sid] = {
            "title": row.get("title", ""),
            "category": category,
            "categories": categories_list,
            "plot": row.get("plot", "")
        }
        for cat in categories_list:
            if cat not in items_by_category:
                items_by_category[cat] = []
            items_by_category[cat].append(sid)

    # category约束模板，带条件的明确限制
    constraint_templates = [
        ("请推荐一部属于{category}类型的最新商品。", "category_latest"),
        ("希望推荐一部{category}类、但风格更现代的。", "category_modern"),
        ("想要{category}类、剧情更紧凑的商品。", "category_compact_plot"),
        ("偏好{category}类，但需要带有动作元素。", "category_action"),
        ("只想看{category}类，不希望有太多情感戏。", "category_no_romance"),
        ("{category}类别，风格欢快为佳。", "category_light"),
        ("希望剧情更深刻的{category}商品。", "category_deep"),
        ("要选{category}类型，还有悬疑成分。", "category_with_mystery"),
        ("求推荐{category}，但题材要新颖。", "category_novel"),
        ("推荐一部{category}且评分较高的作品。", "category_high_rating"),
    ]

    # 指令模板：强调有约束条件
    input_templates = [
        lambda history, cons: f"用户历史序列：{', '.join(history)}\n\n用户需求：{cons}\n请推荐一个符合要求的商品（必须严格满足需求中的约束条件）。",
        lambda history, cons: f"根据用户最近浏览（{', '.join(history)}），TA有如下要求：{cons}\n请挑选一个最合适的。",
        lambda history, cons: f"给定用户序列：{', '.join(history)}\n约束条件：{cons}\n请输出推荐结果，并确保满足全部约束。",
        lambda history, cons: f"TA常看：{', '.join(history)}\n这次需求是：{cons}\n请依据这个限制选出推荐。",
        lambda history, cons: f"用户以往浏览记录包括：{', '.join(history)}。\n但这次用户想要：“{cons}”，请推荐一个商品。"
    ]

    for row in tqdm(sequences_df.iter_rows(named=True), total=len(sequences_df)):
        semantic_id_sequence = row["semantic_id_sequence"]
        if len(semantic_id_sequence) < 2:
            continue

        history_categories = []
        for sid in semantic_id_sequence[:5]:
            if sid in sid_to_item:
                history_categories.extend(sid_to_item[sid].get("categories", []))
        if not history_categories:
            continue

        category_counter = Counter(history_categories)
        preferred_category = category_counter.most_common(1)[0][0]
        # 构造约束
        constraint_text, constraint_type = random.choice(constraint_templates)
        constraint = constraint_text.format(category=preferred_category)

        # 仅推荐来自对应category的商品，且尽量不在用户历史里出现
        candidates = [
            sid for sid in items_by_category.get(preferred_category, [])
            if sid not in semantic_id_sequence
        ]
        if not candidates:
            continue

        # 推荐商品
        recommended_sid = random.choice(candidates)
        recommended_title = sid_to_item[recommended_sid]["title"]

        # 取history
        history = [
            sid_to_item[sid]["title"] if (sid in sid_to_item and sid_to_item[sid]["title"]) else sid
            for sid in semantic_id_sequence[:5]
        ]
        if len(history) > 8:
            history = history[-8:]

        instruction = random.choice(input_templates)(history, constraint)

        samples.append({
            "task": "constraint_aware_recommendation",
            "instruction": instruction.strip(),
            "output": recommended_sid,
            "output_title": recommended_title,
            "constraint": constraint,
            "constraint_type": constraint_type,
            "category": preferred_category
        })

    print(f"生成了 {len(samples):,} 个任务10样本")
    if len(samples) == 0:
        print("⚠️ 可能没有合适的category类别及其商品供约束推荐。")
    return samples


def task11_counterfactual_explanation(
    sequences_df: pl.DataFrame,
    items_df: pl.DataFrame
) -> List[Dict]:
    """
    任务11：反事实/对比推荐（Why A not B）
    
    输入：用户历史序列、商品数据
    输出：解释为什么推荐A而不是B的样本
    """
    import random

    print("\n生成任务11：反事实/对比推荐...")

    MAX_SAMPLES = 150_000
    samples = []

    # 创建语义ID到item信息和类别集合的映射
    sid_to_item = {}
    sid_to_cats = {}
    for row in items_df.iter_rows(named=True):
        sid = row["semantic_id"]
        category_str = row.get("category", "")
        category_set = set(c.strip() for c in category_str.split(",")) if category_str else set()
        sid_to_item[sid] = {
            "title": row.get("title", ""),
            "category": category_str,
            "plot": row.get("plot", "")
        }
        sid_to_cats[sid] = category_set

    # 所有商品sid缓存
    all_sids = list(sid_to_item.keys())
    all_sids_set = set(all_sids)

    # 输入模板
    input_templates = [
        lambda history, rec_a, rec_b: f"用户历史序列：{', '.join(history[:5])}{'...' if len(history) > 5 else ''}\n\n问题：为什么推荐 {rec_a} 而不是 {rec_b}？",
        lambda history, rec_a, rec_b: f"根据用户历史浏览记录（{', '.join(history[:5])}{'...' if len(history) > 5 else ''}），请解释为什么推荐 {rec_a} 而不是 {rec_b}。",
        lambda history, rec_a, rec_b: f"用户浏览记录：{', '.join(history[:5])}{'...' if len(history) > 5 else ''}\n推荐：{rec_a}\n未推荐：{rec_b}\n请说明推荐理由。",
        lambda history, rec_a, rec_b: f"已知用户序列：{', '.join(history[:5])}{'...' if len(history) > 5 else ''}\n为什么选择 {rec_a} 而非 {rec_b}？请给出解释。",
        lambda history, rec_a, rec_b: f"基于用户历史（{', '.join(history[:5])}{'...' if len(history) > 5 else ''}），请对比说明为什么推荐 {rec_a} 而不是 {rec_b}。"
    ]

    # 解释生成函数
    def generate_explanation(history_cats, rec_a_cats, rec_a_title, rec_b_cats, rec_b_title):
        a_overlap = len(rec_a_cats & history_cats)
        b_overlap = len(rec_b_cats & history_cats)
        common_cats_a = rec_a_cats & history_cats
        common_cats_b = rec_b_cats & history_cats

        templates = []
        if a_overlap > b_overlap:
            if common_cats_a:
                cats_str = "、".join(list(common_cats_a)[:2])
                templates.append(
                    f"{rec_a_title} 与用户此前关注的 {cats_str} 类内容更接近，在类别和风格上高度匹配；而 {rec_b_title} 在风格上差异较大，与用户历史偏好的相关性较低。"
                )
                templates.append(
                    f"推荐 {rec_a_title} 是因为它在 {cats_str} 类别上与用户历史偏好高度匹配，符合用户的兴趣模式；而 {rec_b_title} 的相关性较低，不太符合用户的浏览习惯。"
                )
                templates.append(
                    f"从类别匹配度来看，{rec_a_title} 属于用户常浏览的 {cats_str} 类别，与历史兴趣一致；相比之下，{rec_b_title} 与用户偏好的关联性较弱。"
                )
        elif a_overlap == b_overlap and a_overlap > 0:
            if common_cats_a:
                cats_str = "、".join(list(common_cats_a)[:2])
                templates.append(
                    f"虽然两者都属于 {cats_str} 类别，但 {rec_a_title} 在内容特点、主题风格上更符合用户的历史浏览模式，与用户兴趣的契合度更高。"
                )
                templates.append(
                    f"虽然 {rec_a_title} 和 {rec_b_title} 都属于 {cats_str} 类别，但 {rec_a_title} 在内容特征上更贴近用户的历史偏好，因此更适合推荐。"
                )
        else:
            templates.append(
                f"{rec_a_title} 在主题和风格上与用户历史序列的关联性更强，与用户偏好的一致性更高；而 {rec_b_title} 与用户兴趣的匹配度较低，不太符合用户的浏览模式。"
            )
            templates.append(
                f"基于用户历史分析，{rec_a_title} 在内容类型和风格特点上与用户偏好更匹配；相比之下，{rec_b_title} 与用户历史兴趣的关联性较弱。"
            )

        if not templates:
            templates.append(
                f"基于用户历史偏好分析，{rec_a_title} 比 {rec_b_title} 更符合用户的兴趣方向，在内容特点和风格上更匹配。"
            )
        return random.choice(templates)

    # 主采样循环
    for row in tqdm(sequences_df.iter_rows(named=True), total=len(sequences_df)):
        seq = row["semantic_id_sequence"]
        if len(seq) < 2:
            continue

        # 构建history_cat_sets: 每一位置历史兴趣的set副本
        history_cat_sets = []
        cur = set()
        for sid in seq:
            cur |= sid_to_cats.get(sid, set())
            history_cat_sets.append(cur.copy())

        # 位置采样：每条序列最多取2个位置
        positions = list(range(1, min(len(seq), 15)))
        max_pos = 2
        if len(positions) > max_pos:
            positions = random.sample(positions, max_pos)

        for i in positions:
            history = seq[max(0, i-8):i]
            rec_a = seq[i]

            # rec_a必须在item库
            if rec_a not in sid_to_item:
                continue

            history_cats = history_cat_sets[i-1] if i-1 >= 0 else set()
            if not history_cats:
                continue

            rec_a_cats = sid_to_cats[rec_a]

            # 随机选取未在history及不是rec_a的商品
            excluded = set(history)
            excluded.add(rec_a)
            # 万一所有商品都在排除集，break
            if len(all_sids_set - excluded) == 0:
                continue

            try_count = 0
            while True:
                rec_b = random.choice(all_sids)
                if rec_b not in excluded:
                    break
                try_count += 1
                if try_count > 10:  # unlikely, but safe
                    rec_b = None
                    break
            if not rec_b or rec_b not in sid_to_item:
                continue

            rec_a_title = sid_to_item[rec_a].get("title", "该商品")
            rec_b_title = sid_to_item[rec_b].get("title", "另一个商品")
            rec_b_cats = sid_to_cats[rec_b]

            instruction = random.choice(input_templates)(history, rec_a, rec_b)
            output = generate_explanation(
                history_cats,
                rec_a_cats,
                rec_a_title,
                rec_b_cats,
                rec_b_title
            )

            samples.append({
                "task": "counterfactual_explanation",
                "instruction": instruction.strip(),
                "output": output,
                "recommended": rec_a,
                "not_recommended": rec_b
            })

            # 限制总样本数，提前退出
            if len(samples) >= MAX_SAMPLES:
                print(f"任务11已采满最大上限 {MAX_SAMPLES:,} 样本，提前终止。")
                print(f"生成了 {len(samples):,} 个任务11样本")
                return samples

    print(f"生成了 {len(samples):,} 个任务11样本")
    return samples


def task12_composite_recommend_and_reason(
    sequences_df: pl.DataFrame,
    items_df: pl.DataFrame
) -> List[Dict]:
    """
    任务12：复合任务（推荐+理由生成）
    
    输入：用户历史序列、商品数据
    输出：同时包含推荐和理由的样本
    
    让模型一次性完成推荐和解释
    """
    import random
    from collections import Counter
    
    print("\n生成任务12：复合任务（推荐+理由生成）...")
    samples = []
    
    # 创建语义ID到item信息的映射
    sid_to_item = {}
    for row in items_df.iter_rows(named=True):
        sid = row["semantic_id"]
        category = row.get("category", "")
        sid_to_item[sid] = {
            "title": row.get("title", ""),
            "category": category,
            "categories": [c.strip() for c in category.split(",")] if category else [],  # 预处理
            "plot": row.get("plot", "")
        }
    
    # 输入模板
    input_templates = [
        lambda history: f"用户最近观看了以下内容：{', '.join(history[:5])}{'...' if len(history) > 5 else ''}，请推荐一个可能感兴趣的商品，并说明推荐理由。",
        lambda history: f"根据用户历史序列（{', '.join(history[:5])}{'...' if len(history) > 5 else ''}），请推荐一个商品并解释原因。",
        lambda history: f"用户浏览记录：{', '.join(history[:5])}{'...' if len(history) > 5 else ''}\n请推荐一个合适的商品，并说明为什么推荐它。",
        lambda history: f"已知用户序列：{', '.join(history[:5])}{'...' if len(history) > 5 else ''}\n请给出推荐并解释推荐理由。",
        lambda history: f"基于用户的历史浏览行为（{', '.join(history[:5])}{'...' if len(history) > 5 else ''}），请推荐一个符合用户兴趣的商品，并说明推荐依据。"
    ]
    
    # 理由生成函数：生成包含推荐和理由的输出
    def generate_reasoning(history_cats, rec_cat, rec_title, rec_sid, common_cats):
        # 格式1：明确分离格式（推荐和理由分开展示）
        format1_templates = [
            f"推荐：{rec_title} ({rec_sid})\n\n推荐理由：用户此前多次关注 {common_cats} 类内容，而 {rec_title} 在题材和风格上高度相似。",
            f"推荐：{rec_title} ({rec_sid})\n\n理由：用户历史中 {common_cats} 类别出现频率较高，该商品属于同类内容，符合用户偏好。",
            f"推荐商品：{rec_title}\n商品ID：{rec_sid}\n\n推荐原因：基于用户对 {common_cats} 类内容的持续兴趣，推荐了同类型的商品，在主题上高度契合。",
        ]
        
        # 格式2：自然对话式格式（模拟真实对话场景）
        format2_templates = [
            f"我推荐{rec_title}（{rec_sid}）。\n\n推荐理由：用户此前主要关注 {common_cats} 相关内容，该商品在分类和风格上与用户历史偏好一致。",
            f"根据您的浏览历史，我推荐{rec_title}（{rec_sid}）。\n\n这是因为：用户对 {common_cats} 类内容表现出明显兴趣，而 {rec_title} 属于同类内容，在主题和风格上高度匹配。",
            f"为您推荐：{rec_title}（{rec_sid}）\n\n推荐依据：考虑到用户多次浏览 {common_cats} 类型的内容，推荐该商品是因为它在题材和风格上与用户兴趣高度相似。",
        ]
        
        # 格式3：一体式自然语言（推荐和理由融为一体）
        format3_templates = [
            f"推荐 {rec_title}。因为用户此前多次关注 {common_cats} 类内容，而 {rec_title} 在题材和风格上高度相似。",
            f"推荐 {rec_title}。用户历史中 {common_cats} 类别出现频率较高，该商品属于同类内容，符合用户偏好。",
            f"基于用户对 {common_cats} 类内容的持续兴趣，我推荐 {rec_title}。该商品在主题上高度契合用户的浏览模式。",
        ]
        
        # 随机选择格式类型（40%分离格式，40%对话格式，20%一体式）
        rand = random.random()
        if rand < 0.4:
            return random.choice(format1_templates)
        elif rand < 0.8:
            return random.choice(format2_templates)
        else:
            return random.choice(format3_templates)
    
    # 只生成少量样本（1-3%）
    sample_rate = 0.02  # 2%的序列
    
    for row in tqdm(sequences_df.iter_rows(named=True), total=len(sequences_df)):
        semantic_id_sequence = row["semantic_id_sequence"]
        
        if len(semantic_id_sequence) < 2:
            continue
        
        # 采样：只处理2%的序列
        if random.random() > sample_rate:
            continue
        
        # 为每个位置生成样本（但只采样部分位置）
        for i in range(1, len(semantic_id_sequence)):
            if random.random() > 0.3:  # 每个序列只生成30%的位置
                continue
                
            history = semantic_id_sequence[:i]
            rec_sid = semantic_id_sequence[i]
            
            if len(history) > 10:
                history = history[-10:]
            
            if rec_sid not in sid_to_item:
                continue
            
            # 收集历史序列的类别信息
            history_categories = []
            for sid in history:
                if sid in sid_to_item:
                    history_categories.extend(sid_to_item[sid].get("categories", []))
            
            rec_item = sid_to_item[rec_sid]
            rec_category = rec_item.get("category", "")
            rec_title = rec_item.get("title", "")
            
            if not rec_category or not history_categories:
                continue
            
            # 找出共同类别
            rec_cats = sid_to_item[rec_sid].get("categories", [])
            category_counter = Counter(history_categories)
            common_cats = [cat for cat in rec_cats if cat in category_counter]
            
            if not common_cats:
                top_cats = [cat for cat, _ in category_counter.most_common(2)]
                common_cats = top_cats[:1] if top_cats else []
            
            if common_cats:
                common_cats_str = "、".join(common_cats[:2])
                
                # 生成输入
                instruction = random.choice(input_templates)(history)
                
                # 生成输出（推荐+理由）
                output = generate_reasoning(history_categories, rec_category, rec_title, rec_sid, common_cats_str)
                
                samples.append({
                    "task": "recommend_and_reason",
                    "instruction": instruction.strip(),
                    "output": output,
                    "recommended": rec_sid
                })
    
    print(f"生成了 {len(samples):,} 个任务12样本")
    return samples


def task13_feedback_aware_recommendation(
    sequences_df: pl.DataFrame,
    items_df: pl.DataFrame
) -> List[Dict]:
    """
    任务13：反馈感知推荐（Feedback-aware Recommendation）
    
    输入：用户历史序列、商品数据
    输出：基于用户负面反馈推荐新商品的样本
    
    用户对某个商品表示不喜欢，需要推荐新的商品并说明理由
    """
    import random
    from collections import Counter
    
    print("\n生成任务13：反馈感知推荐...")
    samples = []
    
    # 创建语义ID到item信息的映射
    sid_to_item = {}
    for row in items_df.iter_rows(named=True):
        sid = row["semantic_id"]
        category = row.get("category", "")
        sid_to_item[sid] = {
            "title": row.get("title", ""),
            "category": category,
            "categories": [c.strip() for c in category.split(",")] if category else [],  # 预处理
            "plot": row.get("plot", "")
        }
    
    # 输入模板
    input_templates = [
        lambda history, disliked: f"用户历史：{', '.join(history[:5])}{'...' if len(history) > 5 else ''}。用户对 {disliked} 表示不喜欢。请推荐一个新的商品，并说明理由。",
        lambda history, disliked: f"根据用户历史序列（{', '.join(history[:5])}{'...' if len(history) > 5 else ''}），用户明确拒绝 {disliked}。请推荐替代商品并解释原因。",
        lambda history, disliked: f"用户浏览记录：{', '.join(history[:5])}{'...' if len(history) > 5 else ''}\n用户反馈：不喜欢 {disliked}\n请推荐一个新商品并说明理由。",
        lambda history, disliked: f"已知用户序列：{', '.join(history[:5])}{'...' if len(history) > 5 else ''}\n用户对 {disliked} 表示不感兴趣。请给出新的推荐并解释。",
        lambda history, disliked: f"基于用户历史（{', '.join(history[:5])}{'...' if len(history) > 5 else ''}），用户对 {disliked} 给出了负面反馈。请推荐一个更符合用户偏好的商品，并说明原因。"
    ]
    
    # 理由生成函数：生成考虑用户反馈的推荐理由
    def generate_feedback_reasoning(history_cats, disliked_cat, rec_cat, rec_title, rec_sid, common_cats):
        # 格式1：明确分离格式（推荐和理由分开展示）
        format1_templates = [
            f"推荐：{rec_title} ({rec_sid})\n\n推荐理由：用户此前偏好 {common_cats} 类内容，而 {disliked_cat} 属于不同类别，已被用户明确拒绝，因此选择 {rec_title}。",
            f"推荐商品：{rec_title}\n商品ID：{rec_sid}\n\n推荐原因：用户历史显示偏好 {common_cats}，虽然 {disliked_cat} 不符合用户期望，但 {rec_title} 在 {common_cats} 类别上与用户兴趣一致。",
        ]
        
        # 格式2：自然对话式格式
        format2_templates = [
            f"考虑到您对 {disliked_cat} 的负面反馈，我推荐{rec_title}（{rec_sid}）。\n\n理由：用户此前偏好 {common_cats} 类内容，而该商品在 {common_cats} 类别上与您的历史兴趣匹配。",
            f"根据您的反馈，我为您推荐{rec_title}（{rec_sid}）。\n\n这是因为：用户明确表示不喜欢 {disliked_cat}，因此推荐了 {common_cats} 类别的 {rec_title}，更符合您的历史偏好。",
            f"我推荐{rec_title}（{rec_sid}）。\n\n推荐依据：考虑到您对 {disliked_cat} 的不喜欢，我们选择了 {common_cats} 类别的商品，与您的历史浏览偏好一致。",
        ]
        
        # 格式3：一体式自然语言（保留部分）
        format3_templates = [
            f"推荐 {rec_title}。因为用户此前偏好 {common_cats} 类内容，而 {disliked_cat} 属于不同类别，已被用户明确拒绝，因此选择 {rec_title}。",
            f"推荐 {rec_title}。用户历史显示偏好 {common_cats}，虽然 {disliked_cat} 不符合用户期望，但 {rec_title} 在 {common_cats} 类别上与用户兴趣一致。",
        ]
        
        # 随机选择格式类型（40%分离格式，40%对话格式，20%一体式）
        rand = random.random()
        if rand < 0.4:
            return random.choice(format1_templates)
        elif rand < 0.8:
            return random.choice(format2_templates)
        else:
            return random.choice(format3_templates)
    
    # 只生成少量样本（1-3%）
    sample_rate = 0.02
    
    for row in tqdm(sequences_df.iter_rows(named=True), total=len(sequences_df)):
        semantic_id_sequence = row["semantic_id_sequence"]
        
        if len(semantic_id_sequence) < 3:  # 至少需要3个item：历史+不喜欢+推荐
            continue
        
        # 采样：只处理2%的序列
        if random.random() > sample_rate:
            continue
        
        # 随机选择一个位置作为"不喜欢"的商品
        disliked_idx = random.randint(0, len(semantic_id_sequence) - 2)
        disliked_sid = semantic_id_sequence[disliked_idx]
        
        # 历史序列（排除不喜欢的商品）
        history = [sid for sid in semantic_id_sequence[:disliked_idx] if sid != disliked_sid]
        
        # 推荐的商品（在不喜欢之后）
        if disliked_idx + 1 < len(semantic_id_sequence):
            rec_sid = semantic_id_sequence[disliked_idx + 1]
        else:
            continue
        
        if len(history) > 8:
            history = history[-8:]
        
        if disliked_sid not in sid_to_item or rec_sid not in sid_to_item:
            continue
        
        # 收集历史序列的类别信息
        history_categories = []
        for sid in history:
            if sid in sid_to_item:
                history_categories.extend(sid_to_item[sid].get("categories", []))
        
        disliked_item = sid_to_item[disliked_sid]
        disliked_categories = disliked_item.get("categories", [])
        disliked_cat = disliked_categories[0] if disliked_categories else ""
        
        rec_item = sid_to_item[rec_sid]
        rec_category = rec_item.get("category", "")
        rec_title = rec_item.get("title", "")
        
        if not rec_category or not history_categories or not disliked_cat:
            continue
        
        # 找出推荐商品的共同类别（排除不喜欢的类别）
        rec_cats = sid_to_item[rec_sid].get("categories", [])
        category_counter = Counter(history_categories)
        common_cats = [cat for cat in rec_cats if cat in category_counter and cat != disliked_cat]
        
        if not common_cats:
            # 如果没有共同类别，使用历史中最常见的类别
            top_cats = [cat for cat, _ in category_counter.most_common(2) if cat != disliked_cat]
            common_cats = top_cats[:1] if top_cats else []
        
        if common_cats:
            common_cats_str = "、".join(common_cats[:2])
            
            # 生成输入
            instruction = random.choice(input_templates)(history, disliked_sid)
            
            # 生成输出
            output = generate_feedback_reasoning(
                history_categories, 
                disliked_cat, 
                rec_category, 
                rec_title,
                rec_sid,
                common_cats_str
            )
            
            samples.append({
                "task": "feedback_aware_recommendation",
                "instruction": instruction.strip(),
                "output": output,
                "disliked": disliked_sid,
                "recommended": rec_sid
            })
    
    print(f"生成了 {len(samples):,} 个任务13样本")
    return samples


def task14_multi_turn_dialogue_recommendation(
    sequences_df: pl.DataFrame,
    items_df: pl.DataFrame
) -> List[Dict]:
    """
    任务14：多轮对话推荐（Multi-turn dialogue recommendation）
    
    输入：用户历史序列、商品数据
    输出：多轮对话格式的推荐样本
    
    模拟多轮对话场景，用户反馈后系统调整推荐
    """
    import random
    from collections import Counter
    
    print("\n生成任务14：多轮对话推荐...")
    samples = []
    
    # 创建语义ID到item信息的映射
    sid_to_item = {}
    for row in items_df.iter_rows(named=True):
        sid = row["semantic_id"]
        category = row.get("category", "")
        sid_to_item[sid] = {
            "title": row.get("title", ""),
            "category": category,
            "categories": [c.strip() for c in category.split(",")] if category else [],  # 预处理
            "plot": row.get("plot", "")
        }
    
    # 用户消息模板：使用商品名称而非ID
    def create_user_message(history_sids, sid_to_item):
        """
        创建自然的用户消息，使用商品名称而非ID
        
        输入：用户历史序列中的语义ID列表、语义ID到商品信息的映射
        输出：包含商品名称的用户消息字符串
        """
        titles = []
        categories = []
        for sid in history_sids[:3]:
            if sid in sid_to_item:
                item = sid_to_item[sid]
                if item.get("title"):
                    titles.append(item["title"])
                if item.get("category"):
                    cat = item["category"].split(",")[0].strip()
                    if cat and cat not in categories:
                        categories.append(cat)
        
        templates = []
        
        # 类型1：使用商品名称的模板
        if titles:
            title_str = "、".join([f"《{t}》" for t in titles[:2]])
            if len(titles) > 2:
                title_str += f"等{len(titles)}部作品"
            
            templates.extend([
                lambda: f"我看过{title_str}。",
                lambda: f"我最近看了{title_str}。",
                lambda: f"我最近浏览了{title_str}。",
                lambda: f"我之前看过{title_str}。",
            ])
            
            # 结合类别信息，使表达更丰富自然
            if categories:
                cat_str = "、".join(categories[:2])
                templates.extend([
                    lambda: f"我看过几部{cat_str}类型的电影，比如{title_str}。",
                    lambda: f"我对{cat_str}很感兴趣，之前看过{title_str}。",
                    lambda: f"我喜欢{cat_str}类型的作品，看过{title_str}。",
                ])
        
        # 类型2：混合使用（部分用名称，部分用ID）
        if history_sids:
            mixed_str = "、".join(history_sids[:3])
            templates.extend([
                lambda: f"我看过 {mixed_str}。",
                lambda: f"我最近浏览了 {mixed_str}。",
            ])
        
        return random.choice(templates)() if templates else f"我看过 {', '.join(history_sids[:3])}。"
    
    user_message_templates = [
        lambda sids, items: create_user_message(sids, items)
    ]
    
    # 助手推荐消息模板
    assistant_recommend_templates = [
        lambda rec, reason: f"我推荐 {rec}，因为{reason}。",
        lambda rec, reason: f"根据您的浏览历史，我推荐 {rec}。{reason}。",
        lambda rec, reason: f"我建议您看看 {rec}，{reason}。",
        lambda rec, reason: f"为您推荐 {rec}，{reason}。",
        lambda rec, reason: f"基于您的兴趣，我推荐 {rec}，{reason}。"
    ]
    
    # 用户反馈模板：使用商品名称而非ID
    def create_user_feedback(disliked_sid, sid_to_item):
        """
        创建自然的用户反馈，使用商品名称而非ID
        
        输入：用户不喜欢的商品语义ID、语义ID到商品信息的映射
        输出：使用商品名称的用户反馈字符串
        """
        if disliked_sid in sid_to_item:
            disliked_title = sid_to_item[disliked_sid].get("title", "")
            if disliked_title:
                templates = [
                    f"我不喜欢《{disliked_title}》",
                    f"《{disliked_title}》不太符合我的兴趣",
                    f"我对《{disliked_title}》不感兴趣",
                    f"《{disliked_title}》不是我想要的",
                    f"这个《{disliked_title}》不太适合我",
                    f"《{disliked_title}》不太合我的口味",
                ]
                return random.choice(templates)
        
        # 回退方案：如果找不到商品名称，使用ID
        return f"我不喜欢 {disliked_sid}"
    
    user_feedback_templates = [
        lambda sid, items: create_user_feedback(sid, items)
    ]
    
    # 助手调整推荐模板
    assistant_adjust_templates = [
        lambda rec, reason: f"那我推荐 {rec}，因为{reason}。",
        lambda rec, reason: f"了解，那我建议 {rec}，{reason}。",
        lambda rec, reason: f"好的，我为您推荐 {rec}，{reason}。",
        lambda rec, reason: f"明白了，试试 {rec} 吧，{reason}。",
        lambda rec, reason: f"没问题，我重新为您推荐 {rec}，{reason}。"
    ]
    
    # 理由生成函数
    def generate_reasoning(history_cats, rec_cat, rec_title, common_cats, cat_freq=None):
        if common_cats:
            freq_info = f"（您经常浏览此类内容）" if cat_freq and cat_freq.get(common_cats, 0) > 1 else ""
            return f"该商品属于 {common_cats} 类别{freq_info}，与您之前关注的内容类型一致，风格匹配"
        return f"该商品在主题和风格上与您的历史偏好匹配，符合您的兴趣方向"
    
    def generate_adjust_reasoning(history_cats, disliked_cat, rec_cat, rec_title, common_cats, cat_freq=None):
        if common_cats:
            freq_info = f"（您经常浏览此类内容）" if cat_freq and cat_freq.get(common_cats, 0) > 1 else ""
            return f"该商品属于 {common_cats} 类别{freq_info}，与您之前喜欢的内容类型一致，同时避免了 {disliked_cat} 类别的元素，更符合您的偏好"
        return f"该商品更符合您的历史偏好，避免了您不感兴趣的 {disliked_cat} 类型，在内容特点上与您的兴趣匹配"
    
    # 只生成少量样本（1-3%）
    sample_rate = 0.02
    
    for row in tqdm(sequences_df.iter_rows(named=True), total=len(sequences_df)):
        semantic_id_sequence = row["semantic_id_sequence"]
        
        if len(semantic_id_sequence) < 4:  # 至少需要：历史+推荐1+反馈+推荐2
            continue
        
        # 采样：只处理2%的序列
        if random.random() > sample_rate:
            continue
        
        # 构建对话轮次
        # 第一轮：用户提供历史
        history_start = random.randint(0, max(0, len(semantic_id_sequence) - 3))
        history = semantic_id_sequence[history_start:history_start + 3]
        if len(history) < 2:
            continue
        
        # 第一轮推荐
        rec1_idx = history_start + len(history)
        if rec1_idx >= len(semantic_id_sequence):
            continue
        rec1_sid = semantic_id_sequence[rec1_idx]
        
        # 用户反馈（不喜欢第一个推荐）
        # 第二轮推荐
        rec2_idx = rec1_idx + 1
        if rec2_idx >= len(semantic_id_sequence):
            continue
        rec2_sid = semantic_id_sequence[rec2_idx]
        
        if rec1_sid not in sid_to_item or rec2_sid not in sid_to_item:
            continue
        
        # 收集历史类别
        history_categories = []
        for sid in history:
            if sid in sid_to_item:
                history_categories.extend(sid_to_item[sid].get("categories", []))
        
        if not history_categories:
            continue
        
        rec1_item = sid_to_item[rec1_sid]
        rec1_category = rec1_item.get("category", "")
        rec1_title = rec1_item.get("title", "")
        
        rec2_item = sid_to_item[rec2_sid]
        rec2_category = rec2_item.get("category", "")
        rec2_title = rec2_item.get("title", "")
        
        rec1_cats = sid_to_item[rec1_sid].get("categories", [])
        rec1_cat = rec1_cats[0] if rec1_cats else ""
        rec2_cats = sid_to_item[rec2_sid].get("categories", [])
        rec2_cat = rec2_cats[0] if rec2_cats else ""
        
        # 提取类别频率信息
        category_counter = Counter(history_categories)
        cat_freq_dict = dict(category_counter.most_common(5))
        
        # 生成第一轮推荐的理由
        common_cats1 = [cat for cat in rec1_cats if cat in category_counter]
        if not common_cats1:
            top_cats = [cat for cat, _ in category_counter.most_common(1)]
            common_cats1 = top_cats
        common_cats1_str = "、".join(common_cats1[:1]) if common_cats1 else ""
        
        # 生成第二轮推荐的理由（考虑反馈）
        common_cats2 = [cat for cat in rec2_cats if cat in category_counter and cat != rec1_cat]
        if not common_cats2:
            top_cats = [cat for cat, _ in category_counter.most_common(2) if cat != rec1_cat]
            common_cats2 = top_cats[:1] if top_cats else []
        common_cats2_str = "、".join(common_cats2[:1]) if common_cats2 else ""
        
        # 构建多轮对话：优先使用商品名称而非ID
        
        # 第一轮：用户提供历史
        user_msg_content = user_message_templates[0](history, sid_to_item)
        
        # 第二轮：助手首次推荐
        rec1_display = rec1_title if rec1_title else rec1_sid
        assistant_msg1_content = random.choice(assistant_recommend_templates)(
            rec1_display,
            generate_reasoning(history_categories, rec1_category, rec1_title, common_cats1_str, cat_freq_dict)
        )
        
        # 第三轮：用户反馈
        user_feedback_content = user_feedback_templates[0](rec1_sid, sid_to_item)
        
        # 第四轮：助手调整推荐
        rec2_display = rec2_title if rec2_title else rec2_sid
        assistant_msg2_content = random.choice(assistant_adjust_templates)(
            rec2_display,
            generate_adjust_reasoning(history_categories, rec1_cat, rec2_category, rec2_title, common_cats2_str, cat_freq_dict)
        )
        
        messages = [
            {
                "role": "user",
                "content": user_msg_content
            },
            {
                "role": "assistant",
                "content": assistant_msg1_content
            },
            {
                "role": "user",
                "content": user_feedback_content
            },
            {
                "role": "assistant",
                "content": assistant_msg2_content
            }
        ]
        
        samples.append({
            "task": "multi_turn_dialogue_recommendation",
            "messages": messages
        })
    
    print(f"生成了 {len(samples):,} 个任务14样本")
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

    # 加载数据
    items_df, sequences_df = load_data(DATA_DIR, CATEGORY)

    # 只生成总的all_samples
    all_samples = []
    all_samples.extend(task1_semantic_anchoring(items_df))
    all_samples.extend(task2_semantic_recovery(items_df))
    all_samples.extend(task3_sequence_semantic_understanding(sequences_df, items_df))
    all_samples.extend(task4_sequence_prediction(sequences_df))
    all_samples.extend(task5_mixed_space_alignment(items_df))
    all_samples.extend(task6_co_purchase_analysis(sequences_df, items_df))
    all_samples.extend(task7_recommendation_reasoning(sequences_df, items_df))
    all_samples.extend(task8_topk_candidate_recommendation(sequences_df, items_df))
    all_samples.extend(task9_preference_qa(sequences_df, items_df))
    all_samples.extend(task10_constraint_aware_recommendation(sequences_df, items_df))
    all_samples.extend(task11_counterfactual_explanation(sequences_df, items_df))
    all_samples.extend(task12_composite_recommend_and_reason(sequences_df, items_df))
    all_samples.extend(task13_feedback_aware_recommendation(sequences_df, items_df))
    all_samples.extend(task14_multi_turn_dialogue_recommendation(sequences_df, items_df))

    # 只保存总的合并数据
    save_samples(all_samples, OUTPUT_DIR / "sft_train_all_tasks_combined.jsonl")
    
    # 打印统计信息
    print("\n" + "="*50)
    print("数据生成完成！统计信息：")
    print("="*50)
    task_counts = {}
    for sample in all_samples:
        task = sample["task"]
        task_counts[task] = task_counts.get(task, 0) + 1
    
    for task, count in sorted(task_counts.items()):
        print(f"{task}: {count:,} 个样本")
    print(f"\n总计: {len(all_samples):,} 个样本")
    print(f"\n数据保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

