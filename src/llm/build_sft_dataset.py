# -*- coding: utf-8 -*-
"""
LLM SFT 数据构造脚本
====================

职责:
  将 Train_reviews.csv 与 Train_labels.csv 聚合成 LLaMA-Factory 可直接读取的
  alpaca 格式 JSON(sft_train.json), 并生成对应的 dataset_info.json。

聚合规则:
  1. 按 id 对 Train_labels.csv 做 groupby, 同一评论的多个四元组合并为一个 list;
  2. AspectTerms 中的占位符 "_" 统一转为空字符串 ""(与 Prompt 示例一致);
  3. 丢弃偏移列(A_start/A_end/O_start/O_end), 生成式抽取不需要位置信息;
  4. 输出字段顺序固定: AspectTerms, OpinionTerms, Categories, Polarities;
  5. 用 str(list[dict]) 输出单引号风格字符串, 推理时用 ast.literal_eval 解析。

运行:
  python -m src.llm.build_sft_dataset
  或在 src 目录下: python llm/build_sft_dataset.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

# 兼容两种运行方式: 作为模块运行 / 直接运行脚本
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (  # noqa: E402
    LLM_DIR,
    LLM_SFT_TRAIN_PATH,
    LLM_DATASET_INFO_PATH,
    TRAIN_REVIEWS_PATH,
    TRAIN_LABELS_PATH,
)

# ----------------------------------------------------------------------
# Prompt 模板(固定写入 instruction 字段, 与方案文档保持一致)
# ----------------------------------------------------------------------
INSTRUCTION = """你是一个聪明的AI电商助手，现在你需要帮我在商品评论中抽取：
1.商品属性特征（AspectTerms），并判断它的种类（Categories）。种类包括：['整体','使用体验','功效','价格','物流','气味','包装','真伪','服务','其他','成分','尺寸','新鲜度']；
2.消费者观点（OpinionTerms），并确认情感极性（Polarities）。情感极性包括：['正面', '负面', '中性']。
供你参考的评论示例：'很好，遮暇功能差一些，总体还不错'
输出：[{'AspectTerms': '', 'OpinionTerms': '很好', 'Categories': '整体', 'Polarities': '正面'}, {'AspectTerms': '遮暇功能', 'OpinionTerms': '差一些', 'Categories': '功效', 'Polarities': '负面'}, {'AspectTerms': '', 'OpinionTerms': '还不错', 'Categories': '整体', 'Polarities': '正面'}]
下面是需要你抽取的评论："""

# 四元组字段固定顺序
QUAD_KEYS = ("AspectTerms", "OpinionTerms", "Categories", "Polarities")


def build_quad(row: pd.Series) -> dict:
    """将一行标签转为四元组 dict, AspectTerms 的 "_" 转为空字符串。"""
    aspect = row["AspectTerms"]
    aspect = "" if (pd.isna(aspect) or str(aspect).strip() == "_") else str(aspect).strip()
    return {
        "AspectTerms": aspect,
        "OpinionTerms": str(row["OpinionTerms"]).strip(),
        "Categories": str(row["Categories"]).strip(),
        "Polarities": str(row["Polarities"]).strip(),
    }


def main() -> None:
    # 1. 读取原始数据
    reviews = pd.read_csv(TRAIN_REVIEWS_PATH, encoding="utf-8-sig")
    labels = pd.read_csv(TRAIN_LABELS_PATH, encoding="utf-8-sig")

    # 2. 按 id 聚合四元组, 保持原始顺序
    labels["quad"] = labels.apply(build_quad, axis=1)
    quads_by_id = (
        labels.groupby("id", sort=True)["quad"]
        .apply(list)
        .to_dict()
    )

    # 3. 构造 alpaca 格式样本
    samples = []
    for _, row in reviews.iterrows():
        rid = row["id"]
        review = str(row["Reviews"]).strip()
        quads = quads_by_id.get(rid, [])
        # 用 str() 输出单引号风格, 与 Prompt 示例一致; 推理侧用 ast.literal_eval 解析
        output = str(quads)
        samples.append({
            "instruction": INSTRUCTION,
            "input": review,
            "output": output,
        })

    # 4. 写出 sft_train.json
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    with open(LLM_SFT_TRAIN_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    # 5. 写出 dataset_info.json(LLaMA-Factory 数据集注册)
    dataset_info = {
        "sft_train": {
            "file_name": LLM_SFT_TRAIN_PATH.name,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
            },
        }
    }
    with open(LLM_DATASET_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    # 6. 统计输出
    n_reviews = len(reviews)
    n_quads = sum(len(v) for v in quads_by_id.values())
    n_empty_aspect = sum(
        1 for quads in quads_by_id.values() for q in quads if not q["AspectTerms"]
    )
    print(f"[OK] 训练样本数: {len(samples)}")
    print(f"[OK] 评论数: {n_reviews}, 四元组总数: {n_quads}")
    print(f"[OK] 空属性词(整体)四元组数: {n_empty_aspect} ({n_empty_aspect / max(n_quads, 1):.1%})")
    print(f"[OK] 数据文件: {LLM_SFT_TRAIN_PATH}")
    print(f"[OK] 注册文件: {LLM_DATASET_INFO_PATH}")


if __name__ == "__main__":
    main()
