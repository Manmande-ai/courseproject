# -*- coding: utf-8 -*-
"""
LLM 微调模型推理预测 -> 生成官方提交格式 Result.csv
====================================================

支持两种推理后端:
  1. transformers (默认, 无需额外依赖, 速度较慢)
  2. vLLM        (需 pip install vllm, GPU 利用率高, 速度快 5-10x)

依赖训练产物:
  - 全参 SFT: 直接指向 output_dir (如 ./saves/qwen7b_full_sft)
  - LoRA:     指向合并后的模型目录, 或用 adapter_name_or_path 加载

运行方式
--------
  # transformers 后端(全参 SFT 或已合并 LoRA)
  python src/llm/predict_llm.py --model_path ./saves/qwen7b_full_sft

  # LoRA 未合并, 需指定基础模型 + adapter
  python src/llm/predict_llm.py --model_path Qwen/Qwen3-14B-Instruct \\
      --adapter_path ./saves/qwen14b_lora

  # vLLM 后端(推荐批量推理)
  python src/llm/predict_llm.py --model_path ./saves/qwen14b_merged --backend vllm

输出
----
  data/llm/Result.csv  (无表头 / 无 BOM UTF-8)
  列顺序: id, AspectTerms, OpinionTerms, Categories, Polarities
  每个测试 id 至少出现一行; 无四元组则全 '_'
"""

import argparse
import ast
import re
import sys
import time
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# 路径: 把 src/ 加入 sys.path, 复用 config 与 build_sft_dataset 的 Prompt
# ----------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC_DIR = _HERE.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from config import (  # noqa: E402
    LLM_DIR, LLM_RESULT_PATH, LLM_LOG_PATH, TEST_REVIEWS_PATH,
)
from llm.build_sft_dataset import INSTRUCTION  # noqa: E402  复用训练时的 Prompt

# ----------------------------------------------------------------------
# 合法类别与极性白名单(与训练标签体系一致)
# ----------------------------------------------------------------------
VALID_CATEGORIES = {
    "整体", "使用体验", "功效", "价格", "物流", "气味", "包装",
    "真伪", "服务", "其他", "成分", "尺寸", "新鲜度",
}
VALID_POLARITIES = {"正面", "负面", "中性"}

# 四元组字段固定顺序(与训练数据构造一致)
QUAD_KEYS = ("AspectTerms", "OpinionTerms", "Categories", "Polarities")


# ======================================================================
# Prompt 构造
# ======================================================================
def build_prompt(review: str) -> str:
    """拼接推理 Prompt: instruction + 评论原文。"""
    return INSTRUCTION + review


# ======================================================================
# 输出解析: 字符串 -> list[dict]
# ======================================================================
def parse_output(raw: str) -> list:
    """将模型生成的字符串解析为四元组列表。

    优先用 ast.literal_eval(兼容单引号); 失败则用正则逐个提取 {...} 再解析。
    """
    raw = (raw or "").strip()
    if not raw:
        return []

    # 1. 直接解析整个字符串
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, list):
            return obj
    except (SyntaxError, ValueError, TypeError):
        pass

    # 2. 兜底: 正则提取所有 {...} 块
    quads = []
    for m in re.finditer(r"\{[^{}]*\}", raw):
        try:
            d = ast.literal_eval(m.group(0))
            if isinstance(d, dict):
                quads.append(d)
        except (SyntaxError, ValueError, TypeError):
            continue
    return quads


# ======================================================================
# 四元组清洗: 白名单 + 去空 + 去重
# ======================================================================
def clean_quad(q: dict) -> dict:
    """清洗单个四元组: 去空白、类别/极性白名单过滤。"""
    aspect = str(q.get("AspectTerms") or "").strip()
    opinion = str(q.get("OpinionTerms") or "").strip()
    category = str(q.get("Categories") or "").strip()
    polarity = str(q.get("Polarities") or "").strip()

    if category not in VALID_CATEGORIES:
        category = "其他"
    if polarity not in VALID_POLARITIES:
        polarity = "中性"

    return {
        "AspectTerms": aspect,
        "OpinionTerms": opinion,
        "Categories": category,
        "Polarities": polarity,
    }


def clean_quads(quads: list) -> list:
    """清洗并去重四元组列表, 过滤掉观点词为空的无效四元组。"""
    seen = set()
    cleaned = []
    for q in quads:
        if not isinstance(q, dict):
            continue
        cq = clean_quad(q)
        # 观点词为空的四元组无意义, 跳过
        if not cq["OpinionTerms"]:
            continue
        key = (cq["AspectTerms"], cq["OpinionTerms"], cq["Categories"], cq["Polarities"])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(cq)
    return cleaned


# ======================================================================
# 推理后端
# ======================================================================
def predict_transformers(model_path: str, adapter_path: str | None,
                         reviews: dict[int, str], max_new_tokens: int) -> dict[int, list]:
    """transformers 后端: 逐条推理。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[加载] tokenizer: %s" % model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("[加载] model: %s" % model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel
        print("[加载] LoRA adapter: %s" % adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # 合并后推理更快
    model.eval()

    results = {}
    n = len(reviews)
    t0 = time.time()
    for i, (rid, review) in enumerate(reviews.items(), 1):
        prompt = build_prompt(review)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = out[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        results[rid] = clean_quads(parse_output(text))
        if i % 50 == 0 or i == n:
            print("[推理] %d/%d  已耗时 %.1fs" % (i, n, time.time() - t0))
    return results


def predict_vllm(model_path: str, adapter_path: str | None,
                 reviews: dict[int, str], max_new_tokens: int) -> dict[int, list]:
    """vLLM 后端: 批量推理, 速度快。"""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if adapter_path:
        print("[加载] vLLM 模型(基础): %s" % model_path)
        print("[加载] LoRA adapter: %s" % adapter_path)
        llm = LLM(model=model_path, trust_remote_code=True,
                  dtype="bfloat16", enable_lora=True, max_lora_rank=64)
        lora_req = LoRARequest("lora_adapter", 1, adapter_path)
    else:
        print("[加载] vLLM 模型: %s" % model_path)
        llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16")
        lora_req = None

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    ids = list(reviews.keys())
    prompts = [build_prompt(reviews[rid]) for rid in ids]

    print("[推理] vLLM 批量推理 %d 条评论..." % len(ids))
    t0 = time.time()
    if lora_req:
        outputs = llm.generate(prompts, sp, lora_request=lora_req)
    else:
        outputs = llm.generate(prompts, sp)

    results = {}
    for rid, out in zip(ids, outputs):
        text = out.outputs[0].text.strip()
        results[rid] = clean_quads(parse_output(text))
    print("[推理] 完成, 耗时 %.1fs" % (time.time() - t0))
    return results


# ======================================================================
# 结果写出: 官方提交格式
# ======================================================================
def _term(t) -> str:
    """None / 空字符串 -> '_', 供官方提交格式使用。"""
    if t is None or t == "" or (isinstance(t, float) and pd.isna(t)):
        return "_"
    return str(t)


def write_result_csv(pred: dict[int, list]) -> Path:
    """按官方格式写出 Result.csv: 无表头 / 无 BOM UTF-8。

    每个测试 id 必出现, 空结果行全 '_'; 列序 id,AspectTerms,OpinionTerms,Categories,Polarities。
    """
    te = pd.read_csv(TEST_REVIEWS_PATH, encoding="utf-8")
    ids = te["id"].astype(int).tolist()

    rows = []
    for rid in ids:
        quads = pred.get(rid, [])
        if not quads:
            rows.append((rid, "_", "_", "_", "_"))
            continue
        for q in quads:
            rows.append((
                rid,
                _term(q["AspectTerms"]),
                _term(q["OpinionTerms"]),
                q["Categories"],
                q["Polarities"],
            ))

    out = pd.DataFrame(rows, columns=["id", "AspectTerms", "OpinionTerms",
                                      "Categories", "Polarities"])
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(LLM_RESULT_PATH, index=False, header=False, encoding="utf-8")

    # 校验 id 覆盖
    out_ids = set(out["id"])
    assert out_ids == set(ids), \
        "结果 id(%d) 与测试集 id(%d) 不一致" % (len(out_ids), len(set(ids)))
    return LLM_RESULT_PATH


def print_stats(pred: dict[int, list]) -> None:
    """打印预测四元组的分布统计。"""
    from collections import Counter
    cat_cnt, pol_cnt = Counter(), Counter()
    n_empty_aspect = n_quads = 0
    for quads in pred.values():
        for q in quads:
            n_quads += 1
            cat_cnt[q["Categories"]] += 1
            pol_cnt[q["Polarities"]] += 1
            if not q["AspectTerms"]:
                n_empty_aspect += 1
    print("[统计] 预测四元组 %d 个, 平均 %.2f 个/评论"
          % (n_quads, n_quads / max(len(pred), 1)))
    print("[统计] 空属性词(整体) %d (%.1f%%)"
          % (n_empty_aspect, 100 * n_empty_aspect / max(n_quads, 1)))
    print("[统计] 类别分布: %s" % dict(cat_cnt.most_common()))
    print("[统计] 极性分布: %s" % dict(pol_cnt.most_common()))


# ======================================================================
# 主流程
# ======================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM 微调模型推理预测")
    p.add_argument("--model_path", required=True,
                   help="模型路径(全参 SFT 输出目录 / 已合并 LoRA 目录 / 基础模型名)")
    p.add_argument("--adapter_path", default=None,
                   help="LoRA adapter 路径(仅未合并时需要)")
    p.add_argument("--backend", choices=["transformers", "vllm"],
                   default="transformers", help="推理后端 (默认 transformers)")
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="最大生成 token 数 (默认 512)")
    p.add_argument("--output", default=str(LLM_RESULT_PATH),
                   help="输出 Result.csv 路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()

    print("=" * 70)
    print("LLM 微调模型推理预测")
    print("=" * 70)
    print("[配置] model_path=%s" % args.model_path)
    print("[配置] adapter_path=%s" % args.adapter_path)
    print("[配置] backend=%s, max_new_tokens=%d" % (args.backend, args.max_new_tokens))

    # 读取测试集
    te = pd.read_csv(TEST_REVIEWS_PATH, encoding="utf-8")
    reviews = dict(zip(te["id"].astype(int), te["Reviews"].astype(str)))
    print("[数据] 测试评论 %d 条" % len(reviews))

    # 推理
    print("-" * 70)
    if args.backend == "vllm":
        pred = predict_vllm(args.model_path, args.adapter_path, reviews, args.max_new_tokens)
    else:
        pred = predict_transformers(args.model_path, args.adapter_path, reviews, args.max_new_tokens)

    # 统计
    print("-" * 70)
    print_stats(pred)

    # 写出结果
    print("-" * 70)
    path = write_result_csv(pred)
    n_rows = sum(max(len(v), 1) for v in pred.values())
    print("[输出] 提交文件(无表头 / 无 BOM UTF-8): %s" % path)
    print("[输出] 共 %d 行, 覆盖 %d 个测试 id" % (n_rows, len(reviews)))
    print("[总耗时] %.1f 秒" % (time.time() - t_start))

    # 追加日志
    LLM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LLM_LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n[推理] %s  backend=%s  model=%s  耗时=%.1fs  四元组=%d  行数=%d  -> %s\n"
                % (time.strftime("%Y-%m-%d %H:%M:%S"), args.backend, args.model_path,
                   time.time() - t_start, sum(len(v) for v in pred.values()), n_rows, path))


if __name__ == "__main__":
    main()
