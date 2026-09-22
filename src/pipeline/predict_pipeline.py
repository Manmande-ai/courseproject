# -*- coding: utf-8 -*-
"""
方案C - 管道式 ACOS 抽取: 串联 Stage1 + Stage2 推理 / 评估
============================================================

功能
----
  1) predict  (默认): 对 TEST 评论跑 Stage1(对抽取) -> Stage2(分类) -> Result.csv
  2) eval-dev         : 在 dev 切分上跑完整管道, 算四元组级 P/R/F1 + 误差分析

模型串联
--------
  Stage1: PairExtractModel (data/pipeline/pair_extractor.pt + pair_meta.json)
          -> decode_review -> pairs_from_grid(tau=best_threshold)
  Stage2: PairClassifier  (data/pipeline/pair_classifier.pt)
          -> 交叉编码 [CLS] review [SEP] 方面词:a 观点词:o [SEP] -> (cat, pol)

输出格式 (与 data/bert/Result.csv 一致, 无表头)
----
  id,AspectTerms,OpinionTerms,Categories,Polarities   (一行一个四元组)
  隐式 aspect/opinion 用 '_' ; 每条评论至少一行(安全兜底)

运行
----
  python src/pipeline/predict_pipeline.py                  # 预测测试集
  python src/pipeline/predict_pipeline.py --eval-dev       # dev 四元组 F1 评估
  python src/pipeline/predict_pipeline.py --source train   # 预测训练集(分析用)
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 路径: 同时把 src/ 和 src/pipeline/ 加入 sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_THIS_DIR))

from config import (  # noqa: E402
    BASE_DIR, TRAIN_REVIEWS_PATH, TRAIN_LABELS_PARSED_PATH, TEST_REVIEWS_PATH,
    PIPELINE_DIR, PIPELINE_PAIR_WEIGHTS, PIPELINE_PAIR_META,
    PIPELINE_CLF_WEIGHTS, PIPELINE_CLF_META, PIPELINE_RESULT_PATH,
)
import extract_pair as s1   # noqa: E402  Stage1 模型/decode/配置
import classify_pair as s2  # noqa: E402  Stage2 模型/配置


# ======================================================================
# 1. 加载模型
# ======================================================================
def load_stage1(device):
    meta = json.loads(PIPELINE_PAIR_META.read_text(encoding="utf-8"))
    model = s1.PairExtractModel().to(device)
    model.load_state_dict(torch.load(PIPELINE_PAIR_WEIGHTS, map_location=device))
    model.eval()
    tau = meta.get("best_threshold", 0.5)
    print("[Stage1] 已加载 %s  tau=%.2f  (dev pair F1=%.4f)"
          % (PIPELINE_PAIR_WEIGHTS.name, tau,
             (meta.get("best_dev") or {}).get("F1", 0.0)))
    return model, tau


def load_stage2(device):
    meta = json.loads(PIPELINE_CLF_META.read_text(encoding="utf-8"))
    model = s2.PairClassifier().to(device)
    model.load_state_dict(torch.load(PIPELINE_CLF_WEIGHTS, map_location=device))
    model.eval()
    bm = meta.get("best_dev") or {}
    print("[Stage2] 已加载 %s  (dev cat_mF1=%.4f pol_mF1=%.4f)"
          % (PIPELINE_CLF_WEIGHTS.name,
             bm.get("cat_macro_f1", 0.0), bm.get("pol_macro_f1", 0.0)))
    return model


# ======================================================================
# 2. Stage1 推理: 评论 -> 候选对
# ======================================================================
@torch.no_grad()
def review_to_pairs(s1_model, tokenizer, text, device, tau):
    """跑 Stage1, 返回 list[(a_text|'_', o_text|'_')]。"""
    dec = s1.decode_review(s1_model, tokenizer, text, device, s1.MAX_LEN)
    pairs, _ = s1.pairs_from_grid(
        dec["asp_texts"], dec["opn_texts"], dec["probs"], tau)
    return pairs


# ======================================================================
# 3. Stage2 批量推理: (review, pair) -> (cat, pol)
# ======================================================================
@torch.no_grad()
def classify_pairs(s2_model, tokenizer, items, device, max_len, batch_size=32):
    """items: list[(review_text, a_text|None, o_text|None)]。
    返回 list[(cat, pol)]。隐式(None/''/'_')用 marker 占位。"""
    s2_model.eval()
    results = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        texts = [c[0] for c in chunk]
        pair_texts = [s2.build_pair_text(
            (None if a in (None, "", "_") else a),
            (None if o in (None, "", "_") else o))
            for _, a, o in chunk]
        enc = tokenizer(texts, text_pair=pair_texts, max_length=max_len,
                        truncation=True, padding=True, return_tensors="pt",
                        add_special_tokens=True)
        clo, plo = s2_model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            enc.get("token_type_ids",
                    torch.zeros_like(enc["input_ids"])).to(device))
        cats = clo.argmax(-1).cpu().tolist()
        pols = plo.argmax(-1).cpu().tolist()
        results.extend((s2.CATEGORIES[c], s2.POLARITIES[p])
                       for c, p in zip(cats, pols))
    return results


# ======================================================================
# 4. 完整管道: 评论 -> 四元组
# ======================================================================
def predict_reviews(s1_model, s2_model, tokenizer, reviews, device, tau,
                   verbose_every=200):
    """reviews: list[(rid, text)]。返回 dict[rid] -> list[(a,o,cat,pol)]。"""
    # ---- Pass 1: Stage1 抽对 ----
    all_items = []      # (idx_in_review, rid, a_text, o_text)
    rid_pairs = {}      # rid -> list[(a_text|'_', o_text|'_')]
    t0 = time.time()
    for k, (rid, text) in enumerate(reviews):
        pairs = review_to_pairs(s1_model, tokenizer, text, device, tau)
        rid_pairs[rid] = pairs
        for a, o in pairs:
            all_items.append((rid, text, a, o))
        if verbose_every and (k + 1) % verbose_every == 0:
            el = time.time() - t0
            print("  [Stage1] %d/%d 评论, %.0fs" % (k + 1, len(reviews), el),
                  flush=True)
    print("[Stage1] 完成 %d 评论, 抽出 %d 对 (人均 %.2f)"
          % (len(reviews), len(all_items),
             len(all_items) / max(len(reviews), 1)))

    # ---- Pass 2: Stage2 批量分类 ----
    clf_items = [(text, a, o) for _, text, a, o in all_items]
    labels = classify_pairs(s2_model, tokenizer, clf_items, device, s2.MAX_LEN)

    # ---- 拼四元组 ----
    out = {rid: [] for rid in rid_pairs}
    for (rid, _, a, o), (cat, pol) in zip(all_items, labels):
        a_out = a if a and a != "_" else "_"
        o_out = o if o and o != "_" else "_"
        out[rid].append((a_out, o_out, cat, pol))
    print("[Stage2] 分类完成")
    return out


# ======================================================================
# 5. 写 Result.csv (无表头, 一行一四元组, 隐式 '_')
# ======================================================================
def write_result(out, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for rid in sorted(out):
        for a, o, cat, pol in out[rid]:
            lines.append("%s,%s,%s,%s,%s" % (rid, a, o, cat, pol))
    # 用无 BOM 的 UTF-8 (utf-8-sig 会写入 EF BB BF 头, 导致严格 UTF-8 解析报错)
    path.write_bytes(("\n".join(lines) + ("\n" if lines else ""))
                     .encode("utf-8"))
    print("[输出] %s  (%d 行四元组, %d 评论)"
          % (path, len(lines), len(out)))


# ======================================================================
# 6. dev 评估: 完整管道四元组级 P/R/F1
# ======================================================================
def gold_quad(rid, text, q):
    a = text[int(q["a_start"]):int(q["a_end"])] if not pd.isna(q["a_start"]) else "_"
    o = text[int(q["o_start"]):int(q["o_end"])] if not pd.isna(q["o_start"]) else "_"
    return (a, o, q["cat"], q["pol"])


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def eval_dev(s1_model, s2_model, tokenizer, device, tau):
    """在 dev 切分(与 Stage2 同 seed)上算四元组级 P/R/F1 + 误差分析。"""
    # ---- dev 切分 (复用 classify_pair 的 review 加载 + 同 seed) ----
    reviews = s2.load_review_samples()
    rng = random.Random(s2.SEED)
    rng.shuffle(reviews)
    n_dev = max(1, int(len(reviews) * s2.DEV_RATIO))
    dev_reviews = reviews[:n_dev]
    print("[eval-dev] dev 评论=%d" % len(dev_reviews))

    # gold 四元组 (review 级)
    gold_by_rid = {}
    for rid, text, quads in dev_reviews:
        gold_by_rid[rid] = set(gold_quad(rid, text, q) for q in quads)

    # ---- 跑管道 ----
    pred = predict_reviews(s1_model, s2_model, tokenizer,
                          [(rid, text) for rid, text, _ in dev_reviews],
                          device, tau, verbose_every=100)
    pred_by_rid = {rid: set(quads) for rid, quads in pred.items()}

    # ---- 四元组级 micro P/R/F1 ----
    tp = fp = fn = 0
    n_pred = n_gold = n_review_no_pred = 0
    err_types = Counter()
    for rid in gold_by_rid:
        g = gold_by_rid[rid]
        p = pred_by_rid.get(rid, set())
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        n_pred += len(p)
        n_gold += len(g)
        if not p:
            n_review_no_pred += 1
        # 误差归类: 对级 (a,o) 命中但 cat/pol 错
        for q in (p - g):
            for gg in g:
                if q[0] == gg[0] and q[1] == gg[1]:
                    if q[2] != gg[2]:
                        err_types["cat错"] += 1
                    if q[3] != gg[3]:
                        err_types["pol错"] += 1
                    break
            else:
                err_types["对错(span)"] += 1

    p, r, f1 = prf(tp, fp, fn)
    print("")
    print("=" * 72)
    print("[eval-dev] 四元组级结果 (严格集合匹配)")
    print("=" * 72)
    print("  P=%.4f  R=%.4f  F1=%.4f" % (p, r, f1))
    print("  TP=%d FP=%d FN=%d  pred=%d gold=%d  pred/gold=%.2f"
          % (tp, fp, fn, n_pred, n_gold, n_pred / max(n_gold, 1)))
    print("  零抽取评论=%d/%d (%.1f%%)"
          % (n_review_no_pred, len(dev_reviews),
             100 * n_review_no_pred / max(len(dev_reviews), 1)))
    print("  误差归类: %s" % dict(err_types))
    print("=" * 72)
    return {"P": p, "R": r, "F1": f1, "tp": tp, "fp": fp, "fn": fn,
            "err_types": dict(err_types), "n_review_no_pred": n_review_no_pred}


# ======================================================================
# 7. 主入口
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="方案C 管道式推理 (Stage1+Stage2)")
    parser.add_argument("--eval-dev", action="store_true",
                        help="评估 dev 切分四元组级 P/R/F1")
    parser.add_argument("--source", choices=["test", "train"], default="test",
                        help="predict 模式的数据源")
    parser.add_argument("--out", type=str, default=None,
                        help="pipeline_Result.csv 输出路径 (默认 data/pipeline/pipeline_Result.csv)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[环境] device=%s" % device)
    tokenizer = AutoTokenizer.from_pretrained(s1.PRETRAINED)

    s1_model, tau = load_stage1(device)
    s2_model = load_stage2(device)

    if args.eval_dev:
        eval_dev(s1_model, s2_model, tokenizer, device, tau)
        return

    # ---- predict 模式 ----
    if args.source == "test":
        df = pd.read_csv(TEST_REVIEWS_PATH, encoding="utf-8-sig")
    else:
        df = pd.read_csv(TRAIN_REVIEWS_PATH, encoding="utf-8-sig")
    reviews = list(zip(df["id"].astype(int).tolist(),
                       df["Reviews"].astype(str).tolist()))
    print("[数据] %s 评论=%d" % (args.source, len(reviews)))

    out = predict_reviews(s1_model, s2_model, tokenizer, reviews, device, tau)
    out_path = args.out or str(PIPELINE_RESULT_PATH)
    write_result(out, out_path)


if __name__ == "__main__":
    main()
