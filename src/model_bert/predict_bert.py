# -*- coding: utf-8 -*-
"""
BERT+CRF 测试集推理 -> 生成官方提交格式 Result.csv
====================================================

依赖训练脚本 src/model_bert/model_bert.py 产出的:
  data/bert/bert_crf.pt       模型权重
  data/bert/label_vocab.json  标签字典

运行方式
--------
  python src/model_bert/predict_bert.py

输出
----
  data/bert/Result.csv  (无表头 / 无 BOM UTF-8)
  列顺序: id, AspectTerm, OpinionTerm, Category, Polarity
  每个测试 id 至少出现一行; 无四元组则全 '_'
"""

import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

# ----------------------------------------------------------------------
# 路径: 先把 model_bert 目录与上层 src/ 都加入 sys.path
# ----------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC_DIR = _HERE.parent
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_HERE))

import model_bert as mb   # noqa: E402  (复用训练脚本的全部组件)
from config import (      # noqa: E402
    BERT_DIR, TEST_REVIEWS_PROC_PATH, BERT_LOG_PATH,
)


def _term(t):
    """None / 空字符串 -> '_', 供官方提交格式使用。"""
    if t is None or t == "" or (isinstance(t, float) and pd.isna(t)):
        return "_"
    return str(t)


def write_result_csv(test_path, pred):
    """按官方格式写出: 无表头 / 无 BOM UTF-8。
    每个测试 id 必出现, 空结果行全 '_'; 列序 id,Aspect,Opinion,Category,Polarity。
    pred: {rid: set[(a_text|None, cat, o_text|None, pol)]}
    """
    te = pd.read_csv(test_path, encoding="utf-8-sig")
    ids = te["id"].astype(int).tolist()
    rows = []
    for rid in ids:
        qs = pred.get(rid, set())
        # 排序保证可复现 (aspect 字典序)
        qs = sorted(qs, key=lambda q: (q[0] or "", q[1], q[2] or "", q[3]))
        if not qs:
            rows.append((rid, "_", "_", "_", "_"))
        for a, c, o, s in qs:
            rows.append((rid, _term(a), _term(o), c, s))
    out = pd.DataFrame(rows, columns=["id", "AspectTerms", "OpinionTerms",
                                     "Categories", "Polarities"])
    path = BERT_DIR / "Result.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, header=False, encoding="utf-8")
    # 校验 id 覆盖
    out_ids = set(out["id"])
    assert out_ids == set(ids), "结果 id 与测试集不一致"
    return path, len(out)


def main():
    weights = BERT_DIR / "bert_crf.pt"
    vocab_path = BERT_DIR / "label_vocab.json"
    if not weights.exists() or not vocab_path.exists():
        raise FileNotFoundError(
            "未找到训练产物 (%s / %s)。请先运行:\n"
            "  python src/model_bert/model_bert.py"
            % (weights, vocab_path))

    # 加载标签字典 (覆盖到 model_bert 模块全局, 确保与训练一致)
    vocab = mb.load_label_vocab(vocab_path)
    mb.BIO_TAGS = vocab["bio_tags"]
    mb.CATEGORIES = vocab["categories"]
    mb.POLARITIES = vocab["polarities"]
    mb.CAT2ID = {c: i for i, c in enumerate(mb.CATEGORIES)}
    mb.POL2ID = {p: i for i, p in enumerate(mb.POLARITIES)}
    mb.ID2CAT = {i: c for c, i in mb.CAT2ID.items()}
    mb.ID2POL = {i: p for p, i in mb.POL2ID.items()}
    pretrained = vocab.get("pretrained", mb.PRETRAINED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[环境] device=%s, pretrained=%s" % (device, pretrained))

    # 加载模型
    model = mb.BertCrfModel(pretrained=pretrained).to(device)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    print("[加载] 权重: %s" % weights)

    # 读测试集 (id 顺序与 Train 一致)
    te = pd.read_csv(TEST_REVIEWS_PROC_PATH, encoding="utf-8-sig")
    rev_map = dict(zip(te["id"].astype(int), te["Reviews"].astype(str)))
    ids = sorted(rev_map)

    print("[推理] %d 条测试评论..." % len(ids))
    t0 = time.time()
    pred = {}
    for rid in ids:
        text = rev_map[rid]
        pred[rid] = mb.predict_review_quads(
            model, tokenizer, text, device, mb.MAX_LEN)
    elapsed = time.time() - t0

    # 写 Result.csv
    path, n_rows = write_result_csv(TEST_REVIEWS_PROC_PATH, pred)
    n_q = sum(len(v) for v in pred.values())
    print("[完成] 推理耗时 %.1f s, 共 %d 条评论 / %d 个四元组, %d 行结果"
          % (elapsed, len(ids), n_q, n_rows))
    print("[输出] %s" % path)

    # 追加日志
    with open(BERT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n[推理] %s  耗时=%.1fs  四元组=%d  行数=%d  -> %s\n"
                % (time.strftime("%Y-%m-%d %H:%M:%S"), elapsed, n_q, n_rows, path))


if __name__ == "__main__":
    main()
