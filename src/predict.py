# -*- coding: utf-8 -*-
"""
Double-Propagation-ACOS 预测与提交文件生成
==========================================

用全量训练集归纳规则知识, 对测试集推理 ACOS 四元组,
按官方格式写出 data/baseline/Result.csv (无表头 / 无 BOM UTF-8)。

运行方式
--------
    python predict.py

输出文件
--------
    data/baseline/Result.csv          提交文件(官方格式)
    data/baseline/lexicon_opinion.csv 观点词典(训练产物)
    data/baseline/lexicon_aspect.csv  方面词典(训练产物)
"""

import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

# 复用 model_baseline 的全部组件(导入即加载 YAML 配置)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_baseline as mb  # noqa: E402
from config import TEST_REVIEWS_PATH, BASELINE_DIR as OUT_DIR  # noqa: E402


def train_knowledge(reviews_tr, labels):
    """用全量训练集归纳规则知识(观点极性、方面->类别、隐式观点等)。"""
    print("[训练] 从 %d 条标注中归纳规则知识..." % len(labels))
    kn = mb.build_knowledge(labels)
    mb.compute_annot_rates(
        kn, labels,
        zip(reviews_tr["id"].astype(int), reviews_tr["Reviews"]))
    print("[训练] 观点词 %d(含手工补充 %d), 方面词 %d, 隐式观点规则 %d 条"
          % (len(kn.seed_opinions), getattr(kn, "_n_manual_seed", 0),
             len(kn.aspect_vocab), len(kn.implicit_opinion)))
    n_gated = sum(1 for o in kn.opinion_vocab
                  if kn.o_occ_n.get(o, 0) >= mb.IMPLICIT_RATE_OCC_MIN
                  and kn.o_annot_rate.get(o, 1.0) < mb.IMPLICIT_RATE_MIN)
    print("[训练] 隐式观点门控: 标注率<%.2f 且出现>=%d 文档的观点词 %d 个"
          % (mb.IMPLICIT_RATE_MIN, mb.IMPLICIT_RATE_OCC_MIN, n_gated))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mb._dump_lexicons(kn)
    print("[训练] 词典已落盘: %s/{lexicon_opinion,lexicon_aspect}.csv" % OUT_DIR)
    return kn


def predict(kn, rev_map_te):
    """对测试集执行: spaCy 解析 -> 双传播 -> 四元组组装。"""
    vocab_words = kn.aspect_vocab | kn.opinion_vocab | {
        w for ws in mb.MANUAL_SEED.values() for w in ws}
    parser = mb.ZhParser(vocab_words=vocab_words)
    test_rows = [(i, rev_map_te[i]) for i in sorted(rev_map_te)]
    print("[推理] 解析 + 双传播抽取 %d 条测试评论..." % len(test_rows))
    pred, meta = mb.predict_corpus(test_rows, kn, parser)

    print("[传播] 初始种子 |O|=%d |A|=%d -> 收敛 |O|=%d |A|=%d; 耗时 %.1fs"
          % (meta["n_seed_o"], meta["n_seed_a"],
             meta["final_O"], meta["final_A"], meta["elapsed"]))
    print("[传播] 迭代轨迹:")
    for r in meta["trace"]:
        print("         #%d  O=%-4d A=%-4d  +O=%-3d +A=%-3d | "
              "R1=%-3d R2=%-3d R3=%-3d R4=%-3d"
              % (r["iter"], r["|O|"], r["|A|"], r["new_O"], r["new_A"],
                 r["R1_O->A"], r["R2_A->O"], r["R3_A->A"], r["R4_O->O"]))
    return pred


def write_result(pred, rev_map_te):
    """按官方格式写出 Result.csv (无表头 / 无 BOM UTF-8)。

    列顺序: id, AspectTerm, OpinionTerm, Category, Polarity
    空结果行: id,_,_,_,_
    """
    te = pd.read_csv(TEST_REVIEWS_PATH, encoding="utf-8")
    ids = te["id"].astype(int).tolist()
    rows = []
    for rid in ids:
        qs = sorted(pred.get(rid, set()),
                    key=lambda q: (q[1] or "", q[2] or ""))
        if not qs:
            rows.append((rid, "_", "_", "_", "_"))
        for a, c, o, s in qs:
            rows.append((rid, mb._term(a), mb._term(o), c, s))
    out = pd.DataFrame(rows, columns=["id", "AspectTerms", "OpinionTerms",
                                      "Categories", "Polarities"])
    path = OUT_DIR / "Result.csv"
    out.to_csv(path, index=False, header=False, encoding="utf-8")

    # 校验 id 覆盖
    out_ids = set(out["id"])
    assert out_ids == set(ids), \
        "结果 id(%d) 与测试集 id(%d) 不一致" % (len(out_ids), len(set(ids)))
    return path, len(rows)


def print_stats(pred, rev_map_te):
    """打印预测四元组的分布统计。"""
    cat_cnt, pol_cnt = Counter(), Counter()
    n_imp_a = n_imp_o = n_quads = 0
    for qs in pred.values():
        for a, c, o, s in qs:
            n_quads += 1
            cat_cnt[c] += 1
            pol_cnt[s] += 1
            n_imp_a += int(a is None)
            n_imp_o += int(o is None)
    print("[统计] 预测四元组 %d 个, 平均 %.2f 个/评论"
          % (n_quads, n_quads / max(len(rev_map_te), 1)))
    print("[统计] 隐式方面 %d (%.1f%%), 隐式观点 %d (%.1f%%)"
          % (n_imp_a, 100 * n_imp_a / max(n_quads, 1),
             n_imp_o, 100 * n_imp_o / max(n_quads, 1)))
    print("[统计] 类别分布: %s" % dict(cat_cnt.most_common()))
    print("[统计] 极性分布: %s" % dict(pol_cnt.most_common()))


def main():
    t_start = time.time()
    print("=" * 70)
    print("Double-Propagation-ACOS 预测与提交文件生成")
    print("=" * 70)

    # ---------- 数据加载 ----------
    reviews_tr, labels, reviews_te = mb.load_raw_data()
    rev_map_tr = dict(zip(reviews_tr["id"].astype(int), reviews_tr["Reviews"]))
    rev_map_te = dict(zip(reviews_te["id"].astype(int), reviews_te["Reviews"]))
    print("[数据] 训练评论 %d 条 / 标签 %d 行 / 测试评论 %d 条"
          % (len(rev_map_tr), len(labels), len(rev_map_te)))

    # ---------- 训练(规则归纳) ----------
    print("-" * 70)
    kn = train_knowledge(reviews_tr, labels)

    # ---------- 测试推理 ----------
    print("-" * 70)
    pred = predict(kn, rev_map_te)

    # ---------- 统计 ----------
    print("-" * 70)
    print_stats(pred, rev_map_te)

    # ---------- 生成提交文件 ----------
    print("-" * 70)
    result_path, n_rows = write_result(pred, rev_map_te)
    print("[输出] 提交文件(无 BOM UTF-8 / 无表头): %s" % result_path)
    print("[输出] 共 %d 行, 覆盖 %d 个测试 id" % (n_rows, len(rev_map_te)))
    print("[总耗时] %.1f 秒" % (time.time() - t_start))


if __name__ == "__main__":
    main()
