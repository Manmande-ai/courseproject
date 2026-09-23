# -*- coding: utf-8 -*-
"""
方案C - Stage1: (Aspect, Opinion) 对抽取 (管道式 ACOS 的第一步)
====================================================================

任务
----
对每条评论输出若干 (Aspect, Opinion) 对, 供 Stage2 逐对分类
Category / Polarity。隐式方面/观点用 '_' 表示(哨兵候选)。

模型
----
  bert-base-chinese
      |
      |-- (a) tag_proj -> CRF  : 5 标签 BIO, 抽 aspect / opinion span
      |-- (b) span MLP x2      : g(s,e)=MLP([h_s; h_e; mean_pool])
      |-- (c) 双仿射 PairScorer: 对候选格逐格二分类(多标签 sigmoid)
                               最后一行/列 = 隐式 A / 隐式 O 哨兵

候选格 (n_asp+1) x (n_opn+1), 训练时行/列来自 gold span(teacher forcing),
推理时来自 CRF 解码 span。右下角(双隐式)训练集中为 0 例, 推理时屏蔽。

数据
----
  评论用 data/raw/TRAIN/Train_reviews.csv 原文(含标点/原始空格),
  标签偏移来自 outputs/processed/Train_labels_parsed.csv。
  注意: 不能用 Train_reviews_processed.csv -- 其 Reviews 压缩了连续空格
  (78 条), 与按原文计算的字符偏移错位。

运行
----
  python src/pipeline/extract_pair.py                 # 完整训练 + dev 评估
  python src/pipeline/extract_pair.py --epochs 3      # 覆盖 epoch
  python src/pipeline/extract_pair.py --max-train 64 --max-dev 32  # 冒烟

输出
----
  data/pipeline/pair_extractor.pt    权重(state_dict)
  data/pipeline/pair_meta.json       标签/维度/最优阈值/dev 指标
  config/model_pipeline_trained.yaml 训练快照
  outputs/logs/pipeline_pair_report.txt
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# ----------------------------------------------------------------------
# 路径: 把 src/ 加入 sys.path 以便 import config
# ----------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR))
from config import (  # noqa: E402
    BASE_DIR, TRAIN_REVIEWS_PATH, TRAIN_LABELS_PARSED_PATH,
    PIPELINE_DIR, PIPELINE_CONFIG_PATH, PIPELINE_TRAINED_SNAPSHOT_PATH,
    PIPELINE_PAIR_WEIGHTS, PIPELINE_PAIR_META, PIPELINE_PAIR_LOG_PATH,
)

# ======================================================================
# 0. 默认超参 (可被 config/model_pipeline.yaml 覆盖)
# ======================================================================
PRETRAINED = "bert-base-chinese"
BIO_TAGS = ["O", "B-ASP", "I-ASP", "B-OPN", "I-OPN"]

SEED = 42
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 6
LR_BERT = 2e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_CLIP = 1.0
DROPOUT = 0.1
DEV_RATIO = 0.15

MLP_DIM = 200
PAIR_LOSS_WEIGHT = 1.0
PAIR_POS_WEIGHT = 2.0
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

TAG2ID = {t: i for i, t in enumerate(BIO_TAGS)}


# ======================================================================
# 1. YAML 配置
# ======================================================================
def load_config(path=PIPELINE_CONFIG_PATH, verbose=True):
    path = Path(path)
    if not path.exists():
        if verbose:
            print("[配置] %s 不存在, 使用代码内默认值" % path)
        return
    import yaml
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    g = globals()

    def _set(name, value, cast=None):
        try:
            g[name] = cast(value) if cast is not None else value
        except (TypeError, ValueError):
            if verbose:
                print("[配置] %s=%r 无效, 保留默认" % (name, value))

    if cfg.get("pretrained"):
        _set("PRETRAINED", str(cfg["pretrained"]))
    if (cfg.get("labels") or {}).get("bio_tags"):
        _set("BIO_TAGS", list(cfg["labels"]["bio_tags"]))
        g["TAG2ID"] = {t: i for i, t in enumerate(g["BIO_TAGS"])}

    hp = cfg.get("hyperparams") or {}
    for key, name, cast in (
        ("seed", "SEED", int), ("max_len", "MAX_LEN", int),
        ("batch_size", "BATCH_SIZE", int), ("epochs", "EPOCHS", int),
        ("lr_bert", "LR_BERT", float), ("lr_head", "LR_HEAD", float),
        ("weight_decay", "WEIGHT_DECAY", float),
        ("warmup_ratio", "WARMUP_RATIO", float), ("grad_clip", "GRAD_CLIP", float),
        ("dropout", "DROPOUT", float), ("dev_ratio", "DEV_RATIO", float),
    ):
        if hp.get(key) is not None:
            _set(name, hp[key], cast)

    p = cfg.get("pair") or {}
    for key, name, cast in (
        ("mlp_dim", "MLP_DIM", int),
        ("loss_weight", "PAIR_LOSS_WEIGHT", float),
        ("pos_weight", "PAIR_POS_WEIGHT", float),
    ):
        if p.get(key) is not None:
            _set(name, p[key], cast)
    if p.get("thresholds"):
        _set("THRESHOLDS", [float(x) for x in p["thresholds"]])

    if verbose:
        print("[配置] 已加载: %s" % path)


load_config()


# ======================================================================
# 2. 数据工具: 偏移换算 / BIO + 配对目标
# ======================================================================
def char_span_to_token_span(offsets, c_start, c_end):
    """字符偏移 [c_start, c_end) -> token 闭区间 (s, e); 不相交返回 None。"""
    s_tok = e_tok = None
    for i, (cs, ce) in enumerate(offsets):
        if cs == 0 and ce == 0:
            continue  # 特殊 token
        if cs < c_end and ce > c_start:
            if s_tok is None:
                s_tok = i
            e_tok = i
    if s_tok is None:
        return None
    return s_tok, e_tok


def mark_bio(labels, s, e, kind):
    labels[s] = TAG2ID["B-%s" % kind]
    for k in range(s + 1, e + 1):
        labels[k] = TAG2ID["I-%s" % kind]


def build_targets(text, quads, tokenizer, max_len=MAX_LEN):
    """单条评论 -> BIO 标签 + span 列表 + gold 对索引 + gold 对文本。

    返回:
      input_ids/attention_mask/labels
      asp_spans/opn_spans: list[(s,e)] token 闭区间(去重, 稳定顺序)
      gold_ij:  set[(i,j)], i==len(asp_spans) 表示隐式 A 哨兵行,
                j==len(opn_spans) 表示隐式 O 哨兵列
      gold_pairs: set[(a_text|None, o_text|None)], 与推理同口径
                  (经 token 偏移对齐; 截断丢失的显式 span 不计入)
      n_skip: 因截断无法对齐而丢弃的四元组数
    """
    enc = tokenizer(text, max_length=max_len, truncation=True,
                    return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    input_ids, attn = enc["input_ids"], enc["attention_mask"]
    labels = [TAG2ID["O"]] * len(input_ids)

    asp_map, opn_map = {}, {}
    asp_spans, opn_spans = [], []
    raw_pairs = []   # (a_tok|None, o_tok|None)
    n_skip = 0

    for q in quads:
        a_tok = o_tok = None
        if q.get("a_start") is not None and not pd.isna(q["a_start"]):
            a_tok = char_span_to_token_span(
                offsets, int(q["a_start"]), int(q["a_end"]))
            if a_tok is None:
                n_skip += 1
                continue
            if a_tok not in asp_map:
                asp_map[a_tok] = len(asp_spans)
                asp_spans.append(a_tok)
            mark_bio(labels, a_tok[0], a_tok[1], "ASP")
        if q.get("o_start") is not None and not pd.isna(q["o_start"]):
            o_tok = char_span_to_token_span(
                offsets, int(q["o_start"]), int(q["o_end"]))
            if o_tok is None:
                n_skip += 1
                continue
            if o_tok not in opn_map:
                opn_map[o_tok] = len(opn_spans)
                opn_spans.append(o_tok)
            mark_bio(labels, o_tok[0], o_tok[1], "OPN")
        raw_pairs.append((a_tok, o_tok))

    na, no = len(asp_spans), len(opn_spans)
    gold_ij = set()
    gold_pairs = set()
    for a_tok, o_tok in raw_pairs:
        i = asp_map[a_tok] if a_tok is not None else na
        j = opn_map[o_tok] if o_tok is not None else no
        gold_ij.add((i, j))

        def _slice(tok):
            s, e = tok
            return text[offsets[s][0]:offsets[e][1]]

        gold_pairs.add((
            _slice(a_tok) if a_tok is not None else None,
            _slice(o_tok) if o_tok is not None else None,
        ))

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "asp_spans": asp_spans,
        "opn_spans": opn_spans,
        "gold_ij": gold_ij,
        "gold_pairs": gold_pairs,
        "n_skip": n_skip,
    }


def extract_spans_from_tags(tags):
    """BIO id 序列 -> {'ASP':[(s,e)], 'OPN':[(s,e)]}。"""
    out = {"ASP": [], "OPN": []}
    i, n = 0, len(tags)
    while i < n:
        t = BIO_TAGS[tags[i]] if tags[i] < len(BIO_TAGS) else "O"
        if t in ("B-ASP", "B-OPN"):
            kind = "ASP" if t == "B-ASP" else "OPN"
            j = i
            itag = "I-%s" % kind
            while j + 1 < n and BIO_TAGS[tags[j + 1]] == itag:
                j += 1
            out[kind].append((i, j))
            i = j + 1
        else:
            i += 1
    return out


# ======================================================================
# 3. 数据加载 / Dataset
# ======================================================================
def load_samples():
    """读原始评论 + 解析后标签 -> list[(rid, text, quads)]。

    评论必须用原文(TRAIN_REVIEWS_PATH): 标签字符偏移按原文标注,
     processed 版压缩空格会导致 78 条评论 span 错位。
    """
    rev_df = pd.read_csv(TRAIN_REVIEWS_PATH, encoding="utf-8-sig")
    lab_df = pd.read_csv(TRAIN_LABELS_PARSED_PATH, encoding="utf-8-sig")
    rev_map = dict(zip(rev_df["id"].astype(int), rev_df["Reviews"].astype(str)))
    grouped = {}
    for _, row in lab_df.iterrows():
        rid = int(row["id"])
        grouped.setdefault(rid, []).append({
            "a_start": row.get("A_start"), "a_end": row.get("A_end"),
            "o_start": row.get("O_start"), "o_end": row.get("O_end"),
        })
    return [(rid, rev_map[rid], grouped.get(rid, [])) for rid in sorted(rev_map)]


class PairDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=MAX_LEN):
        self.samples, self.tokenizer, self.max_len = samples, tokenizer, max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rid, text, quads = self.samples[idx]
        t = build_targets(text, quads, self.tokenizer, self.max_len)
        t["id"] = rid
        return t


def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    attn = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.zeros(B, max_len, dtype=torch.long)
    meta = []
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"])
        attn[i, :L] = torch.tensor(b["attention_mask"])
        labels[i, :L] = torch.tensor(b["labels"])
        meta.append({
            "id": b["id"],
            "asp_spans": b["asp_spans"],
            "opn_spans": b["opn_spans"],
            "gold_ij": b["gold_ij"],
        })
    return {"input_ids": input_ids, "attention_mask": attn,
            "labels": labels, "meta": meta}


# ======================================================================
# 4. 模型
# ======================================================================
class PairScorer(nn.Module):
    """双仿射打分: s(a,b) = a^T U b + Wa a + Wb b + bias。"""

    def __init__(self, dim):
        super().__init__()
        self.U = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.U)
        self.Wa = nn.Linear(dim, 1)
        self.Wb = nn.Linear(dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, a, b):
        # a: (na, d), b: (nb, d) -> (na, nb)
        bil = torch.einsum("ai,ij,bj->ab", a, self.U, b)
        return bil + self.Wa(a) + self.Wb(b).transpose(0, 1) + self.bias


class PairExtractModel(nn.Module):
    def __init__(self, pretrained=PRETRAINED, num_tags=len(BIO_TAGS),
                 mlp_dim=MLP_DIM, dropout=DROPOUT):
        super().__init__()
        from torchcrf import CRF
        self.bert = AutoModel.from_pretrained(pretrained)
        H = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.tag_proj = nn.Linear(H, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        self.asp_mlp = nn.Sequential(
            nn.Linear(3 * H, mlp_dim), nn.GELU(), nn.Dropout(dropout))
        self.opn_mlp = nn.Sequential(
            nn.Linear(3 * H, mlp_dim), nn.GELU(), nn.Dropout(dropout))
        self.scorer = PairScorer(mlp_dim)
        # 隐式 A / 隐式 O 哨兵(已是 mlp_dim 空间, 直接作为候选格最后一行/列)
        self.imp_a = nn.Parameter(torch.zeros(mlp_dim))
        self.imp_o = nn.Parameter(torch.zeros(mlp_dim))

    def encode(self, input_ids, attention_mask):
        seq = self.dropout(self.bert(
            input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        return self.tag_proj(seq), seq

    def _span_feats(self, seq_row, spans, mlp):
        if not spans:
            return seq_row.new_zeros((0, self.imp_a.numel()))
        rows = []
        for s, e in spans:
            rows.append(torch.cat([
                seq_row[s], seq_row[e], seq_row[s:e + 1].mean(dim=0),
            ], dim=-1))
        return mlp(torch.stack(rows, dim=0))

    def pair_logits(self, seq_row, asp_spans, opn_spans):
        """候选格 (na+1) x (no+1) logits; 末行/列为隐式哨兵。"""
        fa = self._span_feats(seq_row, asp_spans, self.asp_mlp)
        fo = self._span_feats(seq_row, opn_spans, self.opn_mlp)
        fa = torch.cat([fa, self.imp_a.unsqueeze(0)], dim=0)
        fo = torch.cat([fo, self.imp_o.unsqueeze(0)], dim=0)
        return self.scorer(fa, fo)

    def pair_loss(self, seq, meta, pos_weight=PAIR_POS_WEIGHT):
        loss_sum = seq.new_zeros(())
        n_cells = 0
        for b, m in enumerate(meta):
            na, no = len(m["asp_spans"]), len(m["opn_spans"])
            logits = self.pair_logits(seq[b], m["asp_spans"], m["opn_spans"])
            tgt = torch.zeros_like(logits)
            for i, j in m["gold_ij"]:
                tgt[i, j] = 1.0
            w = torch.where(tgt > 0,
                            torch.full_like(tgt, pos_weight),
                            torch.ones_like(tgt))
            loss_sum = loss_sum + F.binary_cross_entropy_with_logits(
                logits, tgt, weight=w, reduction="sum")
            n_cells += (na + 1) * (no + 1)
        return loss_sum / max(n_cells, 1)


# ======================================================================
# 5. 推理: CRF 解码 -> span 文本 -> 候选格概率 -> 阈值取对
# ======================================================================
@torch.no_grad()
def decode_review(model, tokenizer, text, device, max_len=MAX_LEN):
    """返回 dict: asp/opn span 文本列表, 候选格概率矩阵。"""
    model.eval()
    enc = tokenizer(text, max_length=max_len, truncation=True,
                    return_offsets_mapping=True, add_special_tokens=True,
                    return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].tolist()

    emissions, seq = model.encode(input_ids, attn)
    tags = model.crf.decode(emissions, mask=attn.bool())[0]
    spans = extract_spans_from_tags(tags)
    asp, opn = spans["ASP"], spans["OPN"]

    logits = model.pair_logits(seq[0], asp, opn)
    probs = torch.sigmoid(logits).detach().cpu().tolist()

    def _text(tok):
        s, e = tok
        return text[offsets[s][0]:offsets[e][1]]

    return {"asp_texts": [_text(t) for t in asp],
            "opn_texts": [_text(t) for t in opn],
            "asp_spans": asp, "opn_spans": opn,
            "probs": probs}


def pairs_from_grid(asp_texts, opn_texts, probs, tau):
    """按阈值从候选格取对。返回 (pairs, mode):
      pairs: list[(a_text|'_', o_text|'_')]
      mode : 'threshold' 命中阈值 / 'top1' 无命中取最高分 / 'safety' 无 span
    """
    na, no = len(asp_texts), len(opn_texts)
    imp_i, imp_j = na, no

    def _pair(i, j):
        return (asp_texts[i] if i < na else "_",
                opn_texts[j] if j < no else "_")

    if na == 0 and no == 0:
        return [("_", "_")], "safety"

    hit, best = [], None
    for i in range(na + 1):
        for j in range(no + 1):
            if i == imp_i and j == imp_j:
                continue  # 屏蔽双隐式(训练集中 0 例)
            p = probs[i][j]
            if p >= tau:
                hit.append((i, j, p))
            if best is None or p > best[2]:
                best = (i, j, p)

    if hit:
        hit.sort(key=lambda x: -x[2])
        pairs, seen = [], set()
        for i, j, _ in hit:
            pair = _pair(i, j)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
        return pairs, "threshold"

    return [_pair(best[0], best[1])], "top1"


# ======================================================================
# 6. dev 评估 (一次前向, 多阈值扫描)
# ======================================================================
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


@torch.no_grad()
def evaluate(model, tokenizer, samples, device, max_len=MAX_LEN,
             thresholds=None):
    """返回 {'threshold': {tau: metrics}, 'span': {...}, 'detail': [...]}。
    detail 每评论存 gold/pred(各阈值)/mode, 供误差分析。
    """
    if thresholds is None:
        thresholds = THRESHOLDS
    model.eval()
    agg = {t: dict(tp=0, fp=0, fn=0, pred=0,
                   top1=0, safety=0) for t in thresholds}
    span_cnt = {"ASP": dict(tp=0, fp=0, fn=0),
                "OPN": dict(tp=0, fp=0, fn=0)}
    total_gold = 0
    detail = []

    for rid, text, quads in samples:
        t = build_targets(text, quads, tokenizer, max_len)
        gold_pairs = t["gold_pairs"]
        gold_disp = {(a or "_", o or "_") for a, o in gold_pairs}
        total_gold += len(gold_disp)

        dec = decode_review(model, tokenizer, text, device, max_len)

        # span 级 exact match
        for kind, gold_spans, pred_spans in (
            ("ASP", t["asp_spans"], dec["asp_spans"]),
            ("OPN", t["opn_spans"], dec["opn_spans"]),
        ):
            gs, ps = set(gold_spans), set(pred_spans)
            c = span_cnt[kind]
            c["tp"] += len(gs & ps)
            c["fp"] += len(ps - gs)
            c["fn"] += len(gs - ps)

        row = {"id": rid, "gold": gold_disp}
        for tau in thresholds:
            pairs, mode = pairs_from_grid(
                dec["asp_texts"], dec["opn_texts"], dec["probs"], tau)
            pred_set = set(pairs)
            c = agg[tau]
            c["tp"] += len(pred_set & gold_disp)
            c["fp"] += len(pred_set - gold_disp)
            c["fn"] += len(gold_disp - pred_set)
            c["pred"] += len(pred_set)
            c["top1"] += int(mode == "top1")
            c["safety"] += int(mode == "safety")
            row[tau] = {"pred": pred_set, "mode": mode}
        detail.append(row)

    result = {"threshold": {}, "span": {}, "total_gold": total_gold,
              "n_review": len(samples)}
    for tau, c in agg.items():
        p, r, f1 = prf(c["tp"], c["fp"], c["fn"])
        result["threshold"][tau] = {
            "P": p, "R": r, "F1": f1,
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
            "pred/gold": round(c["pred"] / max(total_gold, 1), 3),
            "top1兜底率": round(c["top1"] / max(len(samples), 1), 3),
            "安全行率": round(c["safety"] / max(len(samples), 1), 3),
        }
    for kind, c in span_cnt.items():
        p, r, f1 = prf(c["tp"], c["fp"], c["fn"])
        result["span"][kind] = {"P": p, "R": r, "F1": f1}
    result["detail"] = detail
    return result


def pick_best_threshold(ev):
    """F1 最大, 并列取更高阈值(偏精度, 抑制过度抽取)。"""
    best_tau, best_m = None, None
    for tau, m in ev["threshold"].items():
        if best_m is None or (m["F1"], tau) > (best_m["F1"], best_tau):
            best_tau, best_m = tau, m
    return best_tau, best_m


# ======================================================================
# 7. 训练
# ======================================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_artifacts(model, best_tau, best_metrics, span_metrics):
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), PIPELINE_PAIR_WEIGHTS)
    meta = {
        "pretrained": PRETRAINED,
        "bio_tags": BIO_TAGS,
        "mlp_dim": MLP_DIM,
        "best_threshold": best_tau,
        "thresholds": THRESHOLDS,
        "best_dev": best_metrics,
        "span_dev": span_metrics,
    }
    PIPELINE_PAIR_META.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def save_snapshot(*, elapsed, n_total, n_train, n_dev, best_tau,
                  best_metrics, span_metrics, history):
    import yaml

    def _rel(p):
        try:
            return str(Path(p).relative_to(BASE_DIR)).replace("\\", "/")
        except ValueError:
            return str(p)

    snap = {
        "_doc": "方案C Stage1 对抽取训练快照(自动生成, 勿手工编辑)",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pretrained": PRETRAINED,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hyperparams": {
            "seed": SEED, "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "lr_bert": LR_BERT, "lr_head": LR_HEAD,
            "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
            "grad_clip": GRAD_CLIP, "dropout": DROPOUT, "dev_ratio": DEV_RATIO,
        },
        "pair": {
            "mlp_dim": MLP_DIM, "loss_weight": PAIR_LOSS_WEIGHT,
            "pos_weight": PAIR_POS_WEIGHT, "thresholds": THRESHOLDS,
        },
        "training": {
            "duration_min": round(elapsed / 60, 2),
            "total_samples": n_total, "train_samples": n_train,
            "dev_samples": n_dev,
            "best_threshold": best_tau,
            "best_dev_pair": best_metrics,
            "span_dev": span_metrics,
            "history": history,
        },
        "artifacts": {
            "weights": _rel(PIPELINE_PAIR_WEIGHTS),
            "meta": _rel(PIPELINE_PAIR_META),
            "train_log": _rel(PIPELINE_PAIR_LOG_PATH),
        },
    }
    with open(PIPELINE_TRAINED_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        yaml.dump(snap, f, allow_unicode=True, sort_keys=False,
                  default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description="方案C Stage1 (A,O) 对抽取训练")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None,
                        help="冒烟用: 截断训练样本数")
    parser.add_argument("--max-dev", type=int, default=None,
                        help="冒烟用: 截断 dev 样本数")
    args = parser.parse_args()

    global EPOCHS, BATCH_SIZE, MAX_LEN
    EPOCHS = args.epochs or EPOCHS
    BATCH_SIZE = args.batch_size or BATCH_SIZE
    MAX_LEN = args.max_len or MAX_LEN

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[环境] device=%s pretrained=%s" % (device, PRETRAINED))
    print("[超参] epochs=%d batch=%d max_len=%d lr_bert=%.g lr_head=%.g"
          % (EPOCHS, BATCH_SIZE, MAX_LEN, LR_BERT, LR_HEAD))
    print("[配对] mlp_dim=%d loss_weight=%.1f pos_weight=%.1f thresholds=%s"
          % (MLP_DIM, PAIR_LOSS_WEIGHT, PAIR_POS_WEIGHT, THRESHOLDS))

    # ---- 数据 ----
    samples = load_samples()
    rng = random.Random(SEED)
    rng.shuffle(samples)
    n_dev = max(1, int(len(samples) * DEV_RATIO))
    dev, train = samples[:n_dev], samples[n_dev:]
    if args.max_train:
        train = train[:args.max_train]
    if args.max_dev:
        dev = dev[:args.max_dev]
    print("[数据] 全量=%d train=%d dev=%d" % (len(samples), len(train), len(dev)))

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
    # 统计截断丢弃
    n_skip = sum(build_targets(text, quads, tokenizer, MAX_LEN)["n_skip"]
                 for _, text, quads in samples)
    if n_skip:
        print("[数据] 警告: %d 个四元组因 max_len=%d 截断未对齐, 已跳过"
              % (n_skip, MAX_LEN))

    train_dl = DataLoader(PairDataset(train, tokenizer, MAX_LEN),
                          batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn)

    # ---- 模型 / 优化器 ----
    model = PairExtractModel().to(device)
    head_params = (list(model.tag_proj.parameters())
                   + list(model.crf.parameters())
                   + list(model.asp_mlp.parameters())
                   + list(model.opn_mlp.parameters())
                   + list(model.scorer.parameters())
                   + [model.imp_a, model.imp_o])
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": LR_BERT},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    total_steps = max(len(train_dl) * EPOCHS, 1)
    warmup = int(total_steps * WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        return max(0.0, (total_steps - step) / max(total_steps - warmup, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- 训练 ----
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 72)
    log("方案C Stage1 (Aspect,Opinion) 对抽取  pretrained=%s device=%s"
        % (PRETRAINED, device))
    log("=" * 72)

    best_f1, best_tau, best_metrics = -1.0, None, None
    best_span = None
    history = []
    t0 = time.time()
    n_batch_total = len(train_dl)
    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss = ep_crf = ep_pair = 0.0
        n_b = 0
        ep_t0 = time.time()
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            emissions, seq = model.encode(input_ids, attn)
            crf_loss = -model.crf(emissions, labels, mask=attn.bool()).mean()
            pair_loss = model.pair_loss(seq, batch["meta"])
            loss = crf_loss + PAIR_LOSS_WEIGHT * pair_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()
            ep_crf += crf_loss.item()
            ep_pair += pair_loss.item()
            n_b += 1
            if n_b % 20 == 0 or n_b == n_batch_total:
                el = time.time() - ep_t0
                remain = el / n_b * (n_batch_total - n_b)
                print("  [Ep %d] %d/%d loss=%.4f %.0fs/剩%.0fs"
                      % (ep, n_b, n_batch_total, ep_loss / n_b, el, remain),
                      flush=True)

        # ---- dev 评估 ----
        ev = evaluate(model, tokenizer, dev, device, MAX_LEN, THRESHOLDS)
        bt, bm = pick_best_threshold(ev)
        m05 = ev["threshold"][0.5] if 0.5 in ev["threshold"] else bm
        log("[Epoch %d/%d] loss=%.4f (crf=%.4f pair=%.4f)"
            % (ep, EPOCHS, ep_loss / n_b, ep_crf / n_b, ep_pair / n_b))
        log("  span  ASP F1=%.4f  OPN F1=%.4f"
            % (ev["span"]["ASP"]["F1"], ev["span"]["OPN"]["F1"]))
        log("  pair@0.5 P=%.4f R=%.4f F1=%.4f pred/gold=%.2f"
            % (m05["P"], m05["R"], m05["F1"], m05["pred/gold"]))
        log("  pair@%.1f(最优) P=%.4f R=%.4f F1=%.4f pred/gold=%.2f "
            "top1兜底=%.3f 安全行=%.3f"
            % (bt, bm["P"], bm["R"], bm["F1"], bm["pred/gold"],
               bm["top1兜底率"], bm["安全行率"]))
        history.append({
            "epoch": ep,
            "loss": round(ep_loss / n_b, 4),
            "crf": round(ep_crf / n_b, 4),
            "pair": round(ep_pair / n_b, 4),
            "span_asp_f1": round(ev["span"]["ASP"]["F1"], 4),
            "span_opn_f1": round(ev["span"]["OPN"]["F1"], 4),
            "pair_f1_0.5": round(m05["F1"], 4),
            "best_tau": bt, "best_f1": round(bm["F1"], 4),
        })
        if bm["F1"] > best_f1:
            best_f1, best_tau, best_metrics = bm["F1"], bt, bm
            best_span = ev["span"]
            save_artifacts(model, best_tau, best_metrics, best_span)
            log("  [*] best F1=%.4f (tau=%.1f), 已保存权重" % (best_f1, bt))

    elapsed = time.time() - t0
    log("[完成] 耗时 %.1f min, best dev pair F1=%.4f @tau=%.1f"
        % (elapsed / 60, best_f1, best_tau))
    log("[输出] 权重: %s" % PIPELINE_PAIR_WEIGHTS)
    log("[输出] 元数据: %s" % PIPELINE_PAIR_META)

    save_snapshot(elapsed=elapsed, n_total=len(samples),
                  n_train=len(train), n_dev=len(dev),
                  best_tau=best_tau, best_metrics=best_metrics,
                  span_metrics=best_span, history=history)
    log("[输出] 快照: %s" % PIPELINE_TRAINED_SNAPSHOT_PATH)

    PIPELINE_PAIR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PIPELINE_PAIR_LOG_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("[输出] 日志: %s" % PIPELINE_PAIR_LOG_PATH)


if __name__ == "__main__":
    main()
