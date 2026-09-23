# -*- coding: utf-8 -*-
"""
方案C - Stage2: 逐对 (Category, Polarity) 交叉编码分类 (管道式 ACOS 第二步)
========================================================================

任务
----
Stage1 已抽得每条评论的若干 (Aspect, Opinion) 对; Stage2 对每个对独立
分类 Category(13 类) 与 Polarity(3 类), 拼成四元组。

模型 (交叉编码器 / Cross-Encoder)
----------------------------------
  输入: [CLS] review [SEP] 方面词:a 观点词:o [SEP]
         (隐式 aspect/opinion 用 "无" 占位, BERT 以 token_type_ids 区分段)
  bert-base-chinese -> [CLS] -> dropout
        |-- cat_head (Linear)  -> 13 类 Category
        |-- pol_head (Linear)  -> 3 类 Polarity

  损失: L = w_cat * CE_cat(class_weight) + w_pol * CE_pol(class_weight)
        类权重 = 截断逆频 (抑制 整体/正面 主导, 保护 新鲜度/中性 等稀有类)
  选模型: (cat_macro_F1 + pol_macro_F1) / 2 最大者

数据
----
  评论用 data/raw/TRAIN/Train_reviews.csv 原文 (与 Stage1 同口径, 避免空格压缩);
  标签来自 outputs/processed/Train_labels_parsed.csv。
  切分与 Stage1 一致: review 级 shuffle(SEED=42) 后 85/15, 保证 dev 评论相同。

运行
----
  python src/pipeline/classify_pair.py                 # 完整训练 + dev 评估
  python src/pipeline/classify_pair.py --epochs 3      # 覆盖 epoch
  python src/pipeline/classify_pair.py --max-train 64 --max-dev 32  # 冒烟

输出
----
  data/pipeline/pair_classifier.pt    权重(state_dict)
  data/pipeline/pair_clf_meta.json     标签/类权重/dev 指标
  config/model_pipeline_clf_trained.yaml  训练快照
  outputs/logs/pipeline_clf_report.txt 训练日志
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# 避免联网超时: 模型已缓存到本地, 默认离线
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ----------------------------------------------------------------------
# 路径: 把 src/ 加入 sys.path 以便 import config
# ----------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR))
from config import (  # noqa: E402
    BASE_DIR, TRAIN_REVIEWS_PATH, TRAIN_LABELS_PARSED_PATH,
    PIPELINE_DIR, PIPELINE_CONFIG_PATH, PIPELINE_CLF_WEIGHTS,
    PIPELINE_CLF_META, PIPELINE_CLF_LOG_PATH,
)

# ======================================================================
# 0. 默认超参 (可被 config/model_pipeline.yaml 的 clf 段覆盖)
# ======================================================================
PRETRAINED = "bert-base-chinese"
CATEGORIES = ["整体", "使用体验", "功效", "价格", "物流", "气味", "包装",
               "真伪", "服务", "其他", "成分", "尺寸", "新鲜度"]
POLARITIES = ["正面", "负面", "中性"]

SEED = 42
MAX_LEN = 320
BATCH_SIZE = 12
EPOCHS = 5
LR_BERT = 2e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_CLIP = 1.0
DROPOUT = 0.1
DEV_RATIO = 0.15
EVAL_BATCH_SIZE = 32

W_CAT = 1.0
W_POL = 1.0
CLASS_WEIGHT_CAP = 10.0
IMPLICIT_MARKER = "无"

CAT2ID = {c: i for i, c in enumerate(CATEGORIES)}
POL2ID = {p: i for i, p in enumerate(POLARITIES)}


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

    lab = cfg.get("labels") or {}
    if lab.get("categories"):
        _set("CATEGORIES", list(lab["categories"]))
    if lab.get("polarities"):
        _set("POLARITIES", list(lab["polarities"]))
    g["CAT2ID"] = {c: i for i, c in enumerate(g["CATEGORIES"])}
    g["POL2ID"] = {p: i for i, p in enumerate(g["POLARITIES"])}

    hp = cfg.get("hyperparams") or {}
    for key, name, cast in (
        ("seed", "SEED", int), ("weight_decay", "WEIGHT_DECAY", float),
        ("warmup_ratio", "WARMUP_RATIO", float), ("grad_clip", "GRAD_CLIP", float),
        ("dev_ratio", "DEV_RATIO", float),
    ):
        if hp.get(key) is not None:
            _set(name, hp[key], cast)

    clf = cfg.get("clf") or {}
    for key, name, cast in (
        ("max_len", "MAX_LEN", int), ("batch_size", "BATCH_SIZE", int),
        ("epochs", "EPOCHS", int), ("lr_bert", "LR_BERT", float),
        ("lr_head", "LR_HEAD", float), ("dropout", "DROPOUT", float),
        ("loss_weight_cat", "W_CAT", float), ("loss_weight_pol", "W_POL", float),
        ("class_weight_cap", "CLASS_WEIGHT_CAP", float),
    ):
        if clf.get(key) is not None:
            _set(name, clf[key], cast)
    if clf.get("implicit_marker"):
        _set("IMPLICIT_MARKER", str(clf["implicit_marker"]))

    if verbose:
        print("[配置] 已加载: %s" % path)


load_config()


# ======================================================================
# 2. 数据加载 / 工具
# ======================================================================
def load_review_samples():
    """读原始评论 + 解析后标签 -> list[(rid, text, quads)]。

    评论必须用原文(TRAIN_REVIEWS_PATH): 与 Stage1 同口径。
    quads: list[dict], 字段 a_start/a_end/o_start/o_end(可能为 NaN)/cat/pol
    """
    rev_df = pd.read_csv(TRAIN_REVIEWS_PATH, encoding="utf-8-sig")
    lab_df = pd.read_csv(TRAIN_LABELS_PARSED_PATH, encoding="utf-8-sig")
    rev_map = dict(zip(rev_df["id"].astype(int), rev_df["Reviews"].astype(str)))
    grouped = {}
    n_bad = 0
    for _, row in lab_df.iterrows():
        rid = int(row["id"])
        cat = row["Categories"]
        pol = row["Polarities"]
        if cat not in CAT2ID or pol not in POL2ID:
            n_bad += 1
            continue
        grouped.setdefault(rid, []).append({
            "a_start": row.get("A_start"), "a_end": row.get("A_end"),
            "o_start": row.get("O_start"), "o_end": row.get("O_end"),
            "cat": cat, "pol": pol,
        })
    if n_bad:
        print("[数据] 警告: %d 条标签的 cat/pol 不在词表内, 已跳过" % n_bad)
    return [(rid, rev_map[rid], grouped.get(rid, [])) for rid in sorted(rev_map)]


def quad_to_example(rid, text, q):
    """一条 gold 四元组 -> (rid, text, a_text|None, o_text|None, cat, pol)。

    显式 span: 用字符偏移从原文切片(与 Stage1 decode 同口径)。
    隐式 span (NaN 偏移): a_text/o_text = None, 后续在 pair 文本里占位。
    """
    a_text = None
    if not pd.isna(q["a_start"]):
        a_text = text[int(q["a_start"]):int(q["a_end"])]
    o_text = None
    if not pd.isna(q["o_start"]):
        o_text = text[int(q["o_start"]):int(q["o_end"])]
    return (rid, text, a_text, o_text, q["cat"], q["pol"])


def build_pair_text(a_text, o_text, marker=IMPLICIT_MARKER):
    a = a_text if a_text else marker
    o = o_text if o_text else marker
    return "方面词：%s 观点词：%s" % (a, o)


class ClfDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=MAX_LEN):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rid, text, a_text, o_text, cat, pol = self.samples[idx]
        pair_text = build_pair_text(a_text, o_text)
        enc = self.tokenizer(text, text_pair=pair_text, max_length=self.max_len,
                             truncation=True, add_special_tokens=True)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_type_ids": enc.get("token_type_ids", [0] * len(enc["input_ids"])),
            "cat_id": CAT2ID[cat],
            "pol_id": POL2ID[pol],
            "id": rid,
        }


def collate_fn(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    attn = torch.zeros(B, max_len, dtype=torch.long)
    tok_type = torch.zeros(B, max_len, dtype=torch.long)
    cat_ids = torch.zeros(B, dtype=torch.long)
    pol_ids = torch.zeros(B, dtype=torch.long)
    ids = []
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"])
        attn[i, :L] = torch.tensor(b["attention_mask"])
        tok_type[i, :L] = torch.tensor(b["token_type_ids"])
        cat_ids[i] = b["cat_id"]
        pol_ids[i] = b["pol_id"]
        ids.append(b["id"])
    return {"input_ids": input_ids, "attention_mask": attn,
            "token_type_ids": tok_type, "cat_ids": cat_ids,
            "pol_ids": pol_ids, "ids": ids}


# ======================================================================
# 3. 模型
# ======================================================================
class PairClassifier(nn.Module):
    """交叉编码器: [CLS] review [SEP] pair [SEP] -> [CLS] -> 双头。"""

    def __init__(self, pretrained=PRETRAINED, num_cat=len(CATEGORIES),
                 num_pol=len(POLARITIES), dropout=DROPOUT):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained)
        H = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cat_head = nn.Linear(H, num_cat)
        self.pol_head = nn.Linear(H, num_pol)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return self.cat_head(cls), self.pol_head(cls)


# ======================================================================
# 4. 指标
# ======================================================================
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def per_class_metrics(y_true, y_pred, labels):
    """每类 P/R/F1 + macro 平均。返回 (macro_f1, [per_class dict])。"""
    out = []
    for c, name in enumerate(labels):
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        p, r, f1 = prf(tp, fp, fn)
        out.append({"label": name, "support": tp + fn,
                    "P": p, "R": r, "F1": f1})
    macro = sum(x["F1"] for x in out) / len(out) if out else 0.0
    return macro, out


def compute_class_weights(counts, cap):
    """截断逆频权重: total/(n*c), 上限 cap (保护稀有类不过度)。"""
    total = sum(counts)
    n = len(counts)
    weights = []
    for c in counts:
        w = total / (n * c) if c > 0 else cap
        weights.append(min(w, cap))
    return weights


# ======================================================================
# 5. dev 评估 (Stage2 隔离: 给定 gold 对, 测 cat/pol 分类)
# ======================================================================
@torch.no_grad()
def evaluate(model, tokenizer, samples, device, max_len=MAX_LEN,
            eval_batch=EVAL_BATCH_SIZE):
    """返回 dict: cat/pol 的 macro-F1 / acc / 每类明细, 以及四元组(给定 gold 对)准确率。"""
    model.eval()
    ds = ClfDataset(samples, tokenizer, max_len)
    dl = DataLoader(ds, batch_size=eval_batch, shuffle=False, collate_fn=collate_fn)
    cat_t, cat_p, pol_t, pol_p = [], [], [], []
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        tt = batch["token_type_ids"].to(device)
        clo, plo = model(input_ids, attn, tt)
        cat_p += clo.argmax(-1).cpu().tolist()
        pol_p += plo.argmax(-1).cpu().tolist()
        cat_t += batch["cat_ids"].tolist()
        pol_t += batch["pol_ids"].tolist()

    cat_macro, cat_detail = per_class_metrics(cat_t, cat_p, CATEGORIES)
    pol_macro, pol_detail = per_class_metrics(pol_t, pol_p, POLARITIES)
    cat_acc = sum(t == p for t, p in zip(cat_t, cat_p)) / max(len(cat_t), 1)
    pol_acc = sum(t == p for t, p in zip(pol_t, pol_p)) / max(len(pol_t), 1)
    quad_acc = sum(ct == cp and pt == pp
                   for ct, cp, pt, pp in zip(cat_t, cat_p, pol_t, pol_p)
                   ) / max(len(cat_t), 1)

    # 最易混淆的 category 对 (错误计数 top5)
    conf = Counter()
    for t, p in zip(cat_t, cat_p):
        if t != p:
            conf[(CATEGORIES[t], CATEGORIES[p])] += 1
    top_conf = [{"gold": k[0], "pred": k[1], "n": v} for k, v in conf.most_common(5)]

    return {
        "n": len(cat_t),
        "cat_macro_f1": cat_macro, "cat_acc": cat_acc,
        "pol_macro_f1": pol_macro, "pol_acc": pol_acc,
        "quad_acc": quad_acc,           # 给定 gold 对时四元组(cat&pol 均对)准确率
        "cat_detail": cat_detail,
        "pol_detail": pol_detail,
        "top_confused": top_conf,
    }


# ======================================================================
# 6. 训练
# ======================================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_artifacts(model, best_metrics, cat_weights, pol_weights):
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), PIPELINE_CLF_WEIGHTS)
    meta = {
        "pretrained": PRETRAINED,
        "categories": CATEGORIES,
        "polarities": POLARITIES,
        "implicit_marker": IMPLICIT_MARKER,
        "max_len": MAX_LEN,
        "class_weights_cat": cat_weights,
        "class_weights_pol": pol_weights,
        "best_dev": best_metrics,
    }
    PIPELINE_CLF_META.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def save_snapshot(*, elapsed, n_train, n_dev, cat_weights, pol_weights,
                  best_metrics, history):
    import yaml
    snap_path = BASE_DIR / "config" / "model_pipeline_clf_trained.yaml"

    snap = {
        "_doc": "方案C Stage2 对分类训练快照(自动生成, 勿手工编辑)",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pretrained": PRETRAINED,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hyperparams": {
            "seed": SEED, "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "lr_bert": LR_BERT, "lr_head": LR_HEAD,
            "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
            "grad_clip": GRAD_CLIP, "dropout": DROPOUT, "dev_ratio": DEV_RATIO,
        },
        "clf": {
            "loss_weight_cat": W_CAT, "loss_weight_pol": W_POL,
            "class_weight_cap": CLASS_WEIGHT_CAP,
            "implicit_marker": IMPLICIT_MARKER,
        },
        "training": {
            "duration_min": round(elapsed / 60, 2),
            "train_samples": n_train, "dev_samples": n_dev,
            "best_dev": best_metrics, "history": history,
        },
        "artifacts": {
            "weights": "data/pipeline/pair_classifier.pt",
            "meta": "data/pipeline/pair_clf_meta.json",
            "train_log": "outputs/logs/pipeline_clf_report.txt",
        },
    }
    with open(snap_path, "w", encoding="utf-8") as f:
        yaml.dump(snap, f, allow_unicode=True, sort_keys=False,
                  default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description="方案C Stage2 (Category,Polarity) 对分类训练")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--max-train", type=int, default=None,
                        help="冒烟用: 截断训练评论数")
    parser.add_argument("--max-dev", type=int, default=None,
                        help="冒烟用: 截断 dev 评论数")
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
    print("[分类] cat=%d类 pol=%d类 w_cat=%.1f w_pol=%.1f cap=%.1f marker=%s"
          % (len(CATEGORIES), len(POLARITIES), W_CAT, W_POL,
             CLASS_WEIGHT_CAP, IMPLICIT_MARKER))

    # ---- 数据: review 级切分 (与 Stage1 同 seed) ----
    reviews = load_review_samples()
    rng = random.Random(SEED)
    rng.shuffle(reviews)
    n_dev = max(1, int(len(reviews) * DEV_RATIO))
    dev_reviews, train_reviews = reviews[:n_dev], reviews[n_dev:]
    if args.max_train:
        train_reviews = train_reviews[:args.max_train]
    if args.max_dev:
        dev_reviews = dev_reviews[:args.max_dev]
    print("[数据] 评论 全量=%d train=%d dev=%d"
          % (len(reviews), len(train_reviews), len(dev_reviews)))

    # ---- 展开为四元组级样本 ----
    train_clf = [quad_to_example(rid, text, q)
                 for rid, text, quads in train_reviews for q in quads]
    dev_clf = [quad_to_example(rid, text, q)
               for rid, text, quads in dev_reviews for q in quads]
    print("[数据] 四元组 train=%d dev=%d (显式A %.1f%% 显式O %.1f%%)"
          % (len(train_clf), len(dev_clf),
             100 * sum(x[2] is not None for x in train_clf) / max(len(train_clf), 1),
             100 * sum(x[3] is not None for x in train_clf) / max(len(train_clf), 1)))

    # ---- 类权重 (来自训练集分布, 截断逆频) ----
    cat_counts = [0] * len(CATEGORIES)
    pol_counts = [0] * len(POLARITIES)
    for _, _, _, _, cat, pol in train_clf:
        cat_counts[CAT2ID[cat]] += 1
        pol_counts[POL2ID[pol]] += 1
    cat_w = compute_class_weights(cat_counts, CLASS_WEIGHT_CAP)
    pol_w = compute_class_weights(pol_counts, CLASS_WEIGHT_CAP)
    print("[类权重] cat 分布=%s 权重=%s" % (
        dict(zip(CATEGORIES, cat_counts)), [round(w, 2) for w in cat_w]))
    print("[类权重] pol 分布=%s 权重=%s" % (
        dict(zip(POLARITIES, pol_counts)), [round(w, 2) for w in pol_w]))

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
    train_dl = DataLoader(ClfDataset(train_clf, tokenizer, MAX_LEN),
                          batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn)

    # ---- 模型 / 优化器 ----
    model = PairClassifier().to(device)
    head_params = list(model.cat_head.parameters()) + list(model.pol_head.parameters())
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

    cat_w_t = torch.tensor(cat_w, dtype=torch.float, device=device)
    pol_w_t = torch.tensor(pol_w, dtype=torch.float, device=device)

    # ---- 训练 ----
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 72)
    log("方案C Stage2 (Category,Polarity) 交叉编码分类  pretrained=%s device=%s"
        % (PRETRAINED, device))
    log("=" * 72)

    best_score, best_metrics = -1.0, None
    history = []
    t0 = time.time()
    n_batch_total = len(train_dl)
    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss = ep_cat = ep_pol = 0.0
        n_b = 0
        ep_t0 = time.time()
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            tt = batch["token_type_ids"].to(device)
            cat_ids = batch["cat_ids"].to(device)
            pol_ids = batch["pol_ids"].to(device)
            optimizer.zero_grad()
            clo, plo = model(input_ids, attn, tt)
            cat_loss = F.cross_entropy(clo, cat_ids, weight=cat_w_t)
            pol_loss = F.cross_entropy(plo, pol_ids, weight=pol_w_t)
            loss = W_CAT * cat_loss + W_POL * pol_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()
            ep_cat += cat_loss.item()
            ep_pol += pol_loss.item()
            n_b += 1
            if n_b % 20 == 0 or n_b == n_batch_total:
                el = time.time() - ep_t0
                remain = el / n_b * (n_batch_total - n_b)
                print("  [Ep %d] %d/%d loss=%.4f %.0fs/剩%.0fs"
                      % (ep, n_b, n_batch_total, ep_loss / n_b, el, remain),
                      flush=True)

        # ---- dev 评估 ----
        ev = evaluate(model, tokenizer, dev_clf, device, MAX_LEN)
        score = (ev["cat_macro_f1"] + ev["pol_macro_f1"]) / 2
        log("[Epoch %d/%d] loss=%.4f (cat=%.4f pol=%.4f)"
            % (ep, EPOCHS, ep_loss / n_b, ep_cat / n_b, ep_pol / n_b))
        log("  cat  macro-F1=%.4f acc=%.4f   pol macro-F1=%.4f acc=%.4f   "
            "quad(给定gold对)acc=%.4f"
            % (ev["cat_macro_f1"], ev["cat_acc"], ev["pol_macro_f1"],
               ev["pol_acc"], ev["quad_acc"]))
        history.append({
            "epoch": ep,
            "loss": round(ep_loss / n_b, 4),
            "cat": round(ep_cat / n_b, 4),
            "pol": round(ep_pol / n_b, 4),
            "cat_macro_f1": round(ev["cat_macro_f1"], 4),
            "pol_macro_f1": round(ev["pol_macro_f1"], 4),
            "quad_acc": round(ev["quad_acc"], 4),
        })
        if score > best_score:
            best_score = score
            best_metrics = ev
            save_artifacts(model, best_metrics, cat_w, pol_w)
            log("  [*] best score=%.4f (cat_mF1=%.4f pol_mF1=%.4f), 已保存权重"
                % (best_score, ev["cat_macro_f1"], ev["pol_macro_f1"]))

    elapsed = time.time() - t0
    log("[完成] 耗时 %.1f min, best score=%.4f" % (elapsed / 60, best_score))

    # ---- 误差分析: 输出每类明细 + 最易混淆对 ----
    log("")
    log("[Category 每类明细]")
    for d in best_metrics["cat_detail"]:
        log("  %-6s support=%-4d P=%.3f R=%.3f F1=%.3f"
            % (d["label"], d["support"], d["P"], d["R"], d["F1"]))
    log("[Polarity 每类明细]")
    for d in best_metrics["pol_detail"]:
        log("  %-6s support=%-4d P=%.3f R=%.3f F1=%.3f"
            % (d["label"], d["support"], d["P"], d["R"], d["F1"]))
    log("[最易混淆 Category 对 (top5)]")
    for c in best_metrics["top_confused"]:
        log("  gold=%s -> pred=%s  n=%d" % (c["gold"], c["pred"], c["n"]))

    save_snapshot(elapsed=elapsed, n_train=len(train_clf), n_dev=len(dev_clf),
                  cat_weights=cat_w, pol_weights=pol_w,
                  best_metrics=best_metrics, history=history)

    log("[输出] 权重: %s" % PIPELINE_CLF_WEIGHTS)
    log("[输出] 元数据: %s" % PIPELINE_CLF_META)
    log("[输出] 快照: config/model_pipeline_clf_trained.yaml")

    PIPELINE_CLF_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PIPELINE_CLF_LOG_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("[输出] 日志: %s" % PIPELINE_CLF_LOG_PATH)


if __name__ == "__main__":
    main()
