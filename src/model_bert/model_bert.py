# -*- coding: utf-8 -*-
"""
BERT + CRF 多任务模型 (化妆品评论 ACOS 四元组抽取)
====================================================================

任务
----
对每条评论抽取 ACOS 四元组 (Aspect, Category, Opinion, Sentiment):
  - Aspect / Opinion: span 抽取 (BIO 序列标注 + CRF 解码)
  - Category: 13 类分类 (span-pooling -> MLP)
  - Sentiment: 3 类分类 (span-pooling -> MLP)

模型架构
--------
  bert-base-chinese
      |
      |-- (a) tag_proj (Linear) -> CRF (5 标签 BIO, 联合抽 A/O span)
      |-- (b) cat_head (MLP)   -> 13 类 Category
      |-- (c) pol_head (MLP)   -> 3 类 Polarity
  分类特征 = [CLS] ⊕ span_pooling(aspect 优先, 否则 opinion, 否则 [CLS] 自拼)

输入数据
--------
  - 数据复用 preprocess.py 产物 (outputs/processed/Train_*_processed.csv)
  - 标签的 A_start/A_end/O_start/O_end 是对原始 Reviews 文本的字符偏移,
    故 BERT 输入也用 Reviews 列(含标点), 不用 clean_text(已去标点)
  - 隐式四元组 (术语为 '_'): 跳过 BIO 标注, 仍参与 Category/Polarity 训练

运行方式
--------
  python src/model_bert/model_bert.py             # 训练 + dev 评估
  python src/model_bert/model_bert.py --skip-test  # 同上, 不影响
  python src/model_bert/model_bert.py --epochs 3  # 临时改 epoch 数

输出
----
  data/bert/bert_crf.pt         模型权重 (state_dict)
  data/bert/label_vocab.json    标签字典 (BIO/Category/Polarity)
  outputs/logs/bert_report.txt  训练日志
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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# ----------------------------------------------------------------------
# 路径配置: 把 src/ 加入 sys.path 以便 import config
# ----------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC_DIR))
from config import (  # noqa: E402
    BASE_DIR, TRAIN_REVIEWS_PROC_PATH, TRAIN_LABELS_PARSED_PATH,
    TEST_REVIEWS_PATH, BERT_DIR, BERT_LOG_PATH, BERT_CONFIG_PATH,
    BERT_TRAINED_SNAPSHOT_PATH,
)

# ======================================================================
# 0. 默认超参 (可被 config/model_bert.yaml 覆盖)
# ======================================================================
PRETRAINED = "bert-base-chinese"
BIO_TAGS = ["O", "B-ASP", "I-ASP", "B-OPN", "I-OPN"]   # id=0 即 O(padding 同标签)
CATEGORIES = ["包装", "成分", "尺寸", "服务", "功效", "价格", "气味",
              "使用体验", "物流", "新鲜度", "真伪", "整体", "其他"]
POLARITIES = ["正面", "中性", "负面"]

SEED = 42
MAX_LEN = 256
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
EPOCHS = 8
LR_BERT = 2e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_CLIP = 1.0
DROPOUT = 0.1
DEV_RATIO = 0.15
AMP = False
W_CRF, W_CAT, W_POL = 1.0, 0.5, 0.5

# 标签字典 (启动时建立)
CAT2ID = {c: i for i, c in enumerate(CATEGORIES)}
POL2ID = {p: i for i, p in enumerate(POLARITIES)}
ID2CAT = {i: c for c, i in CAT2ID.items()}
ID2POL = {i: p for p, i in POL2ID.items()}


# ======================================================================
# 1. YAML 配置加载 (仿 model_baseline.py 的 load_model_config)
# ======================================================================
def load_model_config(path=BERT_CONFIG_PATH, verbose=True):
    """加载 yaml 超参, 覆盖模块级默认值。返回状态字符串。"""
    path = Path(path)
    if not path.exists():
        return "missing"
    try:
        import yaml
    except ImportError:
        if verbose:
            print("[配置] 未安装 PyYAML, 使用代码内默认配置")
        return "no_pyyaml"
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        if verbose:
            print("[配置] YAML 语法错误: %s" % e)
        return "invalid"
    g = globals()

    def _set(name, value, cast=None):
        try:
            if cast is not None:
                value = cast(value)
            g[name] = value
        except (TypeError, ValueError) as e:
            if verbose:
                print("[配置] 字段 %s=%r 无效(%s), 使用默认值" % (name, value, e))

    if cfg.get("pretrained"):
        _set("PRETRAINED", cfg["pretrained"], cast=str)

    lab = cfg.get("labels") or {}
    # 标签顺序不可随意变更, 但允许用户在 yaml 中显式重写(以 yaml 列表为准)
    if lab.get("bio_tags"):
        _set("BIO_TAGS", list(lab["bio_tags"]))
    if lab.get("categories"):
        _set("CATEGORIES", list(lab["categories"]))
    if lab.get("polarities"):
        _set("POLARITIES", list(lab["polarities"]))

    hp = cfg.get("hyperparams") or {}
    for key, name, cast in (
        ("seed", "SEED", int), ("max_len", "MAX_LEN", int),
        ("batch_size", "BATCH_SIZE", int), ("eval_batch_size", "EVAL_BATCH_SIZE", int),
        ("epochs", "EPOCHS", int), ("lr_bert", "LR_BERT", float),
        ("lr_head", "LR_HEAD", float), ("weight_decay", "WEIGHT_DECAY", float),
        ("warmup_ratio", "WARMUP_RATIO", float), ("grad_clip", "GRAD_CLIP", float),
        ("dropout", "DROPOUT", float), ("dev_ratio", "DEV_RATIO", float),
    ):
        if hp.get(key) is not None:
            _set(name, hp[key], cast=cast)
    if hp.get("amp") is not None:
        _set("AMP", bool(hp["amp"]))

    lw = cfg.get("loss_weights") or {}
    for key, name in (("w_crf", "W_CRF"), ("w_cat", "W_CAT"),
                      ("w_pol", "W_POL")):
        if lw.get(key) is not None:
            _set(name, lw[key], cast=float)

    # 重建反向映射
    g["CAT2ID"] = {c: i for i, c in enumerate(g["CATEGORIES"])}
    g["POL2ID"] = {p: i for i, p in enumerate(g["POLARITIES"])}
    g["ID2CAT"] = {i: c for c, i in g["CAT2ID"].items()}
    g["ID2POL"] = {i: p for p, i in g["POL2ID"].items()}

    if verbose:
        print("[配置] 已加载: %s" % path)
    return "ok"


load_model_config()


# ======================================================================
# 2. 工具: 字符偏移 -> token 偏移, BIO 标签构建, span 提取
# ======================================================================
def char_span_to_token_span(offsets, c_start, c_end):
    """offsets: tokenizer 返回的 (char_start, char_end) 列表, 含 [CLS]/[SEP] 为 (0,0)。
    返回 (tok_start, tok_end) 闭区间; token 与 [c_start, c_end) 相交即纳入。
    找不到返回 None。
    """
    s_tok = e_tok = None
    for i, (cs, ce) in enumerate(offsets):
        if cs == 0 and ce == 0:
            continue   # 特殊 token
        if cs < c_end and ce > c_start:   # 相交
            if s_tok is None:
                s_tok = i
            e_tok = i
    if s_tok is None:
        return None
    return (s_tok, e_tok)


def build_bio_labels(text, quads, tokenizer, max_len=MAX_LEN):
    """对单条 review 构建 token 级 BIO 标签与 span/分类目标。
    quads: list[dict], 字段: a_start,a_end,o_start,o_end,cat,pol (位置可为 None/pd.NA)
    返回 dict: input_ids, attention_mask, labels(list[int]),
              cat_targets(list[(feat_span|None, cat_id, pol_id)])
    """
    enc = tokenizer(text, max_length=max_len, truncation=True,
                    return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    L = len(input_ids)
    labels = [BIO_TAGS.index("O")] * L   # 0 = O, padding 也用 0(CRF 由 mask 屏蔽)

    cat_targets = []
    for q in quads:
        a_span = None
        o_span = None
        # aspect span
        if q.get("a_start") is not None and not pd.isna(q["a_start"]):
            a_span = char_span_to_token_span(offsets, int(q["a_start"]), int(q["a_end"]))
            if a_span:
                s, e = a_span
                labels[s] = BIO_TAGS.index("B-ASP")
                for k in range(s + 1, e + 1):
                    labels[k] = BIO_TAGS.index("I-ASP")
        # opinion span
        if q.get("o_start") is not None and not pd.isna(q["o_start"]):
            o_span = char_span_to_token_span(offsets, int(q["o_start"]), int(q["o_end"]))
            if o_span:
                s, e = o_span
                labels[s] = BIO_TAGS.index("B-OPN")
                for k in range(s + 1, e + 1):
                    labels[k] = BIO_TAGS.index("I-OPN")
        # 分类特征 span: aspect 优先, 否则 opinion, 否则 None(用 [CLS])
        feat_span = a_span or o_span
        cat = q.get("cat")
        pol = q.get("pol")
        cat_id = CAT2ID.get(cat, 0)
        pol_id = POL2ID.get(pol, 0)
        cat_targets.append((feat_span, cat_id, pol_id))
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "cat_targets": cat_targets,
    }


def extract_spans_from_tags(tags):
    """tags: list[int] BIO id 序列。返回 {'A':[(s,e)], 'O':[(s,e)]} 闭区间。"""
    spans = {"A": [], "O": []}
    i, n = 0, len(tags)
    while i < n:
        t = BIO_TAGS[tags[i]] if tags[i] < len(BIO_TAGS) else "O"
        if t == "B-ASP":
            j = i
            while j + 1 < n and BIO_TAGS[tags[j + 1]] == "I-ASP":
                j += 1
            spans["A"].append((i, j))
            i = j + 1
        elif t == "B-OPN":
            j = i
            while j + 1 < n and BIO_TAGS[tags[j + 1]] == "I-OPN":
                j += 1
            spans["O"].append((i, j))
            i = j + 1
        else:
            i += 1
    return spans


def tokens_to_text(tokenizer, input_ids, s, e):
    """把 token id 区间 [s, e] (闭) 解码回文本, 去除 ## 前缀和空格。"""
    piece_ids = input_ids[s:e + 1]
    tokens = tokenizer.convert_ids_to_tokens(piece_ids)
    text = tokenizer.convert_tokens_to_string(tokens)
    return text.replace(" ", "").lstrip("#")


# ======================================================================
# 3. 数据加载与 Dataset
# ======================================================================
def load_train_samples():
    """读 Train_reviews_processed.csv + Train_labels_parsed.csv,
    按评论 id 聚合四元组, 返回 list[(rid, text, quads)]。"""
    rev_df = pd.read_csv(TRAIN_REVIEWS_PROC_PATH, encoding="utf-8-sig")
    lab_df = pd.read_csv(TRAIN_LABELS_PARSED_PATH, encoding="utf-8-sig")
    # id -> text
    rev_map = dict(zip(rev_df["id"].astype(int), rev_df["Reviews"].astype(str)))
    # 按 id 聚合标注
    grouped = {}
    for _, row in lab_df.iterrows():
        rid = int(row["id"])
        grouped.setdefault(rid, []).append({
            "a_start": row.get("A_start"),
            "a_end":   row.get("A_end"),
            "o_start": row.get("O_start"),
            "o_end":   row.get("O_end"),
            "cat":     str(row.get("Categories")).strip(),
            "pol":     str(row.get("Polarities")).strip(),
        })
    samples = []
    for rid in sorted(rev_map):
        text = rev_map[rid]
        samples.append((rid, text, grouped.get(rid, [])))
    return samples


class ReviewDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=MAX_LEN):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rid, text, quads = self.samples[idx]
        item = build_bio_labels(text, quads, self.tokenizer, self.max_len)
        return {
            "id": rid,
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "cat_targets": item["cat_targets"],
        }


def collate_fn(batch):
    """动态 padding 到 batch 内最长; padding 标签 0(O), CRF 用 mask 屏蔽。"""
    max_len = max(len(b["input_ids"]) for b in batch)
    B = len(batch)
    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    attn = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.zeros(B, max_len, dtype=torch.long)  # O=0
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = b["input_ids"]
        attn[i, :L] = b["attention_mask"]
        labels[i, :L] = b["labels"]
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "labels": labels,
        "cat_targets": [b["cat_targets"] for b in batch],
        "ids": [b["id"] for b in batch],
    }


# ======================================================================
# 4. 模型
# ======================================================================
class BertCrfModel(nn.Module):
    """bert-base-chinese + CRF + Category/Polarity 双分类头。"""

    def __init__(self, pretrained=PRETRAINED, num_tags=len(BIO_TAGS),
                 num_cats=len(CATEGORIES), num_pols=len(POLARITIES),
                 dropout=DROPOUT):
        super().__init__()
        from torchcrf import CRF
        self.bert = AutoModel.from_pretrained(pretrained)
        h = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.tag_proj = nn.Linear(h, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        # 分类特征 = [CLS] ⊕ span_mean, 长度 2h
        self.cat_head = nn.Sequential(
            nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, num_cats),
        )
        self.pol_head = nn.Sequential(
            nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, num_pols),
        )

    def encode(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(out.last_hidden_state)   # (B, L, H)
        cls = seq[:, 0]                              # (B, H)
        emissions = self.tag_proj(seq)              # (B, L, num_tags)
        return emissions, seq, cls

    def loss(self, emissions, labels, mask, seq, cls, cat_targets,
             w_crf=W_CRF, w_cat=W_CAT, w_pol=W_POL):
        # CRF NLL
        crf_loss = -self.crf(emissions, labels, mask=mask).mean()
        # Category/Polarity: 每个 quad 用其特征 span 做分类
        cat_loss = torch.zeros((), device=cls.device)
        pol_loss = torch.zeros((), device=cls.device)
        n = 0
        for b, targets in enumerate(cat_targets):
            for span, cat_id, pol_id in targets:
                if span is None:
                    feat = torch.cat([cls[b], cls[b]], dim=-1)
                else:
                    s, e = span
                    span_emb = seq[b, s:e + 1].mean(dim=0)
                    feat = torch.cat([cls[b], span_emb], dim=-1)
                cat_logit = self.cat_head(feat)      # (num_cats,)
                pol_logit = self.pol_head(feat)
                cat_loss = cat_loss + F.cross_entropy(
                    cat_logit.unsqueeze(0),
                    torch.tensor([cat_id], device=cls.device))
                pol_loss = pol_loss + F.cross_entropy(
                    pol_logit.unsqueeze(0),
                    torch.tensor([pol_id], device=cls.device))
                n += 1
        if n > 0:
            cat_loss = cat_loss / n
            pol_loss = pol_loss / n
        total = w_crf * crf_loss + w_cat * cat_loss + w_pol * pol_loss
        metrics = {"crf": crf_loss.item(),
                   "cat": cat_loss.item(), "pol": pol_loss.item()}
        return total, metrics

    @torch.no_grad()
    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)


# ======================================================================
# 5. 训练 / 评估
# ======================================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict_review_quads(model, tokenizer, text, device, max_len=MAX_LEN):
    """对单条 review 推理, 返回 set[(a_text|None, cat, o_text|None, pol)]。
    None 表示该字段为隐式('_'), 写 Result.csv 时转成下划线。
    """
    model.eval()
    enc = tokenizer(text, max_length=max_len, truncation=True,
                    return_offsets_mapping=True, add_special_tokens=True,
                    return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"][0].tolist()
    id_list = input_ids[0].tolist()

    emissions, seq, cls = model.encode(input_ids, attn)
    tags = model.crf.decode(emissions, mask=attn.bool())[0]
    spans = extract_spans_from_tags(tags)
    a_spans, o_spans = spans["A"], spans["O"]

    quads = set()

    def _feat(span_b):
        s, e = span_b
        span_emb = seq[0, s:e + 1].mean(dim=0)
        return torch.cat([cls[0], span_emb], dim=-1)

    def _predict_cat_pol(feat):
        cat = ID2CAT[model.cat_head(feat).argmax().item()]
        pol = ID2POL[model.pol_head(feat).argmax().item()]
        return cat, pol

    # 完全无 span: 用 [CLS] 自拼预测一个隐式四元组
    if not a_spans and not o_spans:
        feat = torch.cat([cls[0], cls[0]], dim=-1)
        cat, pol = _predict_cat_pol(feat)
        quads.add((None, cat, None, pol))
        return quads

    # 对每个 aspect span: 预测 cat; 找最近 opinion span 预测 pol
    for a_s, a_e in a_spans:
        feat_a = _feat((a_s, a_e))
        cat = ID2CAT[model.cat_head(feat_a).argmax().item()]
        # 最近 opinion span (token 距离)
        nearest_o, min_d = None, float("inf")
        for o_s, o_e in o_spans:
            d = max(0, o_s - a_e, a_s - o_e)
            if d < min_d:
                min_d, nearest_o = d, (o_s, o_e)
        if nearest_o is not None:
            feat_o = _feat(nearest_o)
            pol = ID2POL[model.pol_head(feat_o).argmax().item()]
            o_text = tokens_to_text(tokenizer, id_list, nearest_o[0], nearest_o[1])
        else:
            # 无配对 opinion: 用 aspect 自己的 feature 预测 pol
            pol = ID2POL[model.pol_head(feat_a).argmax().item()]
            o_text = None
        a_text = tokens_to_text(tokenizer, id_list, a_s, a_e)
        quads.add((a_text, cat, o_text, pol))

    # 无 aspect 但有 opinion: 隐式 aspect, 用 opinion span 预测 cat & pol
    if not a_spans:
        for o_s, o_e in o_spans:
            feat_o = _feat((o_s, o_e))
            cat, pol = _predict_cat_pol(feat_o)
            o_text = tokens_to_text(tokenizer, id_list, o_s, o_e)
            quads.add((None, cat, o_text, pol))

    return quads


def gold_quads_from_sample(text, quads):
    """从训练样本提取 gold 四元组集合 (a_text|None, cat, o_text|None, pol)。
    span 在原始 text 上切片, 隐式字段为 None。
    """
    gold = set()
    for q in quads:
        a_text = None
        o_text = None
        if q.get("a_start") is not None and not pd.isna(q["a_start"]):
            a_text = text[int(q["a_start"]):int(q["a_end"])]
        if q.get("o_start") is not None and not pd.isna(q["o_start"]):
            o_text = text[int(q["o_start"]):int(q["o_end"])]
        gold.add((a_text, str(q["cat"]), o_text, str(q["pol"])))
    return gold


def evaluate(model, tokenizer, samples, device, max_len=MAX_LEN):
    """dev 集 micro P/R/F1 (四元组 exact match)。"""
    pred, gold = {}, {}
    for rid, text, quads in samples:
        pred[rid] = predict_review_quads(model, tokenizer, text, device, max_len)
        gold[rid] = gold_quads_from_sample(text, quads)
    tp = fp = fn = 0
    for rid in set(pred) | set(gold):
        p_set = pred.get(rid, set())
        g_set = gold.get(rid, set())
        tp += len(p_set & g_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"P": p, "R": r, "F1": f1, "tp": tp, "fp": fp, "fn": fn}


# ======================================================================
# 6. 标签字典保存
# ======================================================================
def save_label_vocab(path):
    vocab = {
        "bio_tags": BIO_TAGS,
        "categories": CATEGORIES,
        "polarities": POLARITIES,
        "pretrained": PRETRAINED,
    }
    path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def load_label_vocab(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ======================================================================
# 6.1 训练快照 YAML (训练完自动 dump 到 config/, 勿手工编辑)
# ======================================================================
def save_trained_snapshot(path, *, elapsed, best_f1, last_ep, last_metrics,
                          n_train, n_dev, n_total):
    """把最终生效超参 + 训练指标 + 产物路径引用 dump 成一份 YAML 到 config/。
    本文件由训练自动生成, 勿手工编辑; 改超参请改 config/model_bert.yaml。
    路径以项目根目录为基准的相对路径, 便于跨机器迁移。
    """
    import yaml

    def _rel(p):
        try:
            return str(Path(p).relative_to(BASE_DIR)).replace("\\", "/")
        except ValueError:
            return str(p)

    snap = {
        "_doc": "BERT+CRF 训练快照(自动生成, 勿手工编辑); 改超参请改 model_bert.yaml",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pretrained": PRETRAINED,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hyperparams": {
            "seed": SEED, "max_len": MAX_LEN, "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE, "epochs": EPOCHS,
            "lr_bert": LR_BERT, "lr_head": LR_HEAD,
            "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
            "grad_clip": GRAD_CLIP, "dropout": DROPOUT,
            "dev_ratio": DEV_RATIO, "amp": AMP,
        },
        "loss_weights": {"w_crf": W_CRF, "w_cat": W_CAT, "w_pol": W_POL},
        "labels": {
            "bio_tags": BIO_TAGS,
            "categories": CATEGORIES,
            "polarities": POLARITIES,
        },
        "training": {
            "duration_min": round(elapsed / 60, 2),
            "total_samples": n_total,
            "train_samples": n_train,
            "dev_samples": n_dev,
            "best_dev_f1": round(best_f1, 4),
            "last_epoch": last_ep,                 # {loss, crf, cat, pol, lr}
            "last_dev": last_metrics,              # {P, R, F1, tp, fp, fn}
        },
        "artifacts": {
            "weights": _rel(BERT_DIR / "bert_crf.pt"),
            "label_vocab": _rel(BERT_DIR / "label_vocab.json"),
            "train_log": _rel(BERT_LOG_PATH),
            "test_reviews": _rel(TEST_REVIEWS_PATH),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(snap, f, allow_unicode=True, sort_keys=False,
                  default_flow_style=False)


# ======================================================================
# 7. 主流程
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="BERT+CRF ACOS 训练")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖 YAML 中的 epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="覆盖 YAML 中的 batch_size")
    parser.add_argument("--max-len", type=int, default=None,
                        help="覆盖 YAML 中的 max_len")
    parser.add_argument("--skip-eval", action="store_true",
                        help="跳过每 epoch dev 评估(仅末次评估)")
    args = parser.parse_args()

    global EPOCHS, BATCH_SIZE, MAX_LEN
    EPOCHS = args.epochs or EPOCHS
    BATCH_SIZE = args.batch_size or BATCH_SIZE
    MAX_LEN = args.max_len or MAX_LEN

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[环境] device=%s, pretrained=%s" % (device, PRETRAINED))
    print("[超参] epochs=%d batch=%d max_len=%d lr_bert=%.g lr_head=%.g"
          % (EPOCHS, BATCH_SIZE, MAX_LEN, LR_BERT, LR_HEAD))

    # ---- 数据 ----
    samples = load_train_samples()
    rng = random.Random(SEED)
    rng.shuffle(samples)
    n_dev = max(1, int(len(samples) * DEV_RATIO))
    dev = samples[:n_dev]
    train = samples[n_dev:]
    print("[数据] 全量=%d, train=%d, dev=%d" % (len(samples), len(train), len(dev)))

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
    train_ds = ReviewDataset(train, tokenizer, MAX_LEN)
    dev_ds = ReviewDataset(dev, tokenizer, MAX_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                        collate_fn=collate_fn)

    # ---- 模型 / 优化器 ----
    model = BertCrfModel().to(device)
    bert_params = list(model.bert.parameters())
    head_params = (list(model.tag_proj.parameters())
                   + list(model.crf.parameters())
                   + list(model.cat_head.parameters())
                   + list(model.pol_head.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": LR_BERT},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dl) * EPOCHS
    warmup = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP and device.type == "cuda"))

    # ---- 训练 ----
    BERT_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []
    def log(msg=""):
        print(msg)
        log_lines.append(msg)

    log("=" * 72)
    log("BERT+CRF ACOS 训练  pretrained=%s  device=%s" % (PRETRAINED, device))
    log("=" * 72)

    best_f1 = 0.0
    last_ep = None      # {loss, crf, cat, pol, lr}
    last_metrics = {}   # {P, R, F1, tp, fp, fn}
    t0 = time.time()
    n_total = len(train_dl)
    LOG_EVERY = 20   # 每 20 batch 打印一次进度(CPU 上能看到推进)
    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss = 0.0
        ep_crf = ep_cat = ep_pol = 0.0
        n_batch = 0
        ep_t0 = time.time()
        for batch in train_dl:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            mask = attn.bool()
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(AMP and device.type == "cuda")):
                emissions, seq, cls = model.encode(input_ids, attn)
                loss, m = model.loss(emissions, labels, mask, seq, cls,
                                     batch["cat_targets"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ep_loss += loss.item()
            ep_crf += m["crf"]; ep_cat += m["cat"]; ep_pol += m["pol"]
            n_batch += 1
            if n_batch % LOG_EVERY == 0 or n_batch == n_total:
                elapsed = time.time() - ep_t0
                remain = elapsed / n_batch * (n_total - n_batch)
                print("  [Ep %d] %d/%d  loss=%.4f  %.1fs/%.1fs"
                      % (ep, n_batch, n_total, ep_loss / n_batch,
                         elapsed, remain), flush=True)
        avg = ep_loss / max(n_batch, 1)
        last_lr = scheduler.get_last_lr()[0]
        last_ep = {
            "loss": round(avg, 4),
            "crf": round(ep_crf / max(n_batch, 1), 4),
            "cat": round(ep_cat / max(n_batch, 1), 4),
            "pol": round(ep_pol / max(n_batch, 1), 4),
            "lr": last_lr,
        }
        log("[Epoch %d/%d] loss=%.4f (crf=%.4f cat=%.4f pol=%.4f)  lr=%.2e"
            % (ep, EPOCHS, avg, last_ep["crf"], last_ep["cat"], last_ep["pol"],
               last_lr))

        # ---- dev 评估 ----
        if not args.skip_eval or ep == EPOCHS:
            metrics = evaluate(model, tokenizer, dev, device, MAX_LEN)
            last_metrics = metrics
            log("         dev  P=%.4f R=%.4f F1=%.4f (tp=%d fp=%d fn=%d)"
                % (metrics["P"], metrics["R"], metrics["F1"],
                   metrics["tp"], metrics["fp"], metrics["fn"]))
            if metrics["F1"] > best_f1:
                best_f1 = metrics["F1"]
                torch.save(model.state_dict(), BERT_DIR / "bert_crf.pt")
                save_label_vocab(BERT_DIR / "label_vocab.json")
                log("         [*] best F1=%.4f, 已保存权重 -> %s"
                    % (best_f1, BERT_DIR / "bert_crf.pt"))

    elapsed = time.time() - t0
    log("[完成] 训练耗时 %.1f min, best dev F1=%.4f" % (elapsed / 60, best_f1))

    # ---- 末次务必保存(若 skip_eval 导致未保存) ----
    if not (BERT_DIR / "bert_crf.pt").exists():
        torch.save(model.state_dict(), BERT_DIR / "bert_crf.pt")
        save_label_vocab(BERT_DIR / "label_vocab.json")
        log("[输出] 末次 epoch 保存权重 -> %s" % (BERT_DIR / "bert_crf.pt"))

    # ---- 训练快照 YAML -> config/ ----
    save_trained_snapshot(
        BERT_TRAINED_SNAPSHOT_PATH,
        elapsed=elapsed, best_f1=best_f1,
        last_ep=last_ep, last_metrics=last_metrics,
        n_train=len(train), n_dev=len(dev), n_total=len(samples),
    )
    log("[输出] 训练快照: %s" % BERT_TRAINED_SNAPSHOT_PATH)

    # ---- 日志落盘 ----
    log_path = BERT_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print("[输出] 运行日志: %s" % log_path)
    print("[输出] 模型权重: %s" % (BERT_DIR / "bert_crf.pt"))
    print("[输出] 标签字典: %s" % (BERT_DIR / "label_vocab.json"))
    print("[输出] 训练快照: %s" % BERT_TRAINED_SNAPSHOT_PATH)
    print("[下一步] 推理: python src/model_bert/predict_bert.py")


if __name__ == "__main__":
    main()
