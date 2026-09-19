# -*- coding: utf-8 -*-
"""
化妆品评论数据探索性分析(EDA)
数据: Train_reviews.csv / Train_labels.csv / Test_reviews.csv

分析内容:
  1. 评论数量、评论长度分布、每条评论四元组数量分布
  2. 属性种类(Category)分布、情感极性(Polarity)分布及二者交叉
  3. 缺失值、重复值、异常样本(位置-术语一致性/取值越界/id对齐等)
  4. 分词对人工标注术语的整词保留率(分词质量验证)
产物:
  outputs/logs/eda_report.txt           文字报告
  outputs/figures/*.png                 可视化图表
说明: 文本清洗/分词/标签解析逻辑复用 preprocess.py, 本脚本不重复实现。
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")                       # 无界面后端, 直接存图
import matplotlib.pyplot as plt

# 复用预处理模块(同目录)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess as pp

# ----------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 路径配置(统一由 config.py 提供)
from config import EDA_FIG_DIR as FIG_DIR, EDA_REPORT_PATH as REPORT_PATH
LENGTH_BINS = [0, 10, 20, 30, 40, 50, 200]
LENGTH_LABELS = ["1-10", "11-20", "21-30", "31-40", "41-50", "51+"]

_lines = []


def log(msg=""):
    print(msg)
    _lines.append(msg)


def title(text):
    log("")
    log("=" * 72)
    log(text)
    log("=" * 72)


def pct(n, total):
    return f"{n / total * 100:.2f}%" if total else "0.00%"


# ----------------------------------------------------------------------
# 1. 评论数量
# ----------------------------------------------------------------------
def section_overview(train_rev, test_rev, labels):
    title("一、数据规模 / 评论数量")
    log(f"训练评论 Train_reviews.csv : {len(train_rev)} 条 (id 1~{train_rev['id'].max()})")
    log(f"训练标签 Train_labels.csv  : {len(labels)} 行四元组, 覆盖 "
        f"{labels['id'].nunique()} 条评论")
    log(f"测试评论 Test_reviews.csv  : {len(test_rev)} 条 (id 1~{test_rev['id'].max()})")
    q = labels.groupby("id").size()
    log(f"平均每条训练评论含 {len(labels) / labels['id'].nunique():.2f} 个四元组; "
        f"AspectTerm 非空 {int((labels['AspectTerms'] != '_').sum())} 个, "
        f"OpinionTerm 非空 {int((labels['OpinionTerms'] != '_').sum())} 个")


# ----------------------------------------------------------------------
# 2. 评论长度分布(原始字符 / 清洗后字符 / 分词数)
# ----------------------------------------------------------------------
def length_stats(series):
    desc = series.astype(str).str.len().describe(percentiles=[.25, .5, .75, .9, .95])
    return {k: round(float(v), 2) for k, v in desc.items()}


def section_length(train_rev, test_rev):
    title("二、评论长度分布")
    for name, df in (("TRAIN", train_rev), ("TEST", test_rev)):
        raw_len = df["Reviews"].astype(str).str.len()
        clean = df["Reviews"].astype(str).map(pp.clean_text)
        clean_len = clean.str.len()
        tok_len = clean.map(lambda t: len(pp.tokenize(t)))
        log(f"[{name}] 原始字符长度: {length_stats(df['Reviews'])}")
        log(f"[{name}] 清洗后字符长度: min={clean_len.min()}, "
            f"均值={clean_len.mean():.2f}, 中位数={clean_len.median():.0f}, max={clean_len.max()}")
        log(f"[{name}] 分词数: min={tok_len.min()}, 均值={tok_len.mean():.2f}, "
            f"中位数={tok_len.median():.0f}, max={tok_len.max()}")
        binned = pd.cut(raw_len, bins=LENGTH_BINS, labels=LENGTH_LABELS,
                        right=True, include_lowest=True).value_counts().sort_index()
        log(f"[{name}] 字符长度分箱:")
        for k, v in binned.items():
            log(f"    {str(k):>6} 字: {v:>5} 条  {pct(v, len(df))}")
        log("")

    # 图: TRAIN/TEST 长度直方图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, df, name in zip(axes, (train_rev, test_rev), ("TRAIN", "TEST")):
        ax.hist(df["Reviews"].astype(str).str.len(), bins=20,
                color="#4C78A8", edgecolor="white")
        ax.set_title(f"{name} 评论字符长度分布")
        ax.set_xlabel("字符数")
        ax.set_ylabel("评论数")
    fig.tight_layout()
    out = FIG_DIR / "01_length_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"[图] {out.name}")


# ----------------------------------------------------------------------
# 3. 每条评论四元组数量分布
# ----------------------------------------------------------------------
def section_quads(train_rev, labels):
    title("三、每条评论四元组数量分布")
    counts = labels.groupby("id").size().reindex(train_rev["id"], fill_value=0)
    dist = counts.value_counts().sort_index()
    log("四元组数  评论数   占比(共%d条评论)" % len(train_rev))
    for k, v in dist.items():
        log(f"    {k:>2} 个 : {v:>5} 条  {pct(v, len(train_rev))}")
    log(f"含 0 个四元组的评论: {int((counts == 0).sum())} 条; 最多: {counts.max()} 个")

    asp_present = labels.groupby("id").apply(
        lambda g: int((g["AspectTerms"] != "_").any()), include_groups=False)
    log(f"至少含 1 个显式 AspectTerm 的评论: {int(asp_present.sum())} 条 "
        f"({pct(int(asp_present.sum()), len(train_rev))}); "
        f"其余评论仅有观点词+隐式方面")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dist.index.astype(str), dist.values, color="#59A14F", edgecolor="white")
    for i, v in enumerate(dist.values):
        ax.text(i, v + 8, str(v), ha="center", fontsize=9)
    ax.set_xlabel("每条评论的四元组数量")
    ax.set_ylabel("评论数")
    ax.set_title("TRAIN 每条评论四元组数量分布")
    fig.tight_layout()
    out = FIG_DIR / "02_quads_per_review.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"[图] {out.name}")


# ----------------------------------------------------------------------
# 4. 属性种类分布 / 5. 情感极性分布(含交叉)
# ----------------------------------------------------------------------
def section_category_polarity(labels):
    title("四、属性种类(Category)分布")
    n = len(labels)
    cat_order = labels["Categories"].value_counts()
    log("四元组总数: %d" % n)
    for k, v in cat_order.items():
        log(f"    {k:<6} {v:>5}  {pct(v, n)}")
    illegal_cat = set(labels["Categories"].dropna()) - set(pp.CATEGORIES)
    log(f"合法类别集合外取值: {illegal_cat if illegal_cat else '无'}")

    log("")
    log("高频 AspectTerm Top15:")
    asp = labels.loc[labels["AspectTerms"] != "_", "AspectTerms"]
    for k, v in asp.value_counts().head(15).items():
        log(f"    {k:<8} {v}")
    log("高频 OpinionTerm Top15:")
    opi = labels.loc[labels["OpinionTerms"] != "_", "OpinionTerms"]
    for k, v in opi.value_counts().head(15).items():
        log(f"    {k:<8} {v}")

    title("五、情感极性(Polarity)分布")
    pol_order = labels["Polarities"].value_counts()
    for k, v in pol_order.items():
        log(f"    {k:<4} {v:>5}  {pct(v, n)}")
    illegal_pol = set(labels["Polarities"].dropna()) - set(pp.POLARITIES)
    log(f"合法极性集合外取值: {illegal_pol if illegal_pol else '无'}")
    log("注: 正负样本比约 1:0.09, 正面占绝对多数, 建模需关注类别不平衡。")

    log("")
    log("类别 × 极性 交叉表(行内百分比):")
    cross = pd.crosstab(labels["Categories"], labels["Polarities"])
    cross = cross.reindex(index=cat_order.index,
                          columns=[p for p in pp.POLARITIES if p in cross.columns],
                          fill_value=0)
    log("    类别       正面      中性      负面     总数")
    for cat, row in cross.iterrows():
        tot = int(row.sum())
        log(f"    {cat:<6} "
            f"{pct(row.get('正面', 0), tot):>8} {pct(row.get('中性', 0), tot):>8} "
            f"{pct(row.get('负面', 0), tot):>8} {tot:>7}")

    # 图: 类别条形图
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(cat_order.index[::-1], cat_order.values[::-1],
            color="#4C78A8", edgecolor="white")
    ax.set_xlabel("四元组数量")
    ax.set_title("属性种类(Category)分布")
    fig.tight_layout()
    out = FIG_DIR / "03_category_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"[图] {out.name}")

    # 图: 极性条形图
    fig, ax = plt.subplots(figsize=(5.5, 4))
    colors = {"正面": "#59A14F", "中性": "#F1C232", "负面": "#E15759"}
    ax.bar(pol_order.index, pol_order.values,
           color=[colors.get(k, "#999999") for k in pol_order.index])
    for i, (k, v) in enumerate(pol_order.items()):
        ax.text(i, v + 40, f"{v}\n{pct(v, n)}", ha="center", fontsize=9)
    ax.set_ylabel("四元组数量")
    ax.set_title("情感极性分布")
    fig.tight_layout()
    out = FIG_DIR / "04_polarity_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"[图] {out.name}")

    # 图: 类别×极性堆叠条形(计数)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = [0] * len(cross)
    for pol in ["正面", "中性", "负面"]:
        vals = cross[pol].values if pol in cross.columns else [0] * len(cross)
        ax.bar(cross.index, vals, bottom=bottom, label=pol,
               color=colors[pol], edgecolor="white")
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("四元组数量")
    ax.set_title("各属性类别下的情感极性构成")
    ax.legend()
    plt.xticks(rotation=30)
    fig.tight_layout()
    out = FIG_DIR / "05_category_polarity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log(f"[图] {out.name}")


# ----------------------------------------------------------------------
# 6. 缺失值检查
# ----------------------------------------------------------------------
def section_missing(train_rev, test_rev, labels_raw, labels_parsed):
    title("六、缺失值检查")
    log("[原始文件 pandas 判定 NaN]")
    log(f"    Train_reviews : {train_rev.isna().sum().to_dict()}")
    log(f"    Test_reviews  : {test_rev.isna().sum().to_dict()}")
    log(f"    Train_labels  : {labels_raw.isna().sum().to_dict()}")

    # 评论空字符串/纯空白
    for name, df in (("TRAIN", train_rev), ("TEST", test_rev)):
        s = df["Reviews"].astype(str)
        n_ws = int(s.str.fullmatch(r"\s*").sum())
        log(f"    {name} 空字符串/纯空白评论: {n_ws} 条")

    # 标签占位符与空白位置(原始文件层面)
    n_asp_under = int((labels_raw["AspectTerms"] == "_").sum())
    n_opi_under = int((labels_raw["OpinionTerms"] == "_").sum())
    log("[标签占位符/缺失]")
    log(f"    AspectTerms='_': {n_asp_under} 行 ({pct(n_asp_under, len(labels_raw))})")
    log(f"    OpinionTerms='_': {n_opi_under} 行 ({pct(n_opi_under, len(labels_raw))})")
    for col in ("A_start", "A_end", "O_start", "O_end"):
        n_blank = int(labels_raw[col].astype(str).str.strip().eq("").sum())
        n_na = int(labels_parsed[col].isna().sum())
        log(f"    {col} 空白字符串 {n_blank} 行 -> 解析后 NA {n_na} 行")

    # 术语缺失与位置缺失的一致性
    asp_mismatch = int(((labels_parsed["AspectTerms"].isna())
                        != (labels_parsed["A_start"].isna())).sum())
    opi_mismatch = int(((labels_parsed["OpinionTerms"].isna())
                        != (labels_parsed["O_start"].isna())).sum())
    log(f"    AspectTerm缺失 与 A位置缺失 不一致行数: {asp_mismatch}")
    log(f"    OpinionTerm缺失 与 O位置缺失 不一致行数: {opi_mismatch}")


# ----------------------------------------------------------------------
# 7. 重复值检查
# ----------------------------------------------------------------------
def section_duplicates(train_rev, test_rev, labels_raw):
    title("七、重复值检查")
    for name, df in (("TRAIN", train_rev), ("TEST", test_rev)):
        s = df["Reviews"].astype(str)
        n_exact = int(s.duplicated().sum())
        n_ws = int(s.str.replace(r"\s+", "", regex=True).duplicated().sum())
        ids = df.loc[s.duplicated(), "id"].tolist()
        log(f"[{name}] 完全重复评论: {n_exact} 条; 去空白后重复: {n_ws} 条")
        log(f"       重复记录 id: {ids}")
    cross = set(train_rev["Reviews"]) & set(test_rev["Reviews"])
    log(f"TRAIN/TEST 间重复评论: {len(cross)} 条")
    n_dup_label = int(labels_raw.duplicated().sum())
    dup_rows = labels_raw[labels_raw.duplicated(keep=False)]
    log(f"标签表完全重复行: {n_dup_label} 行")
    if n_dup_label:
        log("    重复明细(原始行):\n" +
            dup_rows.to_string().replace("\n", "\n    "))


# ----------------------------------------------------------------------
# 8. 异常样本检查
# ----------------------------------------------------------------------
def section_abnormal(train_rev, test_rev, labels_raw, labels_parsed):
    title("八、异常样本检查")
    text_map = dict(zip(train_rev["id"], train_rev["Reviews"].astype(str)))

    # 8.1 id 对齐
    lab_ids, rev_ids = set(labels_raw["id"]), set(train_rev["id"])
    log(f"[id 对齐] 标签有但评论缺失的 id: {sorted(lab_ids - rev_ids) or '无'}")
    log(f"[id 对齐] 评论有但无标签的 id: {sorted(rev_ids - lab_ids) or '无'}")
    log(f"[id 对齐] 评论表 id 重复: {int(train_rev['id'].duplicated().sum())} 行; "
        f"标签表 id 重复: {int(labels_raw['id'].duplicated().sum())} 行(一条评论多标签, 正常)")

    # 8.2 位置切片与术语是否一致
    bad_pos, oob, n_checked = [], 0, 0
    for r in labels_parsed.itertuples():
        txt = text_map.get(r.id)
        if txt is None:
            continue
        checks = (("Aspect", r.AspectTerms, r.A_start, r.A_end),
                  ("Opinion", r.OpinionTerms, r.O_start, r.O_end))
        for kind, term, s, e in checks:
            if pd.isna(term):
                continue
            n_checked += 1
            if pd.isna(s) or pd.isna(e):
                bad_pos.append((r.id, kind, term, "位置缺失但术语存在"))
                continue
            s, e = int(s), int(e)
            if s < 0 or e > len(txt) or s > e:
                oob += 1
                bad_pos.append((r.id, kind, term, f"位置越界[{s},{e})/len={len(txt)}"))
            elif txt[s:e] != term:
                bad_pos.append((r.id, kind, term,
                                f"切片='{txt[s:e]}' != 标注='{term}'"))
    log(f"[位置一致性] 共校验 {n_checked} 个术语: 越界 {oob} 个, 切片不匹配 "
        f"{len(bad_pos) - oob} 个")
    for item in bad_pos[:10]:
        log(f"    id={item[0]} {item[1]} '{item[2]}' -> {item[3]}")
    if not bad_pos:
        log("    所有术语位置与原文切片完全一致。")

    # 8.3 含标点等非文本字符的标注词(清洗后不可对齐)
    weird = sorted({t for t in
                    pd.concat([labels_raw.loc[labels_raw.AspectTerms != '_', 'AspectTerms'],
                               labels_raw.loc[labels_raw.OpinionTerms != '_', 'OpinionTerms']])
                    if not pp.RE_DICT_VALID.fullmatch(str(t))})
    log(f"[标注规范] 含标点/特殊字符的标注词: {weird or '无'}(共 {len(weird)} 个, "
        f"分词词典注入与完整率校验时已跳过)")

    # 8.4 评论长度异常(超短/超长)
    for name, df in (("TRAIN", train_rev), ("TEST", test_rev)):
        L = df["Reviews"].astype(str).str.len()
        short_df = df[L <= 3]
        log(f"[长度异常] {name} 超短评论(≤3字): {len(short_df)} 条; "
            f"超长评论(>60字): {int((L > 60).sum())} 条; 最大长度 {L.max()}")
        if len(short_df):
            log("    超短样例(均承载情感, 不删除): "
                + "; ".join(f"{r.id}:{r.Reviews}" for r in short_df.head(8).itertuples()))

    # 8.5 清洗后为空的评论
    for name, df in (("TRAIN", train_rev), ("TEST", test_rev)):
        n_empty = int(df["Reviews"].astype(str).map(pp.clean_text).eq("").sum())
        log(f"[无意义内容] {name} 清洗后为空(纯符号/标点): {n_empty} 条")


# ----------------------------------------------------------------------
# 9. 分词质量: 标注术语整词保留率
# ----------------------------------------------------------------------
def section_tokenizer_quality(train_rev, labels_raw):
    title("九、分词质量验证(人工标注术语整词保留率)")
    # 标签驱动词典必须先加载, 与预处理主流程保持一致
    _, vstats = pp.load_label_vocab(pp.TRAIN_LABELS_PATH)
    pp.audit_stopwords(labels_raw)

    text_map = dict(zip(train_rev["id"], train_rev["Reviews"].astype(str)))
    cache = {}

    def tok_sets(rid):
        if rid not in cache:
            toks = pp.tokenize(pp.clean_text(text_map.get(rid, "")))
            cache[rid] = (set(toks), {w for w in toks if not pp.is_noise_token(w)})
        return cache[rid]

    total = Counter()
    hit_all = Counter()
    hit_keep = Counter()
    missed = Counter()
    for r in labels_raw.itertuples():
        for col in ("AspectTerms", "OpinionTerms"):
            term = getattr(r, col)
            if term == "_" or not pp.RE_DICT_VALID.fullmatch(str(term)):
                continue
            total[col] += 1
            s_all, s_keep = tok_sets(r.id)
            if term in s_all:
                hit_all[col] += 1
            else:
                missed[term] += 1
            if term in s_keep:
                hit_keep[col] += 1

    log(f"jieba 用户词典: {vstats['n_dict_total']} 词(标注词 "
        f"{vstats['n_label_terms_valid']} + 人工词 {vstats['n_manual_terms']})")
    for col, cn in (("AspectTerms", "方面词"), ("OpinionTerms", "观点词")):
        n = total[col]
        log(f"    {cn}(n={n}): 去停用词前 {hit_all[col]/n*100:.2f}% | "
            f"去停用词后 {hit_keep[col]/n*100:.2f}%")
    n_all = sum(total.values())
    log(f"    全部术语(n={n_all}): 去停用词前 {sum(hit_all.values())/n_all*100:.2f}% | "
        f"去停用词后 {sum(hit_keep.values())/n_all*100:.2f}%")
    log("    未整词保留 Top15(多为标注边界重叠, 如'还不错'切为整词而该处标注'不错'):")
    log("    " + ", ".join(f"{w}({c})" for w, c in missed.most_common(15)))


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train_rev = pd.read_csv(pp.TRAIN_REVIEWS_PATH, encoding="utf-8")
    test_rev = pd.read_csv(pp.TEST_REVIEWS_PATH, encoding="utf-8")
    labels_raw = pd.read_csv(pp.TRAIN_LABELS_PATH, encoding="utf-8")
    labels_parsed = pp.parse_labels(pp.TRAIN_LABELS_PATH)

    section_overview(train_rev, test_rev, labels_raw)
    section_length(train_rev, test_rev)
    section_quads(train_rev, labels_raw)
    section_category_polarity(labels_raw)
    section_missing(train_rev, test_rev, labels_raw, labels_parsed)
    section_duplicates(train_rev, test_rev, labels_raw)
    section_abnormal(train_rev, test_rev, labels_raw, labels_parsed)
    section_tokenizer_quality(train_rev, labels_raw)

    title("产物清单")
    log(f"文字报告: {REPORT_PATH}")
    log(f"图表目录: {FIG_DIR}")
    for p in sorted(FIG_DIR.glob('*.png')):
        log(f"    - {p.name}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
