# -*- coding: utf-8 -*-
"""
化妆品评论预处理流水线(TRAIN / TEST 通用)
阶段: 数据加载 -> 数据清洗(去缺失/去重/去噪/特殊符号清理) -> jieba 中文分词 -> 谨慎去停用词
优化点(利用 Train_labels.csv 人工标注):
  1. 数据驱动词典: 从标注的 AspectTerms/OpinionTerms/Categories 抽取领域词注入 jieba,
     保证人工标注的方面词/观点词在分词时作为完整语义单元保留;
  2. 标注词切分完整率量化验证: 以标注四元组为"金标准"度量分词质量;
  3. 停用词安全性交叉校验: 若停用词表误伤任何标注观点词/方面词, 自动剔除并告警。
通用设计原则:
  1. 否定词(不/没/没有/别...)与程度副词(很/非常/挺/还...)一律保留, 避免破坏情感倾向;
  2. 品牌脱敏符 ** / *** 归一化为 "某品牌", 保留语义槽而非直接删除;
  3. 短评论(如 "一般""不好""过敏")本身承载情感, 不按长度删除;
  4. 每个阶段记录样本量与数据变化, 同时输出到控制台和处理报告文件。
"""

import html
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import jieba

# ----------------------------------------------------------------------
# 路径配置(基于本脚本位置定位, 避免工作目录/反斜杠转义问题)
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]          # courseproject/courseproject
TRAIN_REVIEWS_PATH = BASE_DIR / "data" / "raw" / "TRAIN" / "Train_reviews.csv"
TRAIN_LABELS_PATH = BASE_DIR / "data" / "raw" / "TRAIN" / "Train_labels.csv"
TEST_REVIEWS_PATH = BASE_DIR / "data" / "raw" / "TEST" / "Test_reviews.csv"
OUT_DIR = BASE_DIR / "data" / "processed"

# ----------------------------------------------------------------------
# 1) 人工兜底领域词典: 化妆品评价词/方面词/情感词
#    实际领域词主要由 Train_labels.csv 数据驱动注入(见 load_label_vocab),
#    本表保证在脱离标签(如处理 TEST)时仍有基础领域切分能力
# ----------------------------------------------------------------------
DOMAIN_WORDS = [
    # 方面-产品/成分/品类
    "气垫", "BB霜", "bb霜", "隔离霜", "粉底液", "面膜", "护肤品", "卸妆水",
    "爽肤水", "乳液", "精华", "防晒霜", "口红", "色号", "成分", "酒精味",
    # 方面-功效/使用体验
    "遮瑕", "遮暇", "保湿", "补水", "控油", "美白", "提亮肤色", "肤色", "油皮",
    "干皮", "敏感肌", "吸收", "服帖", "上脸", "上妆", "脱妆", "浮粉", "卡粉",
    "掉色", "褪色", "滋润", "细腻", "油腻", "清爽", "温和", "过敏",
    "刺鼻", "好闻", "难闻", "香味", "气味", "异味", "推开",
    # 方面-服务/物流/包装/真伪/价格
    "物流", "快递", "发货", "客服", "售后", "包装", "保质期", "新鲜度",
    "正品", "假货", "真伪", "性价比", "优惠", "优惠券", "打折", "秒杀",
    "双11", "双十一", "划算", "便宜", "物美价廉", "物超所值",
    # 常见情感/评价整体词
    "好用", "不好用", "不错", "喜欢", "不喜欢", "满意", "失望", "舒服",
    "好评", "差评", "回购", "推荐", "给力", "点赞", "值得", "信赖",
    "无语", "开心", "生气", "大哭", "好笑",
    "某品牌",
]

# ----------------------------------------------------------------------
# 2) 谨慎停用词表: 仅收录无实义的虚词/代词/连接词/语气词
#    明确不收录: 否定词、程度副词、情感词、评价动词(感觉/觉得等)
#    运行时还会与标注词交叉校验, 任何标注方面词/观点词若误入此表将被自动剔除
# ----------------------------------------------------------------------
STOP_WORDS = set("""
的 地 得 了 着 过 之 其
是 在 把 被 让 给
我 你 您 他 她 它 俺 咱 我们 你们 他们 她们 它们 咱们 大家 自己 人家 别人
这 那 这个 那个 这些 那些 什么 怎么 怎样 哪 哪里 该 某 各
个 些 们
和 与 及 或 而 且 但 却 并 以及 或者 并且 但是 可是 不过 然而 然后
因为 所以 如果 虽然 因此 于是
也 都 就 才 又 再 已 已经 曾 正在 将
吗 呢 吧 啊 呀 哦 哈 啦 嘛 么 哇 哎 唉 嗯 呵 呗 嘞 呐 喔
就是 一下 一个
""".split())
# 否定词白名单(仅作说明, 不加入停用表): 不 没 没有 别 无 未 莫 非 勿
# 程度副词白名单(同上): 很 挺 蛮 超 太 更 最 还 比较 非常 特别 有点 稍微 十分

# ----------------------------------------------------------------------
# 3) 噪声清理所需的正则与映射
# ----------------------------------------------------------------------
RE_SPACE = re.compile(r"[\s\u3000\xa0]+") # 匹配普通空格、全角空格\u3000、不间断空格\xa0，1个或多个连续空白
RE_URL = re.compile(r"https?://\S+|www\.\S+") # 匹配URL：http/https开头，或者www.开头的链接
RE_HTML_TAG = re.compile(r"<[^>]+>")  # 匹配HTML标签 <xxx>
RE_AT = re.compile(r"@[一-龥A-Za-z0-9_\-]+")
RE_HASHTAG = re.compile(r"#([^#]+)#")
RE_BRAND_MASK = re.compile(r"\*{2,}")                       # 品牌脱敏符
RE_NON_TEXT = re.compile(r"[^一-龥A-Za-z0-9]")              # 仅保留中英文与数字
RE_DICT_VALID = re.compile(r"^[一-龥A-Za-z0-9]+$")          # 可入分词词典的标注词
RE_PURE_DIGIT = re.compile(r"\d+$")
RE_SINGLE_LATIN = re.compile(r"[A-Za-z]$")

# 常见情感表情 -> 中文情感词(仅 2 条样本含表情, 映射后其余符号统一清除)
EMOJI_MAP = {
    "👍": "点赞", "👎": "差评",
    "😄": "开心", "😊": "开心", "😃": "开心", "😍": "喜欢",
    "😂": "好笑", "😭": "大哭",
    "😡": "生气", "😠": "生气", "😒": "无语", "😔": "失望",
    "🌚": "无语",
}
EMOJI_TABLE = {ord(k): v for k, v in EMOJI_MAP.items()}

# 类别集合(来自 train_README.md / 标注文件)
CATEGORIES = ["包装", "成分", "尺寸", "服务", "功效", "价格", "气味",
              "使用体验", "物流", "新鲜度", "真伪", "整体", "其他"]


# ----------------------------------------------------------------------
# 文本处理函数
# ----------------------------------------------------------------------
def clean_text(text: str) -> str:
    """单条评论的去噪与特殊符号清理。"""
    t = html.unescape(text)                # &hellip -> …, &amp -> & 等实体解码
    t = t.translate(EMOJI_TABLE)           # 情感表情转文字
    t = RE_URL.sub(" ", t)
    t = RE_HTML_TAG.sub(" ", t)
    t = RE_AT.sub(" ", t)
    t = RE_HASHTAG.sub(r"\1", t)
    t = RE_BRAND_MASK.sub(" 某品牌 ", t)   # 脱敏品牌名归一化
    t = RE_NON_TEXT.sub(" ", t)            # 清除标点/特殊符号/格式标记/残留表情
    t = RE_SPACE.sub(" ", t).strip()
    return t


def tokenize(text: str):
    """jieba 精确模式分词(启用 HMM 处理未登录词)。"""
    return [w.strip() for w in jieba.cut(text, cut_all=False, HMM=True) if w.strip()]


def is_noise_token(w: str) -> bool:
    """判断分词是否为应过滤的噪声词: 纯数字串、孤立单个英文字母、停用词。"""
    if w in STOP_WORDS:
        return True
    if RE_PURE_DIGIT.fullmatch(w):         # 价格/数量等孤立数字
        return True
    if RE_SINGLE_LATIN.fullmatch(w):       # 如误入的 "s"
        return True
    return False


# ----------------------------------------------------------------------
# 标签驱动: 词典注入 / 停用词安全校验 / 切分完整率验证
# ----------------------------------------------------------------------
def load_label_vocab(labels_path: Path):
    """
    从训练标签中抽取方面词、观点词、类别词注入 jieba 用户词典。
    返回 (labels_df, 统计dict)。含标点等异常字符的标注词(如 '好，棒')跳过,
    因其在去标点后的文本中本就不可对齐。
    """
    labels = pd.read_csv(labels_path, encoding="utf-8")
    raw_terms = pd.concat([
        labels.loc[labels["AspectTerms"] != "_", "AspectTerms"],
        labels.loc[labels["OpinionTerms"] != "_", "OpinionTerms"],
    ]).dropna().astype(str)

    valid_terms, skipped = set(), []
    for t in raw_terms.unique():
        if RE_DICT_VALID.fullmatch(t):
            valid_terms.add(t)
        else:
            skipped.append(t)

    vocab = sorted(set(DOMAIN_WORDS) | valid_terms | set(CATEGORIES))
    for w in vocab:
        jieba.add_word(w)

    n_asp = labels.loc[labels["AspectTerms"] != "_", "AspectTerms"].nunique()
    n_opi = labels.loc[labels["OpinionTerms"] != "_", "OpinionTerms"].nunique()
    stats = {
        "n_rows": len(labels),
        "n_review_ids": labels["id"].nunique(),
        "n_aspect_terms": int(n_asp),
        "n_opinion_terms": int(n_opi),
        "n_label_terms_valid": len(valid_terms),
        "n_manual_terms": len(DOMAIN_WORDS),
        "n_dict_total": len(vocab),
        "skipped_terms": sorted(skipped),
    }
    return labels, stats


def audit_stopwords(labels: pd.DataFrame):
    """停用词安全性交叉校验: 标注方面词/观点词不得出现在停用词表中, 否则自动剔除。"""
    labeled = set()
    for col in ("AspectTerms", "OpinionTerms"):
        labeled |= set(labels.loc[labels[col] != "_", col].dropna().astype(str).unique())
    conflicts = labeled & STOP_WORDS
    for w in conflicts:
        STOP_WORDS.discard(w)
    return sorted(conflicts)


def validate_term_integrity(labels: pd.DataFrame, reviews: pd.DataFrame):
    """
    以人工标注为金标准, 量化分词对标注词的完整保留率。
    对每条评论的清洗文本分别在"停用词过滤前(tokens_all)/过滤后(tokens)"检查
    标注词是否作为一个完整 token 存在。含标点的异常标注词不计入分母。
    """
    text_map = dict(zip(reviews["id"], reviews["Reviews"].astype(str)))
    cache_all, cache_keep = {}, {}

    def tok_sets(rid):
        if rid not in cache_all:
            toks = tokenize(clean_text(text_map.get(rid, "")))
            cache_all[rid] = set(toks)
            cache_keep[rid] = {w for w in toks if not is_noise_token(w)}
        return cache_all[rid], cache_keep[rid]

    total = {"AspectTerms": 0, "OpinionTerms": 0}
    hit_all = {"AspectTerms": 0, "OpinionTerms": 0}
    hit_keep = {"AspectTerms": 0, "OpinionTerms": 0}
    miss_counter = Counter()

    for row in labels.itertuples():
        for col in ("AspectTerms", "OpinionTerms"):
            term = getattr(row, col)
            if term == "_" or not isinstance(term, str) or not RE_DICT_VALID.fullmatch(term):
                continue
            total[col] += 1
            s_all, s_keep = tok_sets(row.id)
            if term in s_all:
                hit_all[col] += 1
            else:
                miss_counter[term] += 1
            if term in s_keep:
                hit_keep[col] += 1

    result = {}
    for col in ("AspectTerms", "OpinionTerms"):
        n = total[col]
        result[col] = {
            "n": n,
            "integrity_all": hit_all[col] / n if n else 0.0,
            "integrity_keep": hit_keep[col] / n if n else 0.0,
        }
    n_all = sum(total.values())
    result["ALL"] = {
        "n": n_all,
        "integrity_all": sum(hit_all.values()) / n_all if n_all else 0.0,
        "integrity_keep": sum(hit_keep.values()) / n_all if n_all else 0.0,
    }
    result["top_missed"] = miss_counter.most_common(20)
    return result


# ----------------------------------------------------------------------
# 处理报告记录
# ----------------------------------------------------------------------
_report_lines = []


def log(msg: str = ""):
    print(msg)
    _report_lines.append(msg)


def process_reviews(df: pd.DataFrame, source_name: str):
    """对单个评论数据集执行完整清洗-分词-去停用词流水线, 返回处理后的 DataFrame 与统计。"""
    df = df.copy()
    n0 = len(df)
    log(f"[{source_name}] 原始数据: {n0} 行, 字段 = {list(df.columns)}")

    # 阶段 1: 结构性清洗(缺失/空白)
    df["Reviews"] = df["Reviews"].astype(str).map(
        lambda x: RE_SPACE.sub(" ", x.replace("\u3000", " ").replace("\xa0", " ")).strip()
    )
    blank_mask = df["Reviews"].eq("") | df["Reviews"].isin(["nan", "None", "NaN"])
    n_blank = int(blank_mask.sum())
    df = df.loc[~blank_mask].copy()
    log(f"[{source_name}] 清洗1-缺失: 移除空评论 {n_blank} 行; 剩余 {len(df)} 行")

    # 阶段 2: 去重(保留首次出现, id 可追溯)
    dup_mask = df["Reviews"].duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    dup_ids = df.loc[dup_mask, "id"].tolist()
    df = df.loc[~dup_mask].copy()
    log(f"[{source_name}] 清洗2-去重: 移除重复评论 {n_dup} 行; 剩余 {len(df)} 行")
    if dup_ids:
        log(f"[{source_name}] 被移除记录的 id: {dup_ids}")

    # 阶段 3: 去噪与特殊符号清理
    df["clean_text"] = df["Reviews"].map(clean_text)
    raw = df["Reviews"]
    stat = {
        "HTML实体(如 &hellip)": int(raw.str.contains(r"&\w+;|&#?\w+", regex=True).sum()),
        "品牌脱敏符(***)": int(raw.str.contains(r"\*{2,}", regex=True).sum()),
        "英文/字母": int(raw.str.contains(r"[A-Za-z]", regex=True).sum()),
        "数字": int(raw.str.contains(r"\d", regex=True).sum()),
        "表情符号": int(raw.str.contains(
            r"[\U0001F000-\U0001FAFF☀-➩⬀-⯿]", regex=True).sum()),
    }
    log(f"[{source_name}] 清洗3-去噪: 各类噪声触达评论数:")
    for k, v in stat.items():
        log(f"    - {k}: {v} 行")

    # 阶段 4: 异常/无意义内容处理
    empty_after_clean = df["clean_text"].eq("")
    n_empty = int(empty_after_clean.sum())
    df = df.loc[~empty_after_clean].copy()
    log(f"[{source_name}] 清洗4-异常: 清洗后为空(纯标点/纯符号)移除 {n_empty} 行; "
        f"剩余 {len(df)} 行(短情感评论不按长度删除)")

    # 阶段 5: jieba 分词
    df["tokens_all"] = df["clean_text"].map(tokenize)

    # 阶段 6: 停用词/噪声词过滤
    removed_counter = Counter()

    def filter_tokens(tokens):
        kept = []
        for w in tokens:
            if is_noise_token(w):
                removed_counter[w] += 1
            else:
                kept.append(w)
        return kept

    df["tokens"] = df["tokens_all"].map(filter_tokens)

    stats = {
        "n0": n0, "n_blank": n_blank, "n_dup": n_dup, "n_empty": n_empty,
        "n_final": len(df),
        "mean_len_raw": df["Reviews"].str.len().mean(),
        "mean_len_clean": df["clean_text"].str.len().mean(),
        "mean_tok_all": df["tokens_all"].map(len).mean(),
        "mean_tok": df["tokens"].map(len).mean(),
        "n_zero_tokens": int((df["tokens"].map(len) == 0).sum()),
        "removed_top": removed_counter.most_common(10),
    }
    return df, stats


def report_dataset(name: str, df: pd.DataFrame, st: dict, out_path: Path, n_samples: int = 5):
    """输出单数据集的汇总、抽样与落盘。"""
    log(f"[{name}] 样本量变化: {st['n0']} -> 去空(-{st['n_blank']}) -> "
        f"去重(-{st['n_dup']}) -> 去无意义(-{st['n_empty']}) = {st['n_final']}")
    log(f"[{name}] 文本长度: 原文平均 {st['mean_len_raw']:.2f} -> "
        f"清洗后 {st['mean_len_clean']:.2f} 字符; "
        f"分词: {st['mean_tok_all']:.2f} -> {st['mean_tok']:.2f} 词/条(去停用词后)")
    log(f"[{name}] 过滤后 0 个有效词的评论: {st['n_zero_tokens']} 条(保留不删)")
    log(f"[{name}] 被过滤高频词 Top10(确认无情感词误删): "
        + ", ".join(f"{w}({c})" for w, c in st["removed_top"]))
    log(f"[{name}] 前 {n_samples} 条抽样:")
    for _, row in df.head(n_samples).iterrows():
        log(f"  id={row['id']} | 原文: {row['Reviews']}")
        log(f"         分词: {' / '.join(row['tokens'])}")

    out_df = df[["id", "Reviews", "clean_text", "tokens"]].copy()
    out_df["tokens"] = out_df["tokens"].map(lambda xs: " ".join(xs))
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"[{name}] 已输出: {out_path} ({len(out_df)} 行)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 72)
    log("化妆品评论预处理报告(TRAIN + TEST, 标签驱动词典)")
    log("=" * 72)

    # ---- 0. 加载训练标签, 数据驱动注入分词词典 ----
    labels, vstats = load_label_vocab(TRAIN_LABELS_PATH)
    log("[标签词典] 标注行数 %d, 覆盖评论 %d 条" % (vstats["n_rows"], vstats["n_review_ids"]))
    log("[标签词典] 唯一方面词 %d 个, 唯一观点词 %d 个"
        % (vstats["n_aspect_terms"], vstats["n_opinion_terms"]))
    log("[标签词典] 注入 jieba: 标注词 %d + 人工词 %d(去重合并) => 词典共 %d 词"
        % (vstats["n_label_terms_valid"], vstats["n_manual_terms"], vstats["n_dict_total"]))
    if vstats["skipped_terms"]:
        log("[标签词典] 含标点等异常字符已跳过(清洗后不可对齐): %s" % vstats["skipped_terms"])

    # ---- 0b. 停用词安全性交叉校验 ----
    conflicts = audit_stopwords(labels)
    if conflicts:
        log("[停用词校验] 警告: 以下标注词误入停用词表, 已自动剔除: %s" % conflicts)
    else:
        log("[停用词校验] 通过: 0 个标注方面词/观点词被停用词表收录")

    # ---- 0c. 标注词切分完整率验证(金标准度量) ----
    train_reviews_raw = pd.read_csv(TRAIN_REVIEWS_PATH, encoding="utf-8")
    integrity = validate_term_integrity(labels, train_reviews_raw)
    log("[分词验证] 标注词整词保留率(TRAIN 金标准):")
    for col, label in (("AspectTerms", "方面词"), ("OpinionTerms", "观点词"), ("ALL", "全部")):
        r = integrity[col]
        log(f"    - {label}(n={r['n']}): 去停用词前 {r['integrity_all']*100:.2f}% | "
            f"去停用词后 {r['integrity_keep']*100:.2f}%")
    if integrity["top_missed"]:
        log("[分词验证] 仍未整词保留的高频标注词 Top20(可继续补词典):")
        log("    " + ", ".join(f"{w}({c})" for w, c in integrity["top_missed"]))

    # ---- 1/2. TRAIN、TEST 两套数据同一流水线处理 ----
    log("-" * 72)
    train_out, train_stats = process_reviews(train_reviews_raw, "TRAIN")
    report_dataset("TRAIN", train_out, train_stats,
                   OUT_DIR / "Train_reviews_processed.csv")

    log("-" * 72)
    test_df = pd.read_csv(TEST_REVIEWS_PATH, encoding="utf-8")
    test_out, test_stats = process_reviews(test_df, "TEST")
    report_dataset("TEST", test_out, test_stats,
                   OUT_DIR / "Test_reviews_processed.csv")

    log("-" * 72)
    report_path = OUT_DIR / "preprocess_report.txt"
    report_path.write_text("\n".join(_report_lines), encoding="utf-8")
    log(f"[输出] 处理报告: {report_path}")


if __name__ == "__main__":
    main()
