# -*- coding: utf-8 -*-
"""
化妆品评论 —— 文本预处理 + 标签解析(TRAIN / TEST 通用)

职责边界(本脚本只做两件事):
  1. 文本预处理: 数据清洗(去缺失/去重/去噪/特殊符号清理) -> jieba 中文分词 -> 谨慎去停用词;
  2. 标签解析: 将 Train_labels.csv 的占位符/位置字段规范化为结构化标签。
EDA(分布统计/异常检查/可视化/分词完整率验证)统一放在 eda.py。

文本处理设计原则:
  1. 否定词(不/没/没有/别...)与程度副词(很/非常/挺/还...)一律保留, 避免破坏情感倾向;
  2. 品牌脱敏符 ** / *** 归一化为 "某品牌", 保留语义槽而非直接删除;
  3. 短评论(如 "一般""不好""过敏")本身承载情感, 不按长度删除;
  4. 利用 Train_labels.csv 人工标注数据驱动扩充 jieba 词典, 保证方面词/观点词整词保留。
"""

import html
import re
from pathlib import Path

import pandas as pd
import jieba

# ----------------------------------------------------------------------
# 路径配置(统一由 config.py 提供, 便于集中维护; OUT_DIR 为预处理产物目录)
# ----------------------------------------------------------------------
from config import (BASE_DIR, TRAIN_REVIEWS_PATH, TRAIN_LABELS_PATH,
                    TEST_REVIEWS_PATH, PROCESSED_DIR as OUT_DIR,
                    PREPROCESS_LOG_PATH)

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
RE_SPACE = re.compile(r"[\s　 ]+")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_AT = re.compile(r"@[一-龥A-Za-z0-9_\-]+")
RE_HASHTAG = re.compile(r"#([^#]+)#")
RE_BRAND_MASK = re.compile(r"\*{2,}")                       # 品牌脱敏符
RE_NON_TEXT = re.compile(r"[^一-龥A-Za-z0-9]")              # 仅保留中英文与数字
RE_DICT_VALID = re.compile(r"^[一-龥A-Za-z0-9]+$")          # 可入分词词典的标注词
RE_PURE_DIGIT = re.compile(r"\d+$")
RE_SINGLE_LATIN = re.compile(r"[A-Za-z]$")

# 常见情感表情 -> 中文情感词(映射后其余符号统一清除)
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
POLARITIES = ["正面", "中性", "负面"]


# ----------------------------------------------------------------------
# 一、文本预处理函数
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


def filter_tokens(tokens):
    """停用词/噪声词过滤, 返回保留词列表。"""
    return [w for w in tokens if not is_noise_token(w)]


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

    stats = {
        "n_rows": len(labels),
        "n_review_ids": labels["id"].nunique(),
        "n_aspect_terms": int(labels.loc[labels["AspectTerms"] != "_",
                                         "AspectTerms"].nunique()),
        "n_opinion_terms": int(labels.loc[labels["OpinionTerms"] != "_",
                                          "OpinionTerms"].nunique()),
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


def process_reviews(df: pd.DataFrame):
    """
    对单个评论数据集执行完整文本预处理流水线。
    返回 (处理后DataFrame, 处理量统计dict)。
    """
    df = df.copy()
    n0 = len(df)

    # 1) 缺失/纯空白
    df["Reviews"] = df["Reviews"].astype(str).map(
        lambda x: RE_SPACE.sub(" ", x.replace("　", " ").replace(" ", " ")).strip()
    )
    blank_mask = df["Reviews"].eq("") | df["Reviews"].isin(["nan", "None", "NaN"])
    n_blank = int(blank_mask.sum())
    df = df.loc[~blank_mask].copy()


    # 3) 去噪/特殊符号清理 + 4) 剔除清洗后为空的无意义内容
    df["clean_text"] = df["Reviews"].map(clean_text)
    n_empty = int(df["clean_text"].eq("").sum())
    df = df.loc[~df["clean_text"].eq("")].copy()

    # 5) 分词 + 6) 去停用词
    df["tokens"] = df["clean_text"].map(
        lambda t: filter_tokens(tokenize(t)))

    stats = {
        "n0": n0, "n_blank": n_blank, 
        "n_empty": n_empty, "n_final": len(df),
        "n_zero_tokens": int((df["tokens"].map(len) == 0).sum()),
    }
    return df, stats


# ----------------------------------------------------------------------
# 二、标签解析: Train_labels.csv 占位符/位置字段规范化
#   - 术语列: "_" 占位符 -> 缺失(NA)
#   - 位置列: 空白字符串 -> NA, 其余转 Int64(A_start/A_end 半开区间字符偏移)
# ----------------------------------------------------------------------
def parse_labels(labels_path: Path) -> pd.DataFrame:
    """将原始标注表解析为结构化标签表(术语 NA 化、位置 Int64 化、字段去空白)。"""
    raw = pd.read_csv(labels_path, encoding="utf-8")

    def norm_term(s: pd.Series) -> pd.Series:
        """术语列: 去空白, '_' 占位符转 NA。"""
        return s.astype(str).str.strip().replace({"_": pd.NA})

    def norm_pos(s: pd.Series) -> pd.Series:
        """位置列: 空白转 NA, 其余转可空 Int64。"""
        return pd.to_numeric(
            s.astype(str).str.strip().replace({"": pd.NA}),
            errors="coerce",
        ).astype("Int64")

    df = pd.DataFrame({
        "id": raw["id"].astype("int64"),
        "AspectTerms": norm_term(raw["AspectTerms"]),
        "A_start": norm_pos(raw["A_start"]),
        "A_end": norm_pos(raw["A_end"]),
        "OpinionTerms": norm_term(raw["OpinionTerms"]),
        "O_start": norm_pos(raw["O_start"]),
        "O_end": norm_pos(raw["O_end"]),
        "Categories": raw["Categories"].astype(str).str.strip(),
        "Polarities": raw["Polarities"].astype(str).str.strip(),
    })
    return df


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 72)
    log("化妆品评论预处理(TRAIN + TEST) —— 文本预处理 + 标签解析")
    log("=" * 72)

    # 0. 标签驱动分词词典 + 停用词安全校验
    raw_labels, vstats = load_label_vocab(TRAIN_LABELS_PATH)
    log("[词典] 标注 %d 行/覆盖 %d 条评论; 唯一方面词 %d, 唯一观点词 %d"
        % (vstats["n_rows"], vstats["n_review_ids"],
           vstats["n_aspect_terms"], vstats["n_opinion_terms"]))
    conflicts = audit_stopwords(raw_labels)
    log("[词典] 停用词安全校验: %s"
        % ("自动剔除误收录标注词 %s" % conflicts if conflicts else "通过(0 冲突)"))

    # 1. 标签解析落盘
    parsed_labels = parse_labels(TRAIN_LABELS_PATH)
    labels_out = OUT_DIR / "Train_labels_parsed.csv"
    parsed_labels.to_csv(labels_out, index=False, encoding="utf-8-sig")
    log("[标签] 解析完成: %s (%d 行; '_'已转NA, 位置已转Int64)"
        % (labels_out, len(parsed_labels)))

    # 2. TRAIN / TEST 文本预处理
    for name, src in (("TRAIN", TRAIN_REVIEWS_PATH), ("TEST", TEST_REVIEWS_PATH)):
        df_raw = pd.read_csv(src, encoding="utf-8")
        df_out, st = process_reviews(df_raw)
        out_path = OUT_DIR / f"{name.title()}_reviews_processed.csv"
        save_df = df_out.copy()
        save_df["tokens"] = save_df["tokens"].map(lambda xs: " ".join(xs))
        save_df = save_df[["id", "Reviews", "clean_text", "tokens"]]
        save_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        log(f"[{name}] {st['n0']} -> 去空(-{st['n_blank']}) "
            f"-> 去无意义(-{st['n_empty']}) = {st['n_final']}; "
            f"0有效词评论 {st['n_zero_tokens']} 条; 输出: {out_path.name}")

    report_path = PREPROCESS_LOG_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log("[输出] 运行日志: %s" % report_path)


if __name__ == "__main__":
    main()
