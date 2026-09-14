# -*- coding: utf-8 -*-
"""
Double-Propagation-ACOS 规则基线模型(化妆品评论 ACOS 四元组抽取)
====================================================================

算法依据
--------
Double-Propagation-ACOS 是 ASQE/ASQP 任务的经典规则基线
(Zhu et al., 2023, World Wide Web):
    "First, the model follows the DP algorithm to extract the triples
     (a, o, s), where the model utilizes the syntactic relations and the
     sentiment lexicon. Then, the model identifies the aspect category of
     each extracted triple."
DP = Double Propagation(Qiu et al., 2011, Computational Linguistics):
  仅需情感种子词典, 利用依存句法关系在 "观点词(O)" 与 "方面词(A)" 之间
  双向自举(bootstrapping)传播, 包含四类传播规则:
      R1: 观点词 -> 方面词   (O->A, OT 直接依存)
      R2: 方面词 -> 观点词   (A->O, OT 直接依存)
      R3: 方面词 -> 方面词   (A->A, 并列/共同依存)
      R4: 观点词 -> 观点词   (O->O, 并列关系, 含极性传播)

本实现的工程化适配(中文电商评论)
--------------------------------
  1. 情感种子词典由训练标签归纳(观点词->极性多数投票), 另配少量通用中文
     情感词手工种子用于发现未登录观点词;
  2. 方面词先由 R1/R3 从句法中抽取, 训练标签归纳出的方面词词典用于
     (a) 短语归一化(复合名词合并为标注级短语), (b) 高精度锚点补充;
  3. 中文依存句法由 spaCy + zh_core_web_sm(Universal Dependencies)给出;
  4. 否定词(不/没/没有...)作用域内极性翻转; 转折并列极性翻转, 顺承并列保持;
  5. 类别识别为规则流水线: 方面词->类别映射 > 类别关键词(标注归纳+手工)
     > 观点词->类别映射 > 子句关键词投票 > 默认"整体";
  6. 隐式方面(A=_): 观点词在子句内无法依存/邻近链接到任何方面词时产生;
     隐式观点(O=_): 仅对训练集中出现过"隐式观点"用法的方面词高精度补产。

运行方式
--------
    python model_baseline.py                # dev 评估 + 全量训练 + 测试推理
    python model_baseline.py --skip-test    # 仅做 dev 评估
输出目录: data/baseline/
"""

import argparse
import html
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import jieba

# 复用预处理模块的常量(导入无副作用, 不会触发其 main)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import CATEGORIES, POLARITIES, EMOJI_TABLE  # noqa: E402

# ======================================================================
# 0. 路径与超参
# ======================================================================
# 路径配置(统一由 config.py 提供, 便于集中维护; OUT_DIR 为基线产物目录)
from config import (BASE_DIR, TRAIN_REVIEWS_PATH, TRAIN_LABELS_PATH,
                    TEST_REVIEWS_PATH, BASELINE_DIR as OUT_DIR,
                    BASELINE_LOG_PATH)

POS, NEU, NEG = "正面", "中性", "负面"
CLAUSE_PUNCT = set("，。！？；、,.!?;")
MAX_PHRASE_LEN = 8          # 词典短语最大长度(标注观点词最长 8 字)
MAX_PROP_ITER = 8           # 双传播最大迭代轮数
DEV_RATIO = 0.15            # 训练集留出 dev 比例
SEED = 42
GAP_WINDOW = 4              # 同子句内 aspect-opinion 邻近兜底窗口(token)
DF_PRUNE = 0.5              # 方面词文档频率剪枝阈值(Qiu 2011 频率剪枝)
# 隐式方面四元组建: 观点词"训练被标注率"门控(压制 一直用/用完/多/少 等标注稀疏词)
IMPLICIT_RATE_MIN = 0.25    # 标注率低于此值则不允许作为隐式方面观点发射
IMPLICIT_RATE_OCC_MIN = 3   # 出现文档数达到此值才启用门控(小样本不武断过滤)

# ----------------------------------------------------------------------
# 1) 通用中文情感种子(仅用于发现训练集未覆盖的观点词, 高精度通用词)
# ----------------------------------------------------------------------
MANUAL_SEED = {
    POS: set("""好 棒 优秀 赞 完美 舒服 舒适 顺手 惊艳 贴心 放心 用心 精致 漂亮
给力 牛 神器 水润 轻薄 透气 自然 温和 高级 嫩滑 滑嫩 亮 白嫩 白嫩 划算 实惠
便宜 快 神速 满意 喜欢 爱 值得 推荐 回购 好评 点赞 开心 惊喜 靠谱 良心 实在
香 好闻 清淡 补水 保湿 滋润 贴心 正品 超值 值""".split()),
    NEU: set("一般 还行 普通 中等 正常 平常 还好 将就".split()),
    NEG: set("""差 烂 垃圾 糟 差劲 糟糕 假 坑 骗人 失望 无语 生气 后悔 过敏 痒
红肿 刺痛 干 油 油腻 卡粉 浮粉 脱妆 掉色 褪色 贵 慢 迟 破损 破 漏 碎 脏 旧
过期 暗沉 紧绷 搓泥 闷 厚重 黏稠 粘 恶心 讨厌 麻烦 简陋 随便 敷衍 小气 抠门
亏 刺鼻 难闻 疼 痛 泛红 起皮 脱皮 假滑 拔干 难用 不好""".split()),
}

# R2/R4 观点词候选的动词处理: 中文形容词(好/差/喜欢/划算)在 zh_core_web_sm
# 中普遍被标为 VERB, 故不使用动词白名单, 改为"功能性动词黑名单 + 极性线索门控"
# 注: 发货/送货/到货 等物流词不在此表 —— 它们是金标物流类方面词
FUNCTIONAL_VERBS = set("""是 有 买 卖 用 试 试过 擦 抹 涂 花 送 收 收到 来 去 做
给 让 使 说 看 看到 看见 想 觉得 感觉 知道 发现 听说 评价 评 认为 以为 开始
继续 选择 决定 打算 记得 忘记 查 验 验证 下单 拍 付款 申请 联系 找 换 退
退款 退货 解决 处理 拿 放 挤 洗 敷 喷 倒 带 等 算 觉得 讲 告诉
使用 适合 属于 显得 变 变成 保持 需要 应该 会 能 可以 以为 看起来 看来""".split())

# 否定词: 作用域内翻转极性
NEGATORS = {"不", "没", "没有", "别", "无", "未", "莫", "非", "勿", "不是", "没什么"}
# 转折/顺承并列标记(R4 极性传播)
CONTRAST_CC = {"但", "但是", "可", "可是", "不过", "然而", "却", "反而", "倒是"}
COORD_CC = {"和", "与", "且", "并且", "以及", "或者", "又", "也", "还", "而且", "并", "既"}
# R2 程度副词: 修饰"未知形容词"时倾向认定为情感表达
DEGREE_ADV = {"很", "非常", "挺", "蛮", "超", "太", "好", "真", "特别", "十分",
              "比较", "有点", "稍微", "更", "最", "格外", "异常", "相当", "好容易",
              "还", "还能", "蛮", "特别地", "真的", "确实", "的确", "超级", "巨"}
# 观点短语尾部语气词(高频茎优先时剥离; 的 对重叠式形容词保留)
FINAL_PARTICLES = ("的", "了", "吧", "啊", "啦", "呢", "嘛")

# 通用名词(即使被抽为方面也按隐式方面处理, 与标注习惯一致)
GENERIC_TARGETS = {"产品", "商品", "宝贝", "东西", "东东", "牌子", "品牌",
                   "某品牌", "整体", "总体", "这款", "此款", "一款", "这家"}
# 产品品类名词: 训练集中从不(或几乎不)作方面标注, "很好的隔离霜/很不错的面膜"
# 金标一律按隐式方面处理; 由 R3 误传播为方面时剔除
PRODUCT_TERMS = {"面膜", "隔离霜", "洗面奶", "面霜", "眼霜", "防晒霜", "洁面乳",
                 "洁面奶", "BB霜", "bb霜", "气垫", "气垫cc", "CC霜", "cc霜",
                 "套装", "护肤品", "化妆品", "卸妆水", "爽肤水", "护手霜",
                 "身体乳", "口红", "唇膏", "香水", "粉底", "精华液", "素颜霜",
                 "奶"}
# 身体部位词: 金标从不作方面("不刺激皮肤" -> 隐式方面)
BODY_TERMS = {"皮肤", "脸", "脸上", "手部", "双手", "手", "眼周", "眼睛",
              "嘴唇", "唇", "脖子", "身体", "头皮", "头发"}
# 名词块拆分尾中心: 长度>=3 的复合块若可切成两个训练方面词, 优先拆开
# ("隔离/保湿效果" 金标为两个方面四元组, 共享观点"好")
SPLIT_BLOCK_HEADS = {"效果"}
PRODUCT_SUFFIXES = ("面膜",)   # 眼膜/手膜/鼻膜 等以"膜"结尾的品类
# 观点词典中的标注残留(时助词/语气副词被少量误标为观点), 单独命中时丢弃
OPINION_STOPWORDS = {"过", "次", "真的", "正好"}
# 指人名词(金标从不作方面; "同事也很喜欢" 类)
PERSON_TERMS = {"同事", "朋友", "好友", "老婆", "老公", "妈妈", "爸爸", "妈",
                "爸", "姐姐", "哥哥", "妹妹", "弟弟", "家人", "人家", "别人",
                "大家", "媳妇", "姑娘", "宝宝", "小孩", "儿子", "女儿", "闺蜜"}
# 评价行为词, 金标中只作观点不作方面(词典里有零星误标)
ASPECT_STOPWORDS = {"好评", "差评"}
# 价格类无观点词描述的触发词(做活动 -> 隐式正面观点)
PRICE_CUE_TERMS = {"活动", "促销", "打折", "特价", "优惠", "秒杀", "活动价"}

# ----------------------------------------------------------------------
# 2) 类别关键词手工兜底表(标注归纳表之外的未登录防线)
#    词均为多字, 匹配 spaCy token(包含关系); 类别顺序即平局时的优先级
# ----------------------------------------------------------------------
MANUAL_CATEGORY_KW = {
    "物流": "物流 快递 发货 送货 到货 运送 速度 小哥 运费 顺丰 圆通 中通 韵达 申通 邮政 EMS".split(),
    "价格": "价格 价位 价钱 活动 优惠 优惠券 折扣 赠品 礼品 礼物 小样 性价比 便宜 划算 实惠 物美价廉 物超所值 打折 秒杀 双11 双十一 降价 差价 价钱 减价 促销".split(),
    "包装": "包装 包装盒 外包装 包材 盒子 瓶装 瓶子 罐装 封口 密封 包装纸 箱子 快递盒".split(),
    "气味": "味道 香味 气味 异味 香气 好闻 难闻 刺鼻 浓香 清香 酒精味".split(),
    "真伪": "正品 假货 真货 真伪 山寨 高仿 旗舰店 官网 防伪 行货".split(),
    "服务": "客服 售后 服务 态度 卖家 店家 商家 小二 导购".split(),
    "成分": "成分 酒精 激素 添加剂 配方 香精 防腐剂 荧光剂 铅汞".split(),
    "功效": "补水 保湿 控油 美白 提亮 遮瑕 遮暇 防晒 卸妆 祛斑 抗皱 紧致 滋润 持久 修复 防护 淡斑 滋养 嫩肤 抗老 抗氧化 防水 防汗 护肤".split(),
    "使用体验": "细腻 服帖 贴合 肤色 颜色 色号 色差 油皮 干皮 敏感肌 过敏 卡粉 浮粉 脱妆 掉色 推开 吸收 上妆 上脸 质感 质地 触感 轻薄 清爽 油腻 温和 搓泥 假白 暗沉 透气 自然 毛孔 肤感 起皮 拔干".split(),
    "尺寸": "尺寸 大小 容量 规格 克数 毫升 片数 个数 净含量".split(),
    "新鲜度": "保质期 新鲜 过期 生产日期 日期".split(),
    "整体": "产品 商品 宝贝 东西 牌子 品牌 某品牌 整体 总体 质量".split(),
}
# 单字气味锚点(token 以该字结尾时命中, 如 香味/异味/味道)
CHAR_CATEGORY_KW = {"味": "气味"}

# ----------------------------------------------------------------------
# 3) 轻量清洗(保留子句标点; 供依存解析使用, 不与 preprocess 的去标点流程混用)
# ----------------------------------------------------------------------
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_TAG = re.compile(r"<[^>]+>")
_RE_AT = re.compile(r"@[一-龥A-Za-z0-9_\-]+")
_RE_HASHTAG = re.compile(r"#([^#]+)#")
_RE_BRAND = re.compile(r"\*{2,}")
_RE_SPACE = re.compile(r"\s+")
_RE_KEEP = re.compile(r"[^一-龥A-Za-z0-9，。！？；、,.!?;]")


def light_clean(text: str) -> str:
    """供句法解析的清洗: 去 url/@/标签/表情转写/品牌脱敏归一, 但保留子句标点。"""
    t = html.unescape(str(text))
    t = t.translate(EMOJI_TABLE)
    t = _RE_URL.sub(" ", t)
    t = _RE_TAG.sub(" ", t)
    t = _RE_AT.sub(" ", t)
    t = _RE_HASHTAG.sub(r"\1", t)
    t = _RE_BRAND.sub(" 某品牌 ", t)
    t = _RE_KEEP.sub(" ", t)
    return _RE_SPACE.sub(" ", t).strip()


def norm_term(t):
    """术语归一: 空白/'_' -> None; 去首尾空格。"""
    if t is None:
        return None
    s = str(t).strip()
    return None if s in ("", "_", "nan", "None", "NaN") else s


# ======================================================================
# 1. 数据加载
# ======================================================================
def load_raw_data():
    reviews_tr = pd.read_csv(TRAIN_REVIEWS_PATH, encoding="utf-8")
    reviews_te = pd.read_csv(TEST_REVIEWS_PATH, encoding="utf-8")
    labels = pd.read_csv(TRAIN_LABELS_PATH, encoding="utf-8")
    labels.columns = [c.strip() for c in labels.columns]
    labels = labels.rename(columns={"AspectTerm": "AspectTerms"})  # 容错
    labels["id"] = labels["id"].astype(int)
    for c in ("AspectTerms", "OpinionTerms", "Categories", "Polarities"):
        labels[c] = labels[c].map(norm_term)
    return reviews_tr, labels, reviews_te


# ======================================================================
# 2. 知识库(规则模型的"训练"产物)
# ======================================================================
@dataclass
class Knowledge:
    # 观点词 -> {pol: 计数}; 归纳自训练集
    opinion_pol: dict = field(default_factory=dict)
    # 观点词 -> {cat: 计数}(全部行 / 仅隐式方面行)
    o2cat_all: dict = field(default_factory=dict)
    o2cat_imp: dict = field(default_factory=dict)
    # 方面词 -> {cat: 计数}
    a2cat: dict = field(default_factory=dict)
    # 训练集中以"隐式观点"出现的方面词: (a,cat) -> {pol: 计数}
    implicit_opinion: dict = field(default_factory=dict)
    # 类别 -> {方面词: 计数}(供类别关键词归纳与短语包含匹配)
    cat_aspects: dict = field(default_factory=lambda: defaultdict(Counter))
    aspect_vocab: set = field(default_factory=set)
    opinion_vocab: set = field(default_factory=set)
    seed_opinions: dict = field(default_factory=dict)   # 短语 -> 极性(含手工种子)
    opinion_freq: dict = field(default_factory=dict)    # 观点短语 -> 训练标注行数
    # 观点短语在训练文本中的"被标注率": 标注文档数 / 出现文档数(隐式四元组建门控用)
    o_occ_n: dict = field(default_factory=dict)
    o_annot_rate: dict = field(default_factory=dict)

    def majority(self, table, key, default=None):
        d = table.get(key)
        if not d:
            return default
        return max(d.items(), key=lambda kv: (kv[1], kv[0] == POS))[0]


def build_knowledge(labels: pd.DataFrame) -> Knowledge:
    """由标注四元组归纳观点极性、方面->类别、观点->类别、隐式观点等规则知识。"""
    kn = Knowledge()
    op_pol = defaultdict(Counter)
    o2c_all = defaultdict(Counter)
    o2c_imp = defaultdict(Counter)
    a2c = defaultdict(Counter)
    imp_o = defaultdict(Counter)
    cat_a = defaultdict(Counter)

    for a, o, c, s in labels[["AspectTerms", "OpinionTerms",
                              "Categories", "Polarities"]].itertuples(index=False):
        if o:
            op_pol[o][s] += 1
            o2c_all[o][c] += 1
            if not a:
                o2c_imp[o][c] += 1
        if a:
            a2c[a][c] += 1
            cat_a[c][a] += 1
            if not o:
                imp_o[(a, c)][s] += 1

    kn.opinion_pol = dict(op_pol)
    kn.o2cat_all = dict(o2c_all)
    kn.o2cat_imp = dict(o2c_imp)
    kn.a2cat = dict(a2c)
    kn.implicit_opinion = dict(imp_o)
    kn.cat_aspects = {c: dict(v) for c, v in cat_a.items()}
    kn.aspect_vocab = set(a2c)
    kn.opinion_vocab = set(op_pol)
    kn.opinion_freq = {o: sum(cnts.values()) for o, cnts in op_pol.items()}

    # 种子观点: 训练归纳(多数极性) + 手工通用种子
    seeds = {o: max(cnts.items(), key=lambda kv: kv[1])[0]
             for o, cnts in op_pol.items()}
    n_manual = 0
    for pol, words in MANUAL_SEED.items():
        for w in words:
            if w not in seeds:
                n_manual += 1
            seeds.setdefault(w, pol)
    kn.seed_opinions = seeds
    kn._n_manual_seed = n_manual
    return kn


def compute_annot_rates(kn: Knowledge, labels: pd.DataFrame, texts):
    """
    统计每个观点短语在训练文本中的文档级出现次数与"被标注率"。
      texts: 可迭代的 (id, raw_text); 仅统计 fit 划分内文本, 防止 dev 泄漏。
    2 字及以下词按 jieba token 精确命中(避免 多/好 等子串误伤),
    3 字及以上短语允许字符包含(覆盖 一直用/值得推荐 等被切碎短语)。
    """
    label_docs = defaultdict(set)
    for rid, o in labels[["id", "OpinionTerms"]].itertuples(index=False):
        if o:
            label_docs[o].add(int(rid))
    occ = Counter()
    for _, txt in texts:
        s = str(txt)
        toks = set(jieba.lcut(s))
        for o in kn.opinion_vocab:
            if (len(o) <= 2 and o in toks) or (len(o) >= 3 and o in s):
                occ[o] += 1
    kn.o_occ_n = {o: n for o, n in occ.items()}
    kn.o_annot_rate = {o: len(label_docs[o]) / n for o, n in occ.items() if n}


def assign_category(kn: Knowledge, a, o, clause_tokens):
    """
    类别识别规则流水线。
      a              : 方面词字符串或 None(隐式方面)
      o              : 观点词字符串或 None
      clause_tokens  : 子句 token 文本列表(关键词投票用)
    """
    # ---- 显式方面路径 ----
    if a:
        c = kn.majority(kn.a2cat, a)
        if c:
            return c, "aspect_exact"
        # 标注方面词包含匹配(双向包含, 取最长命中类别)
        best, best_len = None, 0
        for cat, terms in kn.cat_aspects.items():
            for t in terms:
                if len(t) >= 2 and (t in a or a in t) and len(t) > best_len:
                    best, best_len = cat, len(t)
        if best:
            return best, "aspect_term_contains"
        hit = _keyword_hit(a, clause_tokens)
        if hit:
            return hit, "aspect_keyword"
        c = kn.majority(kn.o2cat_all, o) if o else None
        if c:
            return c, "opinion_map"
        return _clause_vote(clause_tokens) or ("整体", "default")

    # ---- 隐式方面路径: 观点词->类别(隐式行归纳, 覆盖率最高) ----
    c = kn.majority(kn.o2cat_imp, o) if o else None
    if c:
        return c, "opinion_implicit_map"
    c = kn.majority(kn.o2cat_all, o) if o else None
    if c:
        return c, "opinion_all_map"
    hit = _keyword_hit(None, clause_tokens)
    if hit:
        return hit, "clause_keyword"
    return "整体", "default"


def _keyword_hit(a, clause_tokens):
    """对手工类别关键词做 token 命中; a 给定时只匹配方面词本身。"""
    targets = [a] if a else clause_tokens
    if not targets:
        return None
    scores = Counter()
    for tok in targets:
        if not tok:
            continue
        for cat, kws in MANUAL_CATEGORY_KW.items():
            for kw in kws:
                if tok == kw:
                    scores[cat] += 3
                elif len(kw) >= 2 and kw in tok:
                    scores[cat] += 1
        for ch, cat in CHAR_CATEGORY_KW.items():
            if tok.endswith(ch) and len(tok) <= 3:
                scores[cat] += 2
    return scores.most_common(1)[0][0] if scores else None


def _clause_vote(clause_tokens):
    hit = _keyword_hit(None, clause_tokens)
    return (hit, "clause_keyword") if hit else None


# ======================================================================
# 3. 中文依存句法解析(spaCy + zh_core_web_sm)
# ======================================================================
# UD 关系族(以 zh_core_web_sm 的 Chinese-GSD 标注为准; 其宾语标签为 dobj)
OT_ARC_RELS = {                      # 观点词 <-> 方面词 的直接依存弧
    "amod", "nsubj", "nsubj:pass", "obj", "dobj", "iobj", "obl", "obl:patient",
    "obl:arg", "acl", "acl:relcl", "attr", "xcomp", "ccomp",
}
# 名词性复合修饰(方面短语中心词向外跟随用)
A_COMPOUND_RELS = {"compound", "compound:nn", "flat", "flat:name", "nmod"}
# 高信任的"谓词性观点->名词方面"关系(主/宾语位置), 新词可直接接受
SUBJ_RELS = {"nsubj", "nsubj:pass", "amod", "attr", "acl", "acl:relcl"}
# 2-hop 中介关系允许的桥接 dep(如 补水->保湿(dep)->好)
# 注: conj 不入桥接集 —— "发货迅速而且质量不错" 中并列谓词各有自己的
# 主宾语, 跨配对(不错-发货/迅速-质量)是金标不取的解析器并列结构
BRIDGE_DEPS = {"dep", "ccomp", "xcomp", "advcl", "parataxis"}
NOUN_POS = {"NOUN", "PROPN"}


@dataclass
class Token:
    i: int
    text: str
    pos: str
    dep: str
    head: int
    idx: int                 # 字符偏移
    clause: int = 0
    children: list = field(default_factory=list)


@dataclass
class Span:
    text: str
    t0: int
    t1: int                  # 半开 token 区间
    head: int                # 句法中心 token 下标(尽量取区间外中心)
    kind: str                # 'o' / 'a'
    src: str = ""            # 来源: dict / R1 / R3 / seed_a / gap ...
    pol: str = None          # 观点词极性(传播时赋值)


@dataclass
class ParsedDoc:
    rid: int
    text: str
    tokens: list
    clauses: list            # list[list[int]]
    char2tok: dict


class ZhParser:
    def __init__(self, vocab_words=None):
        try:
            import spacy
            import jieba
        except ImportError as e:
            raise RuntimeError(
                "缺少依赖, 请先安装: pip install spacy 并下载中文模型 "
                "zh_core_web_sm (python -m spacy download zh_core_web_sm)"
            ) from e
        self.nlp = spacy.load("zh_core_web_sm", disable=["ner"])
        # 领域词注入: 仅 jieba.add_word(spaCy 的 ChineseTokenizer 实测使用
        # 独立 jieba 实例且不支持 token_match, 二者对 spaCy 均不生效)。
        # 多词短语(遮暇功能/活动价/很好用)不依赖分词合并 —— scan_phrases
        # 按字符区间最长匹配, 仅要求区间边界与 token 边界对齐; DP 的 R1-R4
        # 新词发现本身也只在单词 token 级进行。
        for w in {w for w in (vocab_words or []) if w}:
            jieba.add_word(w)

    def parse_many(self, rows):
        """rows: list[(rid, text)] -> dict rid -> ParsedDoc"""
        texts = [light_clean(t) for _, t in rows]
        out = {}
        for doc, (rid, _) in zip(self.nlp.pipe(texts, batch_size=256), rows):
            out[rid] = self._convert(rid, doc)
        return out

    @staticmethod
    def _convert(rid, doc):
        toks, char2tok = [], {}
        clause = 0
        for i, t in enumerate(doc):
            if t.text in CLAUSE_PUNCT or (t.is_punct and t.text in CLAUSE_PUNCT):
                clause += 1
                continue
            if t.is_punct or not t.text.strip():
                continue
            tk = Token(i=i, text=t.text, pos=t.pos_, dep=t.dep_,
                       head=t.head.i, idx=t.idx, clause=clause)
            toks.append(tk)
            for ch in range(t.idx, t.idx + len(t.text)):
                char2tok[ch] = len(toks) - 1
        # children 索引(用压缩后下标映射)
        idx_map = {t.i: k for k, t in enumerate(toks)}
        for k, t in enumerate(toks):
            h = idx_map.get(t.head)
            t.head = h if h is not None and h != k else k
            t.children = [idx_map[c.i] for c in doc[t.i].children
                          if c.i in idx_map]
        # 子句聚合
        n_clause = clause + 1
        clauses = [[] for _ in range(n_clause)]
        for k, t in enumerate(toks):
            clauses[t.clause].append(k)
        return ParsedDoc(rid=rid, text=doc.text, tokens=toks,
                         clauses=clauses, char2tok=char2tok)


# ======================================================================
# 4. 词典短语扫描(最长匹配 -> token 区间)
# ======================================================================
def scan_phrases(pdoc: ParsedDoc, phrases: set, kind: str, src: str,
                 extra_token_texts=None, pos_filter=None, pol_map=None,
                 freq_map=None):
    """
    在解析文本上做词典最长匹配, 返回 Span 列表。
      phrases         : 短语词典(精确字符串)
      extra_token_texts: 允许的单词 token 集合(传播新词/未登录)
      pos_filter      : 单词 token 命中时的词性约束
      pol_map         : kind='o' 时的短语极性表
      freq_map        : kind='o' 时的短语训练频次(修剪取舍用)
    """
    text, toks = pdoc.text, pdoc.tokens
    occupied = [False] * len(toks)
    spans, used_chars = [], [False] * len(text)

    # --- 1. 短语最长匹配(优先, 保证标注级整词) ---
    for start in range(len(text)):
        if used_chars[start] or text[start] == " ":
            continue
        hit = None
        for L in range(MAX_PHRASE_LEN, 0, -1):
            piece = text[start:start + L]
            if piece in phrases and " " not in piece:
                hit = piece
                break
        if not hit:
            continue
        t0 = pdoc.char2tok.get(start)
        t1c = pdoc.char2tok.get(start + len(hit) - 1)
        if t0 is None or t1c is None:
            continue
        # 边界必须与 token 对齐(避免 "好像" 中误匹配单词 "好" 之类的词内命中)
        if toks[t0].idx != start:
            continue
        if toks[t1c].idx + len(toks[t1c].text) != start + len(hit):
            continue

        # 观点短语语言学修剪(贴合标注惯例):
        #  (i) 尾部名词修剪 —— "淡淡的香味"(2次) -> "淡淡的"(26次):
        #      短语句法中心是尾部 NOUN, 且同起点存在更短观点短语
        # (ii) 焦点副词前缀修剪 —— "都没有"(1次) -> "没有"(10次):
        #      程度副词(很/太/还...)与否定词保留(很是划算/还不错/不是很喜欢)
        if kind == "o":
            h0 = _span_head(toks, t0, t1c)
            # (i) 尾部名词修剪 —— "淡淡的香味"(2次) -> "淡淡的"(26次):
            #     短语句法中心是尾部 NOUN, 且同起点存在更短观点短语
            if t1c == h0 and toks[h0].pos in NOUN_POS:
                cut = toks[h0].idx
                for L2 in range(cut - start, 0, -1):
                    if text[start:start + L2] in phrases:
                        hit = text[start:start + L2]
                        t1c = pdoc.char2tok[start + len(hit) - 1]
                        break
            # (ii) 焦点副词前缀修剪 —— "都没有"(1次) -> "没有"(10次):
            #      程度副词(很/太/还...)与否定词保留(很是划算/还不错/不是很喜欢)
            while t1c > t0:
                ft = toks[t0]
                if ft.pos != "ADV" or ft.text in DEGREE_ADV or ft.text in NEGATORS:
                    break
                s2 = ft.idx + len(ft.text)
                p2 = text[s2:start + len(hit)]
                e2c = pdoc.char2tok.get(s2 + len(p2) - 1) if p2 in phrases else None
                if p2 not in phrases or e2c is None:
                    break
                if toks[pdoc.char2tok[s2]].idx != s2:
                    break
                if toks[e2c].idx + len(toks[e2c].text) != s2 + len(p2):
                    break
                start, hit = s2, p2
                t0, t1c = pdoc.char2tok[s2], e2c
            # (iii) 名词前缀修剪 —— "效果好"(1次) -> "好":
            #      句法中心为末尾单/双字谓词, 前缀均为名词, 后缀在观点词典中
            h0 = _span_head(toks, t0, t1c)
            if (h0 == t1c and t0 < h0 and toks[h0].pos in {"VERB", "ADJ"}
                    and all(toks[j].pos in NOUN_POS for j in range(t0, h0))
                    and toks[h0].idx - start <= 4):
                suffix = text[toks[h0].idx:start + len(hit)]
                if suffix in phrases and 0 < len(suffix) <= 2:
                    start = toks[h0].idx
                    hit, t0, t1c = suffix, h0, h0
            # (iv) 尾部语气词频次修剪 —— "很不错的"(3) -> "很不错",
            #      "还可以吧" -> "还可以", "太简陋了" -> 太简陋(按训练频次取舍,
            #      故 "太随便了" 这类金标保留的高频整体形不被拆);
            #      "的" 对重叠式形容词保留(淡淡的/滑滑的), 并允许 的 并入词干 token
            if len(hit) >= 3 and hit[-1] in FINAL_PARTICLES:
                particle = hit[-1]
                stem = hit[:-1]
                fq = freq_map or {}
                dup = particle == "的" and len(stem) >= 2 and stem[-1] == stem[-2]
                if (stem in phrases and not dup
                        and fq.get(stem, 1) >= 3 * fq.get(hit, 1)):
                    e2c = pdoc.char2tok.get(start + len(stem) - 1)
                    if e2c is not None:
                        strict = (toks[e2c].idx + len(toks[e2c].text)
                                  == start + len(stem))
                        inside = (toks[e2c].text.endswith(particle)
                                  and toks[e2c].idx + len(toks[e2c].text) - 1
                                  == start + len(stem))
                        if strict or inside:
                            hit, t1c = stem, e2c
            # (v) 并列短语拆分 —— "便宜实惠"(1次) -> 便宜 + 实惠(各自高频):
            #     块内存在 conj 边界且两段都在词典, 频次占优时把本次匹配截为
            #     前段(后段会在外层扫描到其起点时自然补出)
            elif t1c > t0:
                for j in range(t0, t1c):
                    if toks[j + 1].dep != "conj" and toks[j].dep != "conj":
                        continue
                    cutc = toks[j + 1].idx
                    p1, p2 = text[start:cutc], text[cutc:start + len(hit)]
                    if p1 in phrases and p2 in phrases:
                        fq = freq_map or {}
                        if fq.get(p1, 1) + fq.get(p2, 1) >= 3 * fq.get(hit, 1):
                            hit, t1c = p1, j

        if any(used_chars[k] for k in range(start, start + len(hit))):
            continue
        head = _span_head(toks, t0, t1c)
        spans.append(Span(text=hit, t0=t0, t1=t1c + 1, head=head,
                          kind=kind, src=src,
                          pol=pol_map.get(hit) if pol_map else None))
        for k in range(t0, t1c + 1):
            occupied[k] = True
        for ch in range(start, start + len(hit)):
            used_chars[ch] = True

    # --- 2. 单词 token 补扫(传播得到的新词) ---
    if extra_token_texts:
        for k, tk in enumerate(toks):
            if occupied[k]:
                continue
            if tk.text in extra_token_texts and (pos_filter is None or tk.pos in pos_filter):
                spans.append(Span(text=tk.text, t0=k, t1=k + 1, head=k,
                                  kind=kind, src=src,
                                  pol=pol_map.get(tk.text) if pol_map else None))
                occupied[k] = True
    return spans


def _span_head(toks, t0, t1):
    """取短语的句法中心: 中心指向区间外的第一个 token, 否则取最深层 token。"""
    for k in range(t0, t1 + 1):
        if not (t0 <= toks[k].head <= t1):
            return k
    return t1


# ======================================================================
# 5. Double Propagation 双传播引擎
# ======================================================================
class DoublePropagation:
    """
    全局集合 O / A 在整个目标语料上迭代传播(Qiu 2011 bootstrapping)。
    O: {词: (极性, 来源)}; A: {词: 来源}
    """

    def __init__(self, kn: Knowledge, max_iter=MAX_PROP_ITER, df_prune=DF_PRUNE):
        self.kn = kn
        self.max_iter = max_iter
        self.df_prune = df_prune
        self.trace = []          # 每轮新增统计
        self.O = {}              # text -> (pol, src)
        self.A = {}              # text -> src
        self.df = Counter()      # 名词文档频率(剪枝用)

    # ---------- 工具 ----------
    @staticmethod
    def _is_noun(t):
        return t.pos in NOUN_POS and len(t.text) >= 1 and t.text not in GENERIC_TARGETS

    @staticmethod
    def _negated(toks, k):
        """否定作用域: 状语挂接否定词(含 dep=neg), 或前 2 token 内出现否定词。"""
        tk = toks[k]
        for c in tk.children:
            if toks[c].text in NEGATORS or toks[c].dep == "neg":
                return True
        for j in range(k - 1, max(-1, k - 3), -1):
            if toks[j].text in CLAUSE_PUNCT:
                break
            if toks[j].text in NEGATORS:
                return True
        return False

    @staticmethod
    def _degree(toks, k):
        for c in toks[k].children:
            if toks[c].text in DEGREE_ADV:
                return True
        for j in range(k - 1, max(-1, k - 3), -1):
            if toks[j].text in CLAUSE_PUNCT:
                break
            if toks[j].text in DEGREE_ADV:
                return True
        return False

    @staticmethod
    def _is_opinion_candidate(t, via):
        """传播时的新词观点性判定: ADJ/ADV 直接接受; VERB 排除功能性动词。
        via: 'r2'(A->O) / 'r4'(O->O 并列)"""
        if t.pos == "ADJ":
            return True
        if t.pos == "ADV":
            return via == "r4"       # ADV 仅在并列观点语境吸收
        if t.pos == "VERB":
            return t.text not in FUNCTIONAL_VERBS
        return False

    def _conj_cc(self, toks, k, j):
        """判断 k->j 的并列是顺承还是转折(返回 +1 / -1 / 0)。"""
        tk = toks[k]
        cc_words = set()
        for c in tk.children:
            if toks[c].dep == "cc":
                cc_words.add(toks[c].text)
        # 两词之间出现的连词/标点
        lo, hi = sorted((k, j))
        for x in range(lo + 1, hi):
            if toks[x].dep == "cc" or toks[x].text in COORD_CC | CONTRAST_CC:
                cc_words.add(toks[x].text)
        if cc_words & CONTRAST_CC:
            return -1
        return 1

    # ---------- 主流程 ----------
    def run(self, pdocs):
        self._compute_df(pdocs)
        # 初始化: 种子观点(训练归纳+手工); 方面词以训练词典作高精度种子锚点
        for w, pol in self.kn.seed_opinions.items():
            self.O[w] = (pol, "seed")
        n_seed_o = len(self.O)
        n_seed_a = 0
        for w in self.kn.aspect_vocab:
            if len(w) <= 4:          # 仅单词级种子参与 token 级传播
                self.A[w] = "seed_vocab"
                n_seed_a += 1

        for it in range(1, self.max_iter + 1):
            n_o, n_a = len(self.O), len(self.A)
            src_counter = Counter()
            for pdoc in pdocs.values():
                self._propagate_doc(pdoc, src_counter)
            delta_o, delta_a = len(self.O) - n_o, len(self.A) - n_a
            self.trace.append({
                "iter": it, "|O|": len(self.O), "|A|": len(self.A),
                "new_O": delta_o, "new_A": delta_a,
                "R1_O->A": src_counter["R1"], "R3_A->A": src_counter["R3"],
                "R2_A->O": src_counter["R2"], "R4_O->O": src_counter["R4"],
            })
            if delta_o == 0 and delta_a == 0:
                break
        return n_seed_o, n_seed_a

    def _compute_df(self, pdocs):
        for pdoc in pdocs.values():
            seen = {t.text for t in pdoc.tokens if self._is_noun(t)}
            for w in seen:
                self.df[w] += 1
        self.n_docs = max(len(pdocs), 1)

    def _add_aspect(self, tok, src, via_rel):
        """
        新增方面词。
          - 训练方面词: 直接接受(标注锚点);
          - 句法新词: 文档频率剪枝(Qiu 2011); 主语/定语位置(SUBJ_RELS)直接接受,
            宾语等其他位置要求命中类别关键词(避免 "提亮了肌肤" 类伪方面)。
        """
        w = tok.text
        if w in self.A:
            return False
        if w not in self.kn.aspect_vocab:
            if self.df.get(w, 0) / self.n_docs > self.df_prune:
                return False
            # 功能性动词/评价行为词即使被标为 NOUN 也不作方面(感觉/好评)
            if w in FUNCTIONAL_VERBS or w in ASPECT_STOPWORDS:
                return False
            if via_rel not in SUBJ_RELS and not _keyword_hit(None, [w]):
                return False
        elif w in ASPECT_STOPWORDS:
            return False
        self.A[w] = src
        return True

    def _add_opinion(self, tok, pol, src):
        w = tok.text
        if w in self.O:
            return False
        self.O[w] = (pol, src)
        return True

    def _propagate_doc(self, pdoc, src_counter):
        toks = pdoc.tokens
        for k, tk in enumerate(toks):
            # ---------- R1: O -> A ----------
            if tk.text in self.O:
                for a, rel in self._ot_neighbors(toks, k, side="o"):
                    if self._is_noun(toks[a]) and self._add_aspect(toks[a], "R1", rel):
                        src_counter["R1"] += 1
            # ---------- R2: A -> O ----------
            if tk.text in self.A and not (
                    tk.pos == "VERB" and tk.text in FUNCTIONAL_VERBS):
                for o, rel in self._ot_neighbors(toks, k, side="a"):
                    ot = toks[o]
                    if ot.text in self.O or not self._is_opinion_candidate(ot, "r2"):
                        continue
                    pol = self._infer_opinion_polarity(toks, o)
                    if pol:      # 门控: 新观点词必须可从否定/并列/程度副词判定极性
                        if self._add_opinion(ot, pol, "R2"):
                            src_counter["R2"] += 1
        # ---------- R3: A -> A(并列/共同依存) ----------
        for k, tk in enumerate(toks):
            if not (tk.text in self.A and self._is_noun(tk)):
                continue
            for j, rel in self._tt_neighbors(toks, k):
                if self._is_noun(toks[j]) and self._add_aspect(toks[j], "R3", rel):
                    src_counter["R3"] += 1
        # ---------- R4: O -> O(并列, 传播极性) ----------
        for k, tk in enumerate(toks):
            if tk.text not in self.O:
                continue
            # 功能性动词(用/买/试...)即使被少量标为观点, 也不作为 R4 传播锚点,
            # 避免 "天天用都不心疼" 中 用->心疼 的误传播
            if tk.pos == "VERB" and tk.text in FUNCTIONAL_VERBS:
                continue
            for j in self._oo_neighbors(toks, k):
                ot = toks[j]
                if ot.text in self.O or not self._is_opinion_candidate(ot, "r4"):
                    continue
                sign = self._conj_cc(toks, k, j)
                base_pol = self.O[tk.text][0]
                pol = base_pol if sign == 1 else _flip(base_pol)
                if self._negated(toks, j):
                    pol = _flip(pol)
                if self._add_opinion(ot, pol, "R4"):
                    src_counter["R4"] += 1

    def _infer_opinion_polarity(self, toks, o):
        """
        R2 新观点词极性判定:
          并列已知观点 > 否定(仅 ADJ, 避免"不心疼/没买"类动词误判) > 程度副词(正面)。
        """
        ot = toks[o]
        for c in ot.children:
            if (toks[c].dep == "conj" and toks[c].text in self.O
                    and toks[c].clause == ot.clause):
                # 功能性动词(用/买...)不作极性来源(用都不心疼 -> 心疼)
                if toks[c].pos == "VERB" and toks[c].text in FUNCTIONAL_VERBS:
                    continue
                sign = self._conj_cc(toks, o, c)
                p = self.O[toks[c].text][0]
                p = p if sign == 1 else _flip(p)
                return _flip(p) if self._negated(toks, o) else p
        if self._negated(toks, o) and ot.pos == "ADJ":
            return NEG
        if self._degree(toks, o) and ot.pos in {"ADJ", "VERB"}:
            return POS
        return None

    # ---------- 依存邻居(OT 直接依存: 直接弧 + 共同依存) ----------
    def _ot_neighbors(self, toks, k, side):
        """
        返回 (对侧 token 下标, 关系标签) 集合。
        side='o': k 是观点词, 找名词方面; side='a': 反之。
        """
        out = set()
        tk = toks[k]
        noun_ok = lambda t: self._is_noun(t)
        opin_ok = lambda t: self._is_opinion_candidate(t, "r2")
        ok = noun_ok if side == "o" else opin_ok
        # 注: Token.children 存的是压缩后的 token 下标(int)
        # (a) 直接依存弧 —— 限同子句(解析器常有跨逗号误挂)
        for c in tk.children:
            ct = toks[c]
            if ct.dep in OT_ARC_RELS and ok(ct) and ct.clause == tk.clause:
                out.add((c, ct.dep))
        h = tk.head
        if (h != k and tk.dep in OT_ARC_RELS and ok(toks[h])
                and toks[h].clause == tk.clause):
            out.add((h, tk.dep))
        # (b) 共同依存同一中心(Qiu 定义1: 两者都直接依存 H), 限同子句
        host = h if h != k else k
        for c in toks[host].children:
            if c == k:
                continue
            ct = toks[c]
            if ok(ct) and ct.dep in OT_ARC_RELS and ct.clause == tk.clause:
                out.add((c, ct.dep))
        return out

    def _tt_neighbors(self, toks, k):
        out = set()
        for c in toks[k].children:
            ct = toks[c]
            if ct.dep == "conj" and ct.clause == toks[k].clause:
                out.add((c, "conj"))
        h = toks[k].head
        if (h != k and toks[k].dep == "conj"
                and toks[h].clause == toks[k].clause):
            out.add((h, "conj"))
        # 同一谓词下的多个名词性主/宾语(共同依存)
        host = h if h != k else k
        for c in toks[host].children:
            if c == k:
                continue
            ct = toks[c]
            if self._is_noun(ct) and ct.dep in {"nsubj", "nsubj:pass", "dobj",
                                                "obj", "obl:patient", "attr"}:
                if ct.clause == toks[k].clause:
                    out.add((c, ct.dep))
        return out

    def _oo_neighbors(self, toks, k):
        # 并列仅限同子句: 解析器会把跨多个逗号的动词链误挂为 conj
        # ("活动价很是划算, 买一送一..., 天天用都不心疼" 中 心疼 conj->划算)
        out = set()
        for c in toks[k].children:
            if toks[c].dep == "conj" and toks[c].clause == toks[k].clause:
                out.add(c)
        h = toks[k].head
        if (h != k and toks[k].dep == "conj"
                and toks[h].clause == toks[k].clause):
            out.add(h)
        return out


def _flip(pol):
    return NEG if pol == POS else (POS if pol == NEG else NEU)


# ======================================================================
# 6. 四元组组装(传播完成后的单文档推理)
# ======================================================================
def assemble_quads(pdoc: ParsedDoc, kn: Knowledge, dp: DoublePropagation,
                   rate_min=None, occ_min=None):
    """
    由最终 O/A 集合在单文档上组装 (a, c, o, s) 四元组。
    返回 (quads, debug_stats)
    rate_min/occ_min: 隐式方面发射的观点词"训练被标注率"门控, None 取全局常量。
    """
    if rate_min is None:
        rate_min = IMPLICIT_RATE_MIN
    if occ_min is None:
        occ_min = IMPLICIT_RATE_OCC_MIN
    toks = pdoc.tokens
    stats = Counter()

    # 6.1 观点词 spans: 训练/种子/传播词全部作为短语词典, 单词补扫新传播 ADJ/VERB
    opinion_phrases = set(dp.O) | kn.opinion_vocab
    o_spans = scan_phrases(
        pdoc, opinion_phrases, kind="o", src="lexicon",
        pol_map={w: v[0] for w, v in dp.O.items()},
        freq_map=kn.opinion_freq)
    # 避免嵌套(长词包含短词时保留最长): 按长度去重已在 scan 中处理,
    # 但短语与单词补扫可能重叠, 再过滤一次
    o_spans = _dedup_spans(o_spans, prefer="long")

    # 6.2 方面词 spans: 训练方面词典短语 + 传播得到的名词
    a_spans = scan_phrases(
        pdoc, kn.aspect_vocab, kind="a", src="vocab",
        extra_token_texts={w for w, src in dp.A.items() if src in ("R1", "R3")},
        pos_filter=NOUN_POS)
    a_spans = _dedup_spans(a_spans, prefer="long")
    # 标注偏好最大复合名词块: 分词把 "遮暇功能/客服态度/补水效果" 切成
    # compound:nn 链时, 即使整词不在训练词典也合并成分(金标多取整短语)
    a_spans = _merge_compound_spans(pdoc, a_spans, kn)
    # 泛指名词/品类名词/指人名词/评价行为词/功能性动词 归一或剔除(不产生显式方面)
    a_spans = [s for s in a_spans
               if s.text not in GENERIC_TARGETS
               and not _is_product_term(s.text)
               and s.text not in BODY_TERMS
               and s.text not in PERSON_TERMS
               and s.text not in ASPECT_STOPWORDS
               and s.text not in FUNCTIONAL_VERBS]
    # 观点词噪声(标注残留): 时助词/语气副词 单独成词时丢弃;
    # 品类名词被 R2/R4 误传播为观点时丢弃(训练观点词典中确实存在的除外)
    o_spans = [s for s in o_spans
               if s.text not in OPINION_STOPWORDS
               and not (_is_product_term(s.text)
                        and s.text not in kn.opinion_vocab)]
    # 区间重叠消解(方面/观点同文本或互为子串):
    #   等长或观点被方面包含 -> 丢观点(活动价); 观点包含方面 -> 丢该方面(很补水>补水)
    a_intervals = {(s.t0, s.t1) for s in a_spans}
    drop_a = set()
    kept_o = []
    for os_ in o_spans:
        ov = [(a0, a1) for a0, a1 in a_intervals
              if not (os_.t1 <= a0 or os_.t0 >= a1)]
        if not ov:
            kept_o.append(os_)
            continue
        for a0, a1 in ov:
            if os_.t0 <= a0 and a1 <= os_.t1 and (os_.t1 - os_.t0) > (a1 - a0):
                drop_a.add((a0, a1))      # 观点长于且包含方面: 方面只是其构词成分
            else:
                break                     # 等长/被包含: 丢弃该观点
        else:
            kept_o.append(os_)
    o_spans = kept_o
    a_spans = [s for s in a_spans if (s.t0, s.t1) not in drop_a]

    # 6.3 OT 链接(弧 / 共同依存 / 并列扩展)
    linked_o2a = defaultdict(set)
    conj_graph = _conj_graph(toks)
    for oi, os_ in enumerate(o_spans):
        for ai, as_ in enumerate(a_spans):
            # 同文本同时被标为方面词与观点词(活动价/淡淡的香味)时禁止自配对
            if not (os_.t1 <= as_.t0 or os_.t0 >= as_.t1):
                continue
            r = _link_relation(toks, os_, as_, conj_graph)
            if r:
                linked_o2a[oi].add(ai)
                stats["link_" + r] += 1

    quads = set()
    used_aspects = set()

    def gate_blocked(ospan):
        """训练被标注率门控; 程度副词/否定前缀形式豁免(金标偏好带修饰形式)。"""
        n_occ = kn.o_occ_n.get(ospan.text, 0)
        rate = kn.o_annot_rate.get(ospan.text)
        if rate is None or n_occ < occ_min or rate >= rate_min:
            return False
        first = toks[ospan.t0].text
        if first in DEGREE_ADV:
            return False
        if first in NEGATORS and ospan.text not in NEGATORS \
                and len(ospan.text) > len(first):
            return False
        return True

    # 6.4 每个观点词 -> 方面词配对(无链接走同子句邻近兜底, 再无则隐式方面)
    for oi, os_ in enumerate(o_spans):
        pol = _resolve_polarity(toks, os_, kn, dp)
        linked = sorted(linked_o2a.get(oi, set()))
        if not linked:
            near = _nearest_aspect(toks, o_spans[oi], a_spans)
            if near is not None:
                linked = [near]
                stats["link_gap"] += 1
        cid = min(toks[os_.head].clause, len(pdoc.clauses) - 1)
        clause_toks = [toks[j].text for j in pdoc.clauses[cid]]
        emitted = False
        if linked:
            for ai in linked:
                aspan = a_spans[ai]
                # 弱方面(非训练词典、非复合块) + 稀疏观点: 配对不可信,
                # 不发显式四元组(如 "好多货" -> 货/好多), 观点可回落隐式发射
                weak_a = (aspan.text not in kn.aspect_vocab
                          and getattr(aspan, "src", "") != "compound")
                if weak_a and gate_blocked(os_):
                    stats["weak_link_gate_drop"] += 1
                    continue
                used_aspects.add(ai)
                emitted = True
                cat, csrc = assign_category(kn, aspan.text, os_.text, clause_toks)
                quads.add((aspan.text, cat, os_.text, pol))
                stats["cat_" + csrc] += 1
        if not emitted:
            # 无任何可信显式配对 -> 隐式方面发射;
            # 标注稀疏词门控: 训练中高频出现却很少被标的词丢弃
            # (一直用/用完/多/少/好多/可以…); 程度/否定修饰形式豁免
            if gate_blocked(os_):
                stats["implicit_gate_drop"] += 1
            else:
                stats["implicit_aspect"] += 1
                cat, csrc = assign_category(kn, None, os_.text, clause_toks)
                quads.add((None, cat, os_.text, pol))
                stats["cat_" + csrc] += 1

    # 6.5 隐式观点补产(仅限训练中出现过隐式观点用法的方面词)
    for ai, as_ in enumerate(a_spans):
        if ai in used_aspects:
            continue
        cat = kn.majority(kn.a2cat, as_.text)
        if cat is None:
            hit = _keyword_hit(as_.text, [as_.text])
            if not hit:
                continue
            cat = hit
        pols = kn.implicit_opinion.get((as_.text, cat))
        if pols:
            pol = max(pols.items(), key=lambda kv: kv[1])[0]
            quads.add((as_.text, cat, None, pol))
            stats["implicit_opinion"] += 1
            stats["cat_impO_train"] += 1
            continue
        # 价格类"无观点词"描述: "做活动/有优惠/打折" 金标记为隐式观点(正面)
        if cat == "价格" and as_.text in kn.a2cat:
            acid = min(toks[as_.head].clause, len(pdoc.clauses) - 1)
            clause = [toks[j].text for j in pdoc.clauses[acid]]
            if any(t in PRICE_CUE_TERMS for t in clause):
                quads.add((as_.text, cat, None, POS))
                stats["implicit_opinion"] += 1
                stats["cat_impO_pricecue"] += 1

    return quads, stats


def _dedup_spans(spans, prefer="long"):
    """去除 token 区间重叠的 span, 保留较长者(短语优先于单词)。"""
    spans = sorted(spans, key=lambda s: (s.t0, -(s.t1 - s.t0)))
    kept = []
    for s in spans:
        if any(not (s.t1 <= k.t0 or s.t0 >= k.t1) for k in kept):
            continue
        kept.append(s)
    return kept


def _is_product_term(text):
    """品类名词判定: 训练金标几乎不把品类名本身作方面(很好的面膜 -> 隐式方面)。"""
    return text in PRODUCT_TERMS or (len(text) >= 2 and text.endswith("膜"))


def _merge_compound_spans(pdoc, spans, kn):
    """
    将 compound:nn 链上相邻的方面成分合并为最大名词块(标注偏好):
      "遮暇/功能" -> 遮暇功能, "客服/态度" -> 客服态度, "补水/效果" -> 补水效果
    即使整块未出现在训练词典(如留一文档 id2), 类别仍可由成分关键词判定。
    """
    toks = pdoc.tokens
    n = len(toks)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n - 1):
        a, b = toks[i], toks[i + 1]
        linked = ((b.head == i and b.dep in A_COMPOUND_RELS)
                  or (a.head == i + 1 and a.dep in A_COMPOUND_RELS))
        # "到货/速度"(很快): 解析器把领域词 到货 标为 ADV, 与名词 速度 并列
        # 作同一谓词的附接语/主语; 二者相邻且 到货 在方面词典中时合并
        if not linked and (a.text in kn.aspect_vocab and b.pos in NOUN_POS
                           and a.head == b.head and a.clause == b.clause):
            linked = True
        if linked and a.clause == b.clause:
            parent[find(i)] = find(i + 1)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    span_toks = {j for s in spans for j in range(s.t0, s.t1)}
    # 类别关键词中的非名词成分也允许入块(到货/速度 -> 到货速度)
    kw_words = {w for ws in MANUAL_CATEGORY_KW.values() for w in ws
                if len(w) >= 2}

    new = list(spans)
    for grp in groups.values():
        if len(grp) < 2 or not any(j in span_toks for j in grp):
            continue
        t0, t1 = grp[0], grp[-1] + 1
        if grp != list(range(t0, t1)):
            continue
        # 块内只允许名词/方面词典词/类别关键词(挡住 VERB/ADV 等非名词成分)
        if not all(toks[j].pos in NOUN_POS or toks[j].text in kn.aspect_vocab
                   or toks[j].text in kw_words for j in grp):
            continue
        st, en = toks[t0].idx, toks[grp[-1]].idx + len(toks[grp[-1]].text)
        text = pdoc.text[st:en]
        if text in GENERIC_TARGETS:
            continue
        # 长块(>=3 token)且尾中心为"效果"等: 若能切成两个训练方面词
        # (隔离|保湿效果), 不产整块 —— 前段沿用扫描 span, 此处补后段
        split_done = False
        if len(grp) >= 3 and toks[grp[-1]].text in SPLIT_BLOCK_HEADS:
            for j in range(1, len(grp) - 1):
                pfx = pdoc.text[st:toks[grp[j]].idx]
                sx0 = grp[j]
                sfx = pdoc.text[toks[sx0].idx:en]
                if pfx in kn.aspect_vocab and sfx in kn.aspect_vocab:
                    new.append(Span(
                        text=sfx, t0=sx0, t1=t1,
                        head=_span_head(toks, sx0, t1 - 1),
                        kind="a", src="compound"))
                    split_done = True
                    break
        if split_done:
            continue
        head = _span_head(toks, t0, t1 - 1)
        new.append(Span(text=text, t0=t0, t1=t1, head=head,
                        kind="a", src="compound"))
    return _dedup_spans(new, prefer="long")


def _conj_graph(toks):
    g = defaultdict(set)
    for k, t in enumerate(toks):
        if t.dep == "conj":
            g[k].add(t.head)
            g[t.head].add(k)
    return g


def _eff_a_head(toks, h):
    """方面短语中心词沿名词性复合修饰向外跟随(活动->价, 客服->态度)。"""
    for _ in range(2):
        t = toks[h]
        if t.dep in A_COMPOUND_RELS and t.head != h and toks[t.head].pos in NOUN_POS:
            h = t.head
        else:
            break
    return h


def _link_relation(toks, os_, as_, conj_graph):
    """判断 o span 与 a span 是否构成 OT 链接, 返回链接类型(None 表示无)。"""
    oh = os_.head
    ah = _eff_a_head(toks, as_.head)
    th, ta = toks[oh], toks[ah]
    same_clause = th.clause == ta.clause

    # 1) 直接依存弧: 严格限同子句(跨子句 conj 实测多为解析器误挂,
    #    如 "很好的隔离霜,感觉还不错" 中 不错 conj->隔离霜)
    if same_clause and th.head == ah and th.dep in OT_ARC_RELS:
        return "arc"
    if same_clause and ta.head == oh and ta.dep in OT_ARC_RELS:
        return "arc"
    # 方面经弱关系(dep 等)直接挂到观点: 同子句按 2-hop 同类处理
    if same_clause and ta.head == oh and ta.dep in BRIDGE_DEPS:
        return "arc2"

    # 2) 共同依存同一中心(Qiu 定义1), 限同子句;
    #    任一方为并列谓词时拒绝(各并列谓词有自己的主宾语, 见 id79)
    if (same_clause and th.head == ta.head and oh != ah
            and th.dep != "conj" and ta.dep != "conj"):
        if ta.dep in {"nsubj", "nsubj:pass", "dobj", "obj", "obl:patient",
                      "attr", "amod"}:
            return "sibling"

    # 3) 2-hop 中介: a -> 桥接谓词 -> o (如 补水 -> 保湿(dep) -> 好)
    if same_clause and th.pos in {"VERB", "ADJ", "ADV"}:
        b = toks[ta.head]
        if b.i not in (oh, ah) and b.dep in BRIDGE_DEPS and b.head == oh:
            return "arc2"
        b2 = toks[th.head]
        if b2.i not in (oh, ah) and b2.dep in BRIDGE_DEPS and b2.head == ah \
                and ta.pos in NOUN_POS:
            return "arc2"

    # 4) 并列扩展 1 跳(a-a / 谓词 conj)
    if ah in conj_graph.get(oh, set()) or oh in conj_graph.get(ah, set()):
        return "conj"
    for x in conj_graph.get(ah, set()):
        tx = toks[x]
        if th.head == tx.head and th.dep in OT_ARC_RELS | {"conj"}:
            return "conj"
        if tx.head == oh and tx.dep in OT_ARC_RELS:
            return "conj"
    return None


def _nearest_aspect(toks, os_, a_spans):
    """同子句内 token 距离最近的方面词(GAP_WINDOW 内), 解析失败时的高精度兜底。"""
    best, best_gap = None, GAP_WINDOW + 1
    for ai, a in enumerate(a_spans):
        if toks[a.head].clause != toks[os_.head].clause:
            continue
        if not (a.t1 <= os_.t0 or a.t0 >= os_.t1):
            continue                    # 区间重叠(同词)不兜底配对
        if a.t1 <= os_.t0:
            gap = os_.t0 - a.t1
        elif a.t0 >= os_.t1:
            gap = a.t0 - os_.t1
        else:
            gap = 0
        if gap <= GAP_WINDOW and gap < best_gap:
            best, best_gap = ai, gap
    return best


def _resolve_polarity(toks, os_, kn, dp):
    """观点极性: 词典/传播极性为基础, 否定作用域翻转(短语内含否定不重复翻)。"""
    if os_.pol in (POS, NEU, NEG):
        pol = os_.pol
    else:
        pol = kn.majority(kn.opinion_pol, os_.text, default=POS)
    # 短语本身已含否定词(不好用/没有), 词典极性已含否定语义, 不再翻转
    if any(n in os_.text for n in ("不", "没", "无", "未", "别", "非")):
        return pol
    if pol != NEU and DoublePropagation._negated(toks, os_.head):
        return _flip(pol)
    return pol


# ======================================================================
# 7. 语料级推理
# ======================================================================
def predict_corpus(rows, kn: Knowledge, parser: ZhParser, dp_kwargs=None,
                   assemble_kwargs=None):
    """对一批 (id, raw_text) 执行: 解析 -> 双传播 -> 四元组组装。"""
    t0 = time.time()
    pdocs = parser.parse_many(rows)
    dp = DoublePropagation(kn, **(dp_kwargs or {}))
    n_seed_o, n_seed_a = dp.run(pdocs)
    results, agg_stats = {}, Counter()
    for rid, pdoc in pdocs.items():
        quads, st = assemble_quads(pdoc, kn, dp, **(assemble_kwargs or {}))
        results[rid] = quads
        agg_stats.update(st)
    elapsed = time.time() - t0
    meta = {
        "n_docs": len(pdocs), "n_seed_o": n_seed_o, "n_seed_a": n_seed_a,
        "final_O": len(dp.O), "final_A": len(dp.A),
        "trace": dp.trace, "elapsed": elapsed, "stats": agg_stats,
    }
    return results, meta


# ======================================================================
# 8. 评估
# ======================================================================
def gold_from_labels(labels: pd.DataFrame, ids):
    gold = defaultdict(set)
    sub = labels[labels["id"].isin(ids)]
    for rid, a, o, c, s in sub[["id", "AspectTerms", "OpinionTerms",
                                "Categories", "Polarities"]].itertuples(index=False):
        gold[int(rid)].add((a, c, o, s))
    return gold


def evaluate(gold, pred):
    """exact-quad P/R/F1 + (a,o) 对/方面词/观点词 F1 + 类别/极性准确率。"""
    tp = fp = fn = 0
    pair_g, pair_p = set(), set()
    a_g, a_p, o_g, o_p = set(), set(), set(), set()
    cat_hit = cat_tot = pol_hit = pol_tot = 0
    ids = set(gold) | set(pred)
    for rid in ids:
        g, p = gold.get(rid, set()), pred.get(rid, set())
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        for (a, c, o, s) in g:
            pair_g.add((rid, a, o)); a_g.add((rid, a)); o_g.add((rid, o))
        for (a, c, o, s) in p:
            pair_p.add((rid, a, o)); a_p.add((rid, a)); o_p.add((rid, o))
        gm = {(a, o): (c, s) for a, c, o, s in g}
        for a, c, o, s in p:
            if (a, o) in gm:
                cat_tot += 1; pol_tot += 1
                cat_hit += int(c == gm[(a, o)][0])
                pol_hit += int(s == gm[(a, o)][1])

    def prf(n_tp, n_p, n_g):
        P = n_tp / n_p if n_p else 0.0
        R = n_tp / n_g if n_g else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        return P, R, F

    return {
        "quad": prf(tp, tp + fp, tp + fn),
        "pair": prf(len(pair_g & pair_p), len(pair_p), len(pair_g)),
        "aspect": prf(len(a_g & a_p), len(a_p), len(a_g)),
        "opinion": prf(len(o_g & o_p), len(o_p), len(o_g)),
        "cat_acc": (cat_hit / cat_tot if cat_tot else 0.0, cat_tot),
        "pol_acc": (pol_hit / pol_tot if pol_tot else 0.0, pol_tot),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ======================================================================
# 9. 主流程
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Double-Propagation-ACOS 规则基线")
    ap.add_argument("--skip-test", action="store_true", help="只做 dev 评估")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--dev-ratio", type=float, default=DEV_RATIO)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report, lines = [], []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    t_start = time.time()
    log("=" * 74)
    log("Double-Propagation-ACOS 规则基线 —— 化妆品评论 ACOS 四元组抽取")
    log("=" * 74)

    # ---------- 数据 ----------
    reviews_tr, labels, reviews_te = load_raw_data()
    rev_map_tr = dict(zip(reviews_tr["id"].astype(int), reviews_tr["Reviews"]))
    rev_map_te = dict(zip(reviews_te["id"].astype(int), reviews_te["Reviews"]))
    all_ids = np.array(sorted(rev_map_tr))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_ids)
    n_dev = int(round(len(all_ids) * args.dev_ratio))
    dev_ids = set(all_ids[:n_dev].tolist())
    fit_ids = set(all_ids[n_dev:].tolist())
    log("[数据] 训练评论 %d 条 / 标签四元组 %d 行 / 测试评论 %d 条"
        % (len(rev_map_tr), len(labels), len(rev_map_te)))
    log("[数据] dev 留出 %d 条(seed=%d), 规则归纳用 %d 条"
        % (len(dev_ids), args.seed, len(fit_ids)))
    log("[超参] max_prop_iter=%d, dev_ratio=%.2f, seed=%d, gap_window=%d, "
        "df_prune=%.2f, max_phrase_len=%d, implicit_rate_min=%.2f, "
        "implicit_rate_occ_min=%d"
        % (MAX_PROP_ITER, args.dev_ratio, args.seed, GAP_WINDOW, DF_PRUNE,
           MAX_PHRASE_LEN, IMPLICIT_RATE_MIN, IMPLICIT_RATE_OCC_MIN))

    # ---------- 知识库(规则"训练") ----------
    fit_labels = labels[labels["id"].isin(fit_ids)]
    kn_fit = build_knowledge(fit_labels)
    compute_annot_rates(kn_fit, fit_labels,
                        ((i, rev_map_tr[i]) for i in fit_ids))
    n_gated = sum(1 for o in kn_fit.opinion_vocab
                  if kn_fit.o_occ_n.get(o, 0) >= IMPLICIT_RATE_OCC_MIN
                  and kn_fit.o_annot_rate.get(o, 1.0) < IMPLICIT_RATE_MIN)
    log("[知识] 观点词 %d(种子, 含手工补充 %d), 方面词 %d, 隐式观点(a,c)规则 %d 条"
        % (len(kn_fit.seed_opinions), getattr(kn_fit, "_n_manual_seed", 0),
           len(kn_fit.aspect_vocab), len(kn_fit.implicit_opinion)))
    log("[知识] 隐式观点门控: 标注率<%.2f 且出现>=%d 文档的观点词 %d 个"
        % (IMPLICIT_RATE_MIN, IMPLICIT_RATE_OCC_MIN, n_gated))

    # ---------- 解析器 ----------
    vocab_words = kn_fit.aspect_vocab | kn_fit.opinion_vocab | {
        w for ws in MANUAL_SEED.values() for w in ws}
    parser = ZhParser(vocab_words=vocab_words)

    # ---------- dev 评估 ----------
    log("-" * 74)
    log("[阶段1] dev 集: 双传播抽取 + 评估")
    dev_rows = [(i, rev_map_tr[i]) for i in sorted(dev_ids)]
    pred_dev, meta_dev = predict_corpus(dev_rows, kn_fit, parser)
    gold_dev = gold_from_labels(labels, dev_ids)
    m = evaluate(gold_dev, pred_dev)
    _log_run_meta(log, meta_dev)
    _log_metrics(log, m)

    # dev 预测明细落盘(便于误差分析)
    _dump_predictions(OUT_DIR / "dev_predictions.csv", pred_dev, gold_dev, rev_map_tr)

    if args.skip_test:
        _save_report(lines)
        return

    # ---------- 全量训练 ----------
    log("-" * 74)
    log("[阶段2] 全量训练集归纳规则知识")
    kn_full = build_knowledge(labels)
    compute_annot_rates(kn_full, labels,
                        zip(reviews_tr["id"].astype(int), reviews_tr["Reviews"]))
    log("[知识] 观点词 %d(含手工补充 %d), 方面词 %d, 隐式观点规则 %d 条"
        % (len(kn_full.seed_opinions), getattr(kn_full, "_n_manual_seed", 0),
           len(kn_full.aspect_vocab), len(kn_full.implicit_opinion)))
    _dump_lexicons(kn_full)

    # ---------- 测试推理 ----------
    log("-" * 74)
    log("[阶段3] 测试集: 双传播抽取四元组")
    vocab_words = kn_full.aspect_vocab | kn_full.opinion_vocab | {
        w for ws in MANUAL_SEED.values() for w in ws}
    parser_full = ZhParser(vocab_words=vocab_words)
    test_rows = [(i, rev_map_te[i]) for i in sorted(rev_map_te)]
    pred_te, meta_te = predict_corpus(test_rows, kn_full, parser_full)
    _log_run_meta(log, meta_te)

    # 分布统计
    cat_cnt, pol_cnt, n_imp_a, n_imp_o, n_quads = Counter(), Counter(), 0, 0, 0
    for qs in pred_te.values():
        for a, c, o, s in qs:
            n_quads += 1
            cat_cnt[c] += 1
            pol_cnt[s] += 1
            n_imp_a += int(a is None)
            n_imp_o += int(o is None)
    log("[测试] 预测四元组 %d 个, 平均 %.2f 个/评论; 隐式方面 %d (%.1f%%), 隐式观点 %d (%.1f%%)"
        % (n_quads, n_quads / len(rev_map_te), n_imp_a, 100 * n_imp_a / max(n_quads, 1),
           n_imp_o, 100 * n_imp_o / max(n_quads, 1)))
    log("[测试] 类别分布: %s" % dict(cat_cnt.most_common()))
    log("[测试] 极性分布: %s" % dict(pol_cnt.most_common()))

    result_path = _write_result_csv(TEST_REVIEWS_PATH, pred_te)
    log("[输出] 提交文件(无 BOM UTF-8/无表头): %s" % result_path)
    log("[总耗时] %.1f 秒" % (time.time() - t_start))
    _save_report(lines)


def _log_run_meta(log, meta):
    log("[传播] 语料 %d 条; 初始种子 |O|=%d |A|=%d -> 收敛 |O|=%d |A|=%d; 耗时 %.1fs"
        % (meta["n_docs"], meta["n_seed_o"], meta["n_seed_a"],
           meta["final_O"], meta["final_A"], meta["elapsed"]))
    log("[传播] 迭代轨迹(iter |O| |A| new_O new_A | R1 R2 R3 R4):")
    for r in meta["trace"]:
        log("         #%d  O=%-4d A=%-4d  +O=%-3d +A=%-3d | R1=%-3d R2=%-3d R3=%-3d R4=%-3d"
            % (r["iter"], r["|O|"], r["|A|"], r["new_O"], r["new_A"],
               r["R1_O->A"], r["R2_A->O"], r["R3_A->A"], r["R4_O->O"]))
    st = meta["stats"]
    log("[组装] 链接来源: 弧=%d 共同依存=%d 并列=%d 邻近兜底=%d; "
        "隐式方面=%d 隐式观点=%d"
        % (st["link_arc"], st["link_sibling"], st["link_conj"], st["link_gap"],
           st["implicit_aspect"], st["implicit_opinion"]))
    log("[组装] 类别规则命中: %s"
        % {k.replace("cat_", ""): v for k, v in st.items() if k.startswith("cat_")})


def _log_metrics(log, m):
    P, R, F = m["quad"]
    log("[评估] >>> Exact-Quad  P=%.4f  R=%.4f  F1=%.4f  (TP=%d FP=%d FN=%d)"
        % (P, R, F, m["tp"], m["fp"], m["fn"]))
    for name, key in (("(a,o)对", "pair"), ("方面词", "aspect"), ("观点词", "opinion")):
        p, r, f = m[key]
        log("[评估]     %-8s P=%.4f R=%.4f F1=%.4f" % (name, p, r, f))
    log("[评估]     (a,o)命中时类别准确率=%.4f (n=%d), 极性准确率=%.4f (n=%d)"
        % (m["cat_acc"][0], m["cat_acc"][1], m["pol_acc"][0], m["pol_acc"][1]))


def _term(t):
    return "_" if t is None else t


def _write_result_csv(test_path, pred):
    """按官方格式写出: 无表头/无 BOM UTF-8; 每个测试 id 必出现, 空结果行全 '_'。"""
    te = pd.read_csv(test_path, encoding="utf-8")
    ids = te["id"].astype(int).tolist()
    rows = []
    for rid in ids:                      # 保持测试文件 id 顺序
        qs = sorted(pred.get(rid, set()), key=lambda q: (q[1] or "", q[2] or ""))
        if not qs:
            rows.append((rid, "_", "_", "_", "_"))
        for a, c, o, s in qs:
            rows.append((rid, _term(a), _term(o), c, s))
    out = pd.DataFrame(rows, columns=["id", "AspectTerms", "OpinionTerms",
                                      "Categories", "Polarities"])
    path = OUT_DIR / "Result.csv"
    out.to_csv(path, index=False, header=False, encoding="utf-8")
    # 校验 id 覆盖
    out_ids = set(out["id"])
    assert out_ids == set(ids), "结果 id 与测试集不一致"
    return path


def _dump_predictions(path, pred, gold, rev_map):
    rows = []
    qsort = lambda qs: sorted(qs, key=lambda q: tuple("" if x is None else x for x in q))
    for rid in sorted(set(pred) | set(gold)):
        g = gold.get(rid, set())
        for a, c, o, s in qsort(pred.get(rid, set())):
            rows.append({"id": rid, "review": rev_map.get(rid, ""),
                         "AspectTerms": _term(a), "Categories": c,
                         "OpinionTerms": _term(o), "Polarities": s,
                         "hit": int((a, c, o, s) in g)})
        for q in qsort(g - pred.get(rid, set())):
            a, c, o, s = q
            rows.append({"id": rid, "review": rev_map.get(rid, ""),
                         "AspectTerms": _term(a), "Categories": c,
                         "OpinionTerms": _term(o), "Polarities": s, "hit": 0})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _dump_lexicons(kn):
    op_rows = [{"term": o, "polarity": kn.majority(kn.opinion_pol, o),
                "pos_n": kn.opinion_pol[o].get(POS, 0),
                "neu_n": kn.opinion_pol[o].get(NEU, 0),
                "neg_n": kn.opinion_pol[o].get(NEG, 0),
                "top_category": kn.majority(kn.o2cat_all, o)}
               for o in sorted(kn.opinion_vocab)]
    pd.DataFrame(op_rows).to_csv(OUT_DIR / "lexicon_opinion.csv",
                                 index=False, encoding="utf-8-sig")
    a_rows = [{"term": a, "category": kn.majority(kn.a2cat, a),
               "counts": dict(kn.a2cat[a]),
               "implicit_opinion_rule": int((a, kn.majority(kn.a2cat, a))
                                            in kn.implicit_opinion)}
              for a in sorted(kn.aspect_vocab)]
    pd.DataFrame(a_rows).to_csv(OUT_DIR / "lexicon_aspect.csv",
                                index=False, encoding="utf-8-sig")


def _save_report(lines):
    path = BASELINE_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print("[输出] 运行日志: %s" % path)


if __name__ == "__main__":
    main()
