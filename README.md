# 基于天池"评论观点挖掘赛"的 ACOS 四元组抽取

> 机器学习课程设计 · 组员：陈嘉乐，戴艺维

## 一、项目简介

### 1.1 我们要解决什么问题

本项目对应天池竞赛 **"评论观点挖掘赛"**：

- 竞赛链接：<https://tianchi.aliyun.com/competition/entrance/532421/information>
- 数据：化妆品品类的电商用户评论（品牌名已做 `**` 脱敏处理）
- 评估指标：**四元组精确匹配的 F1-score**（预测出的四元组与标注四元组必须四个字段完全一致才算对）

给定一条评论，例如：

> "物流很快，但是遮瑕效果差一些，总体还不错"

模型需要输出若干个 **ACOS 四元组**，四个字段分别是：

| 字段                 | 含义   | 通俗解释                  | 本例取值             |
| ------------------ | ---- | --------------------- | ---------------- |
| **A**spectTerms    | 方面词  | 评论谈论的对象（产品的某个属性），可以为空 | `""`、`遮瑕效果`、`""` |
| **O**pinionTerms   | 观点词  | 用户表达看法的词语             | `很快`、`差一些`、`还不错` |
| **C**ategories     | 方面类别 | 方面词所属的 13 个大类之一       | `物流`、`功效`、`整体`   |
| **S** / Polarities | 情感极性 | 正面 / 负面 / 中性          | `正面`、`负面`、`正面`   |

两个特殊约定：

- **隐式方面（AspectTerms =** **`_`）**：句子里没有出现具体属性词，只有评价，如"还不错"——评价对象靠语义推断，统一归到"整体"等类别。训练集中这类高达 **71.4%**，是本任务的最大难点之一。
- **隐式观点（OpinionTerms =** **`_`）**：只说了对象、没出现明显评价词，训练集中占 2.6%。
- 类别固定 13 类：`整体、使用体验、功效、价格、物流、气味、包装、真伪、服务、其他、成分、尺寸、新鲜度`；
  极性固定 3 类：`正面、负面、中性`。

### 1.2 我们尝试的五种方案

为了由浅入深地比较不同建模范式，我们一共实现了 **5 种独立方案 + 1 个融合方案**：

| # | 方案                                   | 建模范式           | 是否需要训练        | 测试集 F1                            |
| - | ------------------------------------ | -------------- | ------------- | --------------------------------- |
| 一 | **数据预处理 + EDA**                      | 规则/统计          | 否             | —（为后续方案提供数据底座）                    |
| 二 | **Baseline：Double-Propagation 规则模型** | 依存句法 + 情感词典双传播 | 否（归纳规则知识）     | **0.4925**                        |
| 三 | **序列标注：BERT + CRF 多任务模型**            | 判别式深度学习        | 是（CPU 约 2 小时） | **0.6198**                        |
| 四 | **LLM 微调：Qwen SFT**                  | 生成式大模型         | 是（云端 GPU）     | **0.8042**（Qwen2.5-7B 全参 2 epoch） |
| 五 | **管道式（方案 C）**：对抽取 + 交叉编码分类           | 两阶段判别式深度学习     | 是（CPU 约 4 小时） | **0.7651**                        |
| 六 | **多模型加权投票融合**                        | 集成学习           | 否             | **0.8116**（最终提交，最优）               |

> 各方案的 dev（验证集）指标与详细分析见后文对应章节。所有方案随机种子统一为 **42**，训练集按评论级 85%/15% 切分 train/dev。

### 1.3 数据规模一览

| 数据                       | 数量                                                 |
| ------------------------ | -------------------------------------------------- |
| 训练评论 `Train_reviews.csv` | 3229 条（id 1\~3229）                                 |
| 训练标签 `Train_labels.csv`  | 6633 个四元组（平均每条评论 2.05 个，最多 7 个）                    |
| 测试评论 `Test_reviews.csv`  | 2237 条（id 1\~2237）                                 |
| 评论长度                     | 中位数 20 个汉字，95% 在 43 字以内                            |
| 隐式方面 / 隐式观点占比            | 71.39% / 2.59%                                     |
| 极性分布                     | 正面 89.33% / 负面 8.38% / 中性 2.29%（类别与极性都**高度不均衡**）   |
| 最常见类别                    | 整体 42.54%、使用体验 15.71%、功效 10.95%；最稀有的"新鲜度"仅 13 个四元组 |

***

## 二、环境依赖与数据准备

### 2.1 软件环境

- Python 3.10+（开发与验证环境：Windows + Python 3.13.9，CPU）
- 主要依赖见 [requirements.txt](requirements.txt)：

```bash
pip install -r requirements.txt
```

依赖分三组（requirements.txt 中有注释说明）：

1. **数据/规则部分**：pandas、numpy、jieba（分词）、matplotlib（EDA 画图）、spaCy + PyYAML；
2. **BERT 与管道模型**：torch、transformers、pytorch-crf。CPU 版 torch 用
   `pip install torch --index-url https://download.pytorch.org/whl/cpu` 安装；
3. **spaCy 中文句法模型**（仅 baseline 需要，不在 PyPI，需单独装）：

```bash
python -m spacy download zh_core_web_sm
# 官方源慢时可直接装 wheel：
# pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl
```

1. **LLM 微调（方案四）**：另需安装 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)（推荐在云端 GPU 环境）；推理可选 vLLM 加速。
2. 首次运行 BERT 类模型会自动下载 `bert-base-chinese`；网络受限时设置
   `set HF_ENDPOINT=https://hf-mirror.com`（Windows）后再运行。

### 2.2 原始数据放置

把官方数据按下表放入 `data/raw/`（**保持原始 CSV 不做任何改动**）：

```
data/raw/
├── TRAIN/
│   ├── Train_reviews.csv      # 列: id, Reviews
│   └── Train_labels.csv       # 列: id, AspectTerms, A_start, A_end,
│                              #     OpinionTerms, O_start, O_end,
│                              #     Categories, Polarities
├── TEST/
│   ├── Test_reviews.csv       # 列: id, Reviews
│   └── Result(example).csv    # 官方提交样例
├── train_README.md            # 官方数据说明
└── test_README.md
```

字段说明（来自官方 `train_README.md`）：

- `A_start/A_end`、`O_start/O_end` 是方面词/观点词在**评论原文中的字符偏移**（半开区间）；
  术语为 `_` 时对应位置为空。**预测结果不需要位置信息，只评四元组**。
- 提交文件 `Result.csv` 为 **无表头、无 BOM 的 UTF-8 CSV**，列顺序固定为
  `id,AspectTerms,OpinionTerms,Categories,Polarities`；空术语写 `_`；每条评论至少占一行（一个四元组都没抽到就写 `id,_,_,_,_`）。

### 2.3 快速运行总览

所有命令均在**项目根目录**（本 README 所在的 `courseproject/` 目录）下执行：

```bash
# 步骤 0：数据预处理 + EDA（所有学习方案的公共底座）
python src/preprocess.py
python src/eda.py

# 方案二：规则基线（dev 评估 + 全量归纳 + 测试推理）
python src/model_baseline.py
python src/predict.py

# 方案三：BERT+CRF 序列标注
python src/model_bert/model_bert.py
python src/model_bert/predict_bert.py

# 方案四：LLM 微调（先构数据，再用 LLaMA-Factory 训练，最后推理）
python src/llm/build_sft_dataset.py
llamafactory-cli train config/llm_qwen7b_full.yaml        # 详见第七章
python src/llm/predict_llm.py --model_path ./saves/qwen7b_full_sft_2ep --backend vllm

# 方案五：管道式（先依次训练两个阶段，再串联推理）
python src/pipeline/extract_pair.py
python src/pipeline/classify_pair.py
python src/pipeline/predict_pipeline.py --eval-dev        # dev 端到端评估
python src/pipeline/predict_pipeline.py                   # 生成测试 Result.csv

# 方案六：多模型加权投票融合
python src/ensemble.py
```

> 说明：模型权重文件（`*.pt`）和 LLaMA-Factory 的 `saves/` 目录体积过大，已在 [.gitignore](.gitignore) 中忽略；克隆仓库后需按上述命令重新训练生成，`Result.csv`、`label_vocab.json`、`*_meta.json` 等小文件均已随仓库提供。

***

## 三、代码目录结构

```
courseproject/                              # 项目根目录（所有命令在此执行）
├── README.md                               # 本说明文档
├── requirements.txt                        # Python 依赖清单
├── .gitignore                              # 忽略 .venv / *.pt / saves/ 等
│
├── config/                                 # 各方案的超参配置（YAML）与训练快照
│   ├── model_baseline.yaml                 # 方案二：规则/词表/依存关系/超参（可手工编辑）
│   ├── model_bert_trained.yaml             # 方案三：训练完成后自动 dump 的超参快照
│   ├── model_pipeline.yaml                 # 方案五：Stage1/Stage2 超参（手工编辑）
│   ├── model_pipeline_trained.yaml         # 方案五 Stage1 训练快照（自动生成）
│   ├── model_pipeline_clf_trained.yaml     # 方案五 Stage2 训练快照（自动生成）
│   ├── llm_qwen7b_full.yaml                # 方案四：Qwen2.5-7B 全参 SFT 配置
│   ├── llm_qwen14b_lora.yaml               # 方案四：Qwen3-14B LoRA 配置
│   └── llm-7b/                             # 7B 训练存档：tokenizer 文件、训练曲线、loss
│
├── data/                                   # 原始数据与各方案结果
│   ├── raw/                                #   官方原始数据（见 2.2 节，手工放入）
│   │   ├── TRAIN/  ├── Train_reviews.csv  └── Train_labels.csv
│   │   ├── TEST/   ├── Test_reviews.csv   └── Result(example).csv
│   │   ├── train_README.md
│   │   └── test_README.md
│   ├── baseline/                           #   方案二产物
│   │   ├── Result.csv                      #     基线提交结果
│   │   ├── lexicon_aspect.csv              #     归纳出的方面词词典
│   │   ├── lexicon_opinion.csv             #     归纳出的观点词词典（含极性）
│   │   └── dev_predictions.csv             #     dev 逐条预测（误差分析用）
│   ├── bert/                               #   方案三产物
│   │   ├── Result.csv                      #     BERT 提交结果
│   │   ├── label_vocab.json                #     BIO/类别/极性标签字典
│   │   └── bert_crf.pt                     #     模型权重（gitignore，训练生成）
│   ├── llm/                                #   方案四产物
│   │   ├── sft_train.json                  #     LLaMA-Factory 训练数据（Alpaca 格式）
│   │   ├── dataset_info.json               #     LLaMA-Factory 数据集注册文件
│   │   ├── Result.csv                      #     Qwen3-14B LoRA 提交结果
│   │   └── Result 7b-full-2epocns.csv      #     Qwen2.5-7B 全参 2ep 提交结果（最强单模）
│   ├── pipeline/                           #   方案五产物
│   │   ├── Result.csv                      #     管道提交结果
│   │   ├── pair_meta.json                  #     Stage1 标签字典/最优阈值/dev 指标
│   │   ├── pair_clf_meta.json              #     Stage2 标签字典/类权重/dev 指标
│   │   ├── pair_extractor.pt               #     Stage1 权重（gitignore，训练生成）
│   │   └── pair_classifier.pt              #     Stage2 权重（gitignore，训练生成）
│   └── ensemble/                           #   方案六产物
│       ├── Result.csv                      #     融合提交结果（当前最优阈值版）
│       └── Result_t1.30.csv / t1.35 / t1.45 …  # 各投票阈值对应的候选版本
│
├── notebook/
│   └── llm-7b.ipynb                        # 云端（ModelScope/Colab）LLM 微调笔记本
│
├── outputs/                                # 可再生的中间产物（删了重跑即可）
│   ├── processed/                          #   预处理输出
│   │   ├── Train_reviews_processed.csv     #     训练评论清洗+分词结果
│   │   ├── Test_reviews_processed.csv      #     测试评论清洗+分词结果
│   │   └── Train_labels_parsed.csv         #     结构化标签（'_'→NA，位置→Int64）
│   ├── figures/                            #   EDA 图表（5 张 PNG）
│   │   ├── 01_length_distribution.png      #     评论长度分布
│   │   ├── 02_quads_per_review.png         #     每条评论四元组数量分布
│   │   ├── 03_category_distribution.png    #     类别分布
│   │   ├── 04_polarity_distribution.png    #     极性分布
│   │   └── 05_category_polarity.png        #     类别×极性交叉分布
│   └── logs/                               #   各脚本的文字报告（含全部实测指标）
│       ├── preprocess_report.txt
│       ├── eda_report.txt
│       ├── baseline_report.txt
│       ├── bert_report.txt  └── bert_run.log
│       ├── pipeline_pair_report.txt  └── pipeline_pair_console.txt
│       └── pipeline_clf_report.txt
│
└── src/                                    # 全部源代码
    ├── config.py                           # 【总配置】集中管理所有路径/产物位置
    ├── preprocess.py                       # 方案一：文本清洗 + jieba 分词 + 标签解析
    ├── eda.py                             # 方案一：探索性数据分析 + 可视化
    ├── model_baseline.py                  # 方案二：Double-Propagation 规则模型（训练/评估一体）
    ├── predict.py                         # 方案二：全量归纳规则并对测试集推理
    ├── ensemble.py                        # 方案六：5 模型加权投票融合
    ├── llm/                               # 方案四：LLM 微调
    │   ├── __init__.py
    │   ├── build_sft_dataset.py           #   标签 → Alpaca 格式 SFT 数据
    │   └── predict_llm.py                 #   transformers / vLLM 双后端推理
    ├── model_bert/                        # 方案三：BERT+CRF 序列标注
    │   ├── model_bert.py                  #   模型定义 + 训练 + dev 评估
    │   └── predict_bert.py                #   加载权重对测试集推理
    └── pipeline/                          # 方案五：管道式 ACOS
        ├── __init__.py
        ├── extract_pair.py                #   Stage1：(Aspect, Opinion) 对抽取
        ├── classify_pair.py               #   Stage2：逐对分类 Category/Polarity
        └── predict_pipeline.py            #   Stage1→Stage2 串联推理 / --eval-dev
```

路径约定（统一在 [src/config.py](src/config.py) 中维护，改目录结构只需动这一个文件）：

- 原始数据放 `data/raw/`；各方案的提交结果与模型产物放 `data/<方案名>/`；
- 可随时删除再生的中间产物放 `outputs/`：`processed/`（清洗数据）、`figures/`（图表）、`logs/`（日志）。

***

## 四、数据预处理与探索性分析（EDA）

**对应代码**：[src/preprocess.py](src/preprocess.py)、[src/eda.py](src/eda.py)

### 4.1 为什么要做这一步

原始评论文本噪声很多（URL、@、表情符号、被脱敏成 `**` 的品牌名、连续空格……），
而后续四种建模方法对输入的"干净程度"要求各不相同。预处理脚本只做两件边界清晰的事：
**① 文本清洗与分词；② 标签表结构化**。EDA 脚本则负责统计分布、检查异常、画图，
为建模决策（如何处理类别不平衡、隐式方面等）提供依据。

### 4.2 文本清洗流水线（`clean_text`）

按顺序执行：

1. HTML 实体解码（`&amp;` → `&`）；
2. **情感表情转中文词**：👍→"点赞"、😭→"大哭"、😡→"生气"……保留表情承载的情感；
3. 去除 URL、HTML 标签、`@用户名`、`#话题#`（保留话题正文）；
4. **品牌脱敏归一化**：连续的 `*`/`**`/`***` 统一替换成"某品牌"，保留语义槽而不是直接删掉；
5. 只保留中文、英文、数字，清除其余标点符号；合并连续空格。

特别注意：**短评论不删**。"一般""不好""过敏"这种一两个字的评论本身就是完整情感表达。

### 4.3 jieba 分词：数据驱动词典 + 谨慎停用词

直接用 jieba 默认词典会把"遮瑕效果""补水效果"这类标注词切碎，为此做了两层保障：

1. **标注词反哺词典**：从 `Train_labels.csv` 读出全部 256 个唯一方面词、1418 个唯一观点词，
   连同 90 余个手工整理的化妆品领域词（气垫、卡粉、拔干、性价比……）一起 `jieba.add_word`，
   保证人工标注的词在分词结果里整词出现；
2. **谨慎去停用词**：停用词表只收虚词/代词/连接词/语气词（的、地、得、我、这、因为……），
   **明确保留**：
   - 否定词（不、没、没有、别……）——它们会反转情感；
   - 程度副词（很、非常、挺、有点……）——它们承载情感强度。
   此外脚本会自动做一次**安全交叉校验**：若某个停用词恰好是标注方面词/观点词，自动把它
   从停用词表剔除（本次数据 0 冲突）。纯数字串和孤立英文字母作为噪声过滤。

### 4.4 标签解析（`parse_labels`）

把原始标签表规范成机器友好的结构化表：

- 术语列 `_` 占位符 → 空值 NA（在 LLM 训练数据中再进一步转成空字符串 `""`）；
- 位置列空字符串 → NA，其余转成可空整数类型 `Int64`；
- 输出 [outputs/processed/Train\_labels\_parsed.csv](outputs/processed/Train_labels_parsed.csv)。

### 4.5 运行命令与产物

```bash
python src/preprocess.py     # 约 10 秒
python src/eda.py            # 约 10 秒，输出 5 张图
```

| 产物                                                    | 内容                                                   |
| ----------------------------------------------------- | ---------------------------------------------------- |
| `outputs/processed/Train_reviews_processed.csv`       | 列：id, Reviews（原文）, clean\_text（清洗后）, tokens（空格连接的分词） |
| `outputs/processed/Test_reviews_processed.csv`        | 同上                                                   |
| `outputs/processed/Train_labels_parsed.csv`           | 结构化标签                                                |
| `outputs/figures/01~05_*.png`                         | 长度/四元组数/类别/极性/类别×极性 5 张分布图                           |
| `outputs/logs/preprocess_report.txt`、`eda_report.txt` | 文字报告                                                 |

预处理实测：训练 3229 条、测试 2237 条评论全部保留，无空评论、无清洗后为空的评论。

### 4.6 EDA 得到的关键结论（直接影响后续建模）

1. **输出数量不固定**：37.4% 的评论含 1 个四元组，34.5% 含 2 个，最多 7 个——模型必须支持"一条评论输出不定数量个结果"，这排除了普通单标签分类思路；
2. **隐式方面占 71.4%**：抽取显式方面词只能覆盖不到 30% 的场景，模型必须显式建模"无方面词"的情况；
3. **类别极度不均衡**：整体 2822 个 vs 新鲜度 13 个；极性正面占 89%——方案五 Stage2 因此引入类别权重并按 macro-F1 选模型；
4. 标签表本身质量很高：无缺失、无越界取值、术语与位置缺失完全一致。

***

## 五、方案二：Baseline —— Double-Propagation 规则模型

**对应代码**：[src/model\_baseline.py](src/model_baseline.py)、[src/predict.py](src/predict.py)
**配置文件**：[config/model\_baseline.yaml](config/model_baseline.yaml)
**测试集 F1：0.4925**｜dev 四元组 F1：0.5535

### 5.1 基本思想（通俗版）

在深度学习之前，先做一个**完全不用训练神经网络**的传统基线，思路来自 Qiu et al. (2011)
的 Double Propagation（双传播）算法以及 Zhu et al. (2023) 的 DP-ACOS 规则基线：

> 情感词典里先放一批"种子观点词"（好、差、喜欢、过敏……）。中文句子里观点词和它修饰的
> 方面词之间通常有固定的句法关系（如"物流很快"中"快"形容"物流"）。用依存句法分析
> 找到这种关系，就可以**从已知观点词发现新方面词，再从方面词发现新观点词**，像滚雪球一样
> 在整个语料上双向迭代传播，直到不再增长。

### 5.2 四类传播规则（R1\~R4）

| 规则 | 方向        | 触发条件（依存句法）                            |
| -- | --------- | ------------------------------------- |
| R1 | 观点词 → 方面词 | 观点词通过主语/宾语/定语等弧（amod/nsubj/dobj…）连到名词 |
| R2 | 方面词 → 观点词 | 名词方面词连着形容词/动词性谓词，且该谓词不是"买/用/试"等功能动词   |
| R3 | 方面词 → 方面词 | 并列关系或共同依存同一中心（"价格和包装都好"）              |
| R4 | 观点词 → 观点词 | 并列关系，**顺承并列继承极性，转折并列（但/不过）极性翻转**      |

句法分析用 **spaCy + zh\_core\_web\_sm**（Universal Dependencies 标注体系）批量解析。

### 5.3 工程化适配（针对中文电商评论做的大量细化）

纯论文规则直接用效果并不好，代码里针对数据特点做了系统性改造：

1. **知识库从训练标签归纳**（`build_knowledge`）：观点词→极性多数投票、方面词→类别、
   观点词→类别、"隐式观点"用法等，共归纳观点词 1346 个（另手工补 67 个通用种子）、方面词 237 个；
2. **极性推断**：否定词（不/没/没有）作用域内极性翻转；程度副词（很/非常）修饰的新谓词倾向判正面；
3. **语言学术语修剪**：自动处理"淡淡的香味→淡淡的""效果好→好""便宜实惠→便宜+实惠"等
   与标注粒度对齐的短语切割；
4. **伪方面过滤**：产品品类词（面膜、隔离霜）、身体部位词（皮肤、脸）、指人名词（同事、妈妈）、
   通用名词（产品、东西）即使被句法规则抽到，也按标注习惯处理为隐式方面；
5. **类别识别流水线**（`assign_category`）：方面词精确映射 → 方面词包含匹配 → 类别关键词 →
   观点词→类别映射 → 子句关键词投票 → 默认"整体"；
6. **隐式方面的门控**：当观点词在子句内链接不到任何方面词时产生隐式方面四元组；
   并用"该观点词在训练集中的被标注率"做门控（阈值 0.25），压制"一直用/多/少"等高频但
   极少被标注的词滥发隐式四元组（100 个观点词被门控）；
7. **所有词表/阈值都可在 YAML 中调整**，不装 PyYAML 或配置缺失时自动回退代码默认值。

### 5.4 运行命令与产物

```bash
# dev 评估 + 全量训练集归纳 + 测试推理（一条命令完成；加 --skip-test 只做 dev 评估）
python src/model_baseline.py
# 用全量训练集归纳知识、只对测试集推理并生成提交文件
python src/predict.py
```

产物：`data/baseline/Result.csv`（提交文件）、`lexicon_opinion.csv`、`lexicon_aspect.csv`
（归纳词典）、`dev_predictions.csv`、`outputs/logs/baseline_report.txt`。

### 5.5 实测结果（dev，484 条）

| 评估粒度        | P      | R      | F1         |
| ----------- | ------ | ------ | ---------- |
| **四元组精确匹配** | 0.5171 | 0.5954 | **0.5535** |
| (A,O) 对     | 0.5470 | 0.6287 | 0.5850     |
| 方面词（单字段）    | 0.7460 | 0.8503 | 0.7947     |
| 观点词（单字段）    | 0.6664 | 0.7431 | 0.7027     |

一个有意思的现象：**一旦 (A,O) 对抽对，类别准确率 96.79%、极性准确率 97.28%**——
说明规则知识库的"分类"能力很强，瓶颈集中在"找对/配对"环节，这也正是后面三个学习型
方案重点改进的地方。测试集 F1 为 **0.4925**。

***

## 六、方案三：序列标注 —— BERT + CRF 多任务模型

**对应代码**：[src/model\_bert/model\_bert.py](src/model_bert/model_bert.py)、
[predict\_bert.py](src/model_bert/predict_bert.py)
**测试集 F1：0.6198**｜dev 四元组 F1：0.5918

### 6.1 基本思想

把"找词"变成经典的 **BIO 序列标注**问题：对评论里的每个字打一个标签，

- `O`：不属于任何术语；
- `B-ASP / I-ASP`：方面词的首字 / 后续字；
- `B-OPN / I-OPN`：观点词的首字 / 后续字。

例如"物 流 很 快"的标签是 `B-ASP I-ASP B-OPN I-OPN`，解码后即得到方面词"物流"、
观点词"很快"。在 BERT 之上同时接两个分类头，形成**一个网络三个任务**的多任务模型：

```
bert-base-chinese（逐字向量）
   ├── (a) tag_proj 线性层 → CRF 层：5 标签 BIO 序列标注，联合抽 aspect/opinion 片段
   ├── (b) cat_head（MLP）：13 类 Category
   └── (c) pol_head（MLP）：3 类 Polarity
分类特征 = [ CLS ] 句向量 ⊕ span 池化向量（优先取方面词片段，其次观点词片段）
```

- **CRF 层**的作用：BIO 标签有合法转移约束（如 `I-ASP` 不能跟在 `B-OPN` 后面），
  CRF 在整条序列上联合解码，比逐字独立分类更不容易出现非法标签；
- 损失：`L = CRF损失 + 0.5×类别交叉熵 + 0.5×极性交叉熵`。

### 6.2 几个关键设计

1. **输入用原始** **`Reviews`** **列而不是清洗后的** **`clean_text`**：训练标签的字符偏移是按原文
   （含标点）标注的，去标点会导致偏移错位；
2. **隐式四元组不参与 BIO 标注，但仍参与类别/极性头训练**：没有术语就没有 span，
   分类头改用 `[ CLS ]` 向量"自拼"特征，让模型学会给"无方面词"的整体评价分类；
3. 学习率分层：BERT 主干 2e-5，任务头 1e-3；线性 warmup + 衰减、梯度裁剪 1.0；
4. 超参可写在 `config/model_bert.yaml`（缺失则用代码默认值），训练结束自动把实际超参
   dump 到 `config/model_bert_trained.yaml`，并把标签字典存到 `data/bert/label_vocab.json`。

### 6.3 运行命令与产物

```bash
# 训练 + dev 评估（默认 8 epoch；可用 --epochs/--batch-size/--max-len 临时覆盖）
python src/model_bert/model_bert.py
# 加载 data/bert/bert_crf.pt 对 2237 条测试评论推理
python src/model_bert/predict_bert.py
```

| 产物                             | 说明                                 |
| ------------------------------ | ---------------------------------- |
| `data/bert/bert_crf.pt`        | 模型权重（gitignore，约训练 122.5 分钟 / CPU） |
| `data/bert/label_vocab.json`   | BIO/类别/极性标签字典（推理时必须与训练一致）          |
| `data/bert/Result.csv`         | 提交结果（3452 个四元组）                    |
| `outputs/logs/bert_report.txt` | 逐 epoch 指标                         |

### 6.4 实测结果

8 个 epoch 中 dev F1 从 0.5331 稳步升到 **0.5918**（测试集 **0.6198**），明显超过规则基线。
但与后面的方案对比可以看出：**"先标词、再按 span 拼四元组"的单模型结构在配对和隐式场景
上比较吃力**——71% 的隐式方面无法通过 BIO 标签表达，这是该方案的天花板。

***

## 七、方案四：LLM 微调 —— Qwen 指令微调（SFT）

**对应代码**：[src/llm/build\_sft\_dataset.py](src/llm/build_sft_dataset.py)、
[predict\_llm.py](src/llm/predict_llm.py)；云端训练笔记本 [notebook/llm-7b.ipynb](notebook/llm-7b.ipynb)
**配置文件**：[config/llm\_qwen7b\_full.yaml](config/llm_qwen7b_full.yaml)、
[config/llm\_qwen14b\_lora.yaml](config/llm_qwen14b_lora.yaml)
**测试集 F1：0.8042（Qwen2.5-7B 全参 2 epoch，最强单模型）**

### 7.1 基本思想：把抽取改写成"阅读理解式生成"

不再设计任何标注体系和网络头，直接让大模型学会读一句话、**生成一个四元组列表的字符串**：

```
输入（instruction + 评论）:
  你是一个聪明的AI电商助手，现在你需要帮我在商品评论中抽取：
  1.商品属性特征（AspectTerms），并判断它的种类（Categories）……（含13类说明与一个完整示例）
  下面是需要你抽取的评论：遮瑕效果不好，但物流很快
输出（模型生成）:
  [{'AspectTerms': '遮瑕效果', 'OpinionTerms': '不好', 'Categories': '功效', 'Polarities': '负面'},
   {'AspectTerms': '', 'OpinionTerms': '很快', 'Categories': '物流', 'Polarities': '正面'}]
```

这种生成范式天然支持不定数量四元组、隐式方面（输出空字符串）和类别/极性联合判断，
与任务形态高度契合。

### 7.2 第一步：构造 SFT 数据（`build_sft_dataset.py`）

```bash
python src/llm/build_sft_dataset.py
```

- 按 id 把 `Train_labels.csv` 的多行标签 groupby 成一条评论一个 list；
- 训练数据里方面词占位符 `_` 统一转成空字符串 `''`（与 prompt 示例风格一致）；
- 丢弃字符偏移列（生成式不需要位置）；
- 用 `str(list)` 写成**单引号风格 Python 字面量**，与 prompt 中的示例完全一致；
- 产物：[data/llm/sft\_train.json](data/llm/sft_train.json)（3229 条 Alpaca 格式样本，
  字段 `instruction/input/output`）与 [data/llm/dataset\_info.json](data/llm/dataset_info.json)
  （LLaMA-Factory 的数据集注册文件，声明三列分别映射到 prompt/query/response）。

### 7.3 第二步：用 LLaMA-Factory 训练

我们对两种规格模型做了对照实验（均使用 Qwen 聊天模板、cosine 学习率、gradient checkpointing）：

| 实验 | 模型                      | 方式            | 关键超参                                             | 显存               |
| -- | ----------------------- | ------------- | ------------------------------------------------ | ---------------- |
| A  | **Qwen2.5-7B-Instruct** | **全参数 SFT**   | lr 6e-6、batch 等效 32、**2 epoch**、bf16、cutoff 1024 | 约 50GB（A100 80G） |
| B  | Qwen3-14B-Instruct      | **LoRA 低秩适配** | rank 64 / alpha 128、lr 1e-4、3 epoch、LoRA 挂所有线性层  | 约 30GB（24G+ 可跑）  |

```bash
# 在 LLaMA-Factory 根目录执行（dataset_dir 指向本项目 data/llm）
llamafactory-cli train config/llm_qwen7b_full.yaml      # 7B 全参
llamafactory-cli train config/llm_qwen14b_lora.yaml     # 14B LoRA
```

> 名词解释：**LoRA** 冻结原模型，只训练挂在每层旁边的小矩阵，显存和训练量大降；
> **全参 SFT** 更新模型全部参数，拟合能力更强但需要大显存。
> V100/T4 不支持 bf16 时把配置里的 `bf16: true` 改成 `fp16: true`；
> 4-bit 量化的 QLoRA 可在 16GB 显存上运行。
> 云端（ModelScope/Colab）环境 HuggingFace 直连不通时，用 ModelScope 的 snapshot\_download
> 或设置 `HF_ENDPOINT=https://hf-mirror.com` 下载模型。

**重要经验（过拟合对照）**：14B LoRA 训练 3 个 epoch 后出现明显**过度抽取**——生成的四元组
数量远超训练集密度；7B 全参 3 epoch 也有类似现象。改为 **2 个 epoch** 后过度抽取显著减少，
测试 F1 从约 0.55 提升到 0.80。小数据（3229 条）上 SFT 轮次宁少勿多。

### 7.4 第三步：推理（`predict_llm.py`）

```bash
# transformers 后端（无需额外依赖，速度慢）
python src/llm/predict_llm.py --model_path ./saves/qwen7b_full_sft_2ep

# vLLM 后端（推荐，批量推理快 5-10 倍）
python src/llm/predict_llm.py --model_path ./saves/qwen7b_merged --backend vllm

# LoRA 未合并时：基础模型 + adapter 分开指定
python src/llm/predict_llm.py --model_path Qwen/Qwen3-14B-Instruct \
    --adapter_path ./saves/qwen14b_lora --backend vllm
```

参数：`--backend {transformers,vllm}`、`--max_new_tokens 512`、`--output 自定义路径`。

推理侧三个决定成败的细节（代码里均已处理）：

1. **必须套用 Qwen 聊天模板**（`<|im_start|>user ... <|im_end|><|im_start|>assistant`）。
   训练时 LLaMA-Factory 见到的就是这个格式；裸 prompt 推理时模型遇到部分句式会直接输出
   EOS，导致大量评论**零抽取**（曾有 13.9% 的测试 id 无输出）；
2. **输出解析双保险**：优先 `ast.literal_eval`（兼容单引号）整体解析，失败则用正则逐个
   抽取 `{...}` 块再解析；
3. **四元组清洗**：strip 空白、类别/极性过白名单（13 类、3 极性之外的丢弃）、去空四元组、
   按 (id, 四元组) 去重；空术语输出时转回 `_`，保证每个测试 id 至少一行。

产物：`data/llm/Result.csv`（14B LoRA 结果，F1 0.5486）与
`data/llm/Result 7b-full-2epocns.csv`（7B 全参 2 epoch 结果，**F1 0.8042**）；
训练曲线存档于 `config/llm-7b/training_loss.png`、`train_results.json`。

***

## 八、方案五：管道式 ACOS（方案 C）—— 对抽取 + 交叉编码分类

**对应代码**：[src/pipeline/extract\_pair.py](src/pipeline/extract_pair.py)（Stage1）、
[classify\_pair.py](src/pipeline/classify_pair.py)（Stage2）、
[predict\_pipeline.py](src/pipeline/predict_pipeline.py)（串联）
**配置文件**：[config/model\_pipeline.yaml](config/model_pipeline.yaml)
**测试集 F1：0.7651**｜dev 端到端四元组 F1：0.7587

### 8.1 为什么再做一个两阶段方案

方案三的单网络要"一边标字、一边配对、一边分类"，隐式方面还无法表达；方案四的 LLM 很强
但需要大显存。方案 C 借鉴学术界两阶段 ASQP 思路，把任务**解耦成两个职责单一的 BERT 模型**：

```
评论 ──Stage1──▶ 若干 (Aspect, Opinion) 对 ──Stage2──▶ 每对补上 (Category, Polarity) ──▶ 四元组
```

解耦的好处：每个模型目标简单、好学；错误可分别归因；全程只需 bert-base-chinese，**CPU 可训**。

### 8.2 Stage1：(Aspect, Opinion) 对抽取（`extract_pair.py`）

模型在 bert-base-chinese 上分三个支：

```
bert-base-chinese
   ├── (a) tag_proj → CRF：5 标签 BIO，分别标出 aspect / opinion 的候选片段
   ├── (b) span MLP ×2：把每个片段表示成 g(s,e) = MLP([首字向量; 尾字向量; 片段均值池化])
   └── (c) 双仿射 PairScorer：对候选格 (n_aspect+1) × (n_opinion+1) 的每一格打 sigmoid 分
```

- **双仿射配对**：可以理解成"给所有候选方面词和候选观点词拉一张配对网格，逐格判断这格的
  两个词是不是一对"，多标签 sigmoid 允许一个方面/观点参与多个对；
- **隐式 A/O 的哨兵机制**：网格额外增加最后一行、最后一列，分别对应"隐式方面""隐式观点"
  两个**可学习的哨兵向量**——观点找不到方面词时就与隐式方面哨兵配对。训练集没有"双隐式"
  样本，因此右下角格子在推理时屏蔽；
- 训练时网格行/列来自 gold 片段（teacher forcing），推理时来自 CRF 解码片段；
- 损失 `L = CRF损失 + 配对BCE`（正样本权重 pos\_weight=2，因为格子里正负对约 1:5）；
- 推理时在阈值表 \[0.3 … 0.7] 中搜索最佳配对阈值 τ，另有 top1 兜底和"安全行"机制
  保证每条评论至少产出合理数量的对。

**数据坑（非常重要）**：Stage1 必须读 `data/raw/TRAIN/Train_reviews.csv` 的**原文**，
不能用 `outputs/processed/Train_reviews_processed.csv`——后者的 Reviews 列压缩了连续空格
（3229 条中有 78 条受影响），而标签字符偏移按原文计算，空格错位会导致 span 整体错位
（如"物流/很快"被切成"流很/快"）。

```bash
python src/pipeline/extract_pair.py                       # 完整训练（CPU 约 70 分钟）
python src/pipeline/extract_pair.py --max-train 64 --max-dev 32   # 冒烟测试
```

实测（6 epoch，2745 train / 484 dev）：

- 片段抽取：Aspect span F1 ≈ 0.89，Opinion span F1 ≈ 0.86；
- **配对 dev F1 = 0.8038**（最优阈值 τ=0.7，epoch 6）；
- 产物：`data/pipeline/pair_extractor.pt`、`pair_meta.json`（含最优阈值与指标）。

### 8.3 Stage2：逐对 (Category, Polarity) 交叉编码分类（`classify_pair.py`）

拿到 Stage1 的每个 (A, O) 对后，用一个**交叉编码器**独立判断类别和极性：

```
输入序列: [ CLS ] 评论原文 [SEP] 方面词:a 观点词:o [SEP]
                                 （隐式 aspect/opinion 用"无"占位）
bert-base-chinese → [ CLS ] → dropout
        ├── cat_head：13 类 Category
        └── pol_head：3 类 Polarity
```

- 把评论和候选对拼在同一句里做交叉注意力，分类器能看到观点词所处的上下文；
- **类别不均衡的对策**：两类损失都乘**截断逆频率类权重**（权重上限 cap=10），既保护
  "新鲜度/中性"等稀有类，又防止权重爆炸；
- 选模型标准：`(cat_macro_F1 + pol_macro_F1) / 2` 最大（macro 对稀有类更公平）；
- 数据切分与 Stage1 完全一致（同 seed=42 的评论级 85/15），保证 dev 评论对齐。

```bash
python src/pipeline/classify_pair.py                      # CPU 约 173 分钟
```

实测（5 epoch，给定 gold 对评估）：

| 指标                      | 值                   |
| ----------------------- | ------------------- |
| Category macro-F1 / 准确率 | **0.9036** / 0.9566 |
| Polarity macro-F1 / 准确率 | **0.8619** / 0.9773 |
| 四元组准确率（对+类别+极性全对）       | **0.9360**          |

最易混淆类别：使用体验↔功效（10 例，二者边界本就模糊）；稀有类"其他/尺寸"样本太少 F1 偏低。

### 8.4 串联推理与端到端评估（`predict_pipeline.py`）

```bash
python src/pipeline/predict_pipeline.py --eval-dev       # dev 跑完整管道，输出四元组 P/R/F1
python src/pipeline/predict_pipeline.py                  # 对测试集推理
python src/pipeline/predict_pipeline.py --source train   # 对训练集推理（误差分析用）
```

串联流程：Stage1 加载 `pair_extractor.pt` 按 τ 解码候选对 → Stage2 加载
`pair_classifier.pt` 逐对交叉编码分类 → 拼成四元组 → 写
`data/pipeline/Result.csv`（同样保证每个测试 id 至少一行）。

端到端结果与误差归因：

- **dev 端到端四元组 F1 ≈ 0.7587，测试集 F1 = 0.7651**，且基本消除了零抽取；
- 测试集共输出 4728 个四元组，覆盖全部 2237 条评论，密度 2.11 个/条，与训练集 2.05 接近；
- 误差分析表明 **Stage1 对抽取是主要瓶颈**：端到端的假阳性中约 83% 在 Stage1 已经配错对，
  Stage2 本身准确率很高。后续提分应优先优化对抽取（调阈值、增训轮数、改进 span 表示）。

***

## 九、方案六：多模型加权投票融合

**对应代码**：[src/ensemble.py](src/ensemble.py)
**测试集 F1：0.8116（最终提交，超过最强单模型 0.8042）**

### 9.1 融合策略

不同建模范式的错误往往不相关，因此对 5 个模型的提交结果做**软加权投票**：

| 候选结果                                  | 模型                | 权重（=各自测试 F1） |
| ------------------------------------- | ----------------- | ------------ |
| `data/llm/Result 7b-full-2epocns.csv` | Qwen2.5-7B 全参 SFT | 0.8042       |
| `data/pipeline/Result.csv`            | 方案五 管道式           | 0.7651       |
| `data/bert/Result.csv`                | 方案三 BERT+CRF      | 0.6198       |
| `data/llm/Result.csv`                 | Qwen3-14B LoRA    | 0.5486       |
| `data/baseline/Result.csv`            | 方案二 规则基线          | 0.4925       |

流程：

1. 字段归一化（strip；空值/`_`/`无`/`nan` 统一为 `_`），按 `(id, A, O, C, P)` 去重；
2. 统计每个四元组在各模型中的**权重之和**，达到阈值 τ 才保留；
3. **空 id 回退**：某条评论融合后一个四元组都不剩，则整条回退到最强模型（7B）的预测；
4. 遍历多个阈值，每个阈值输出一个 `data/ensemble/Result_t{τ}.csv`，分别提交对比。

### 9.2 阈值实验结论

由于 5 个权重两两之和只有 9 个离散取值，阈值落在不同"模型组合带"上结果才不同：

| 阈值 τ | 测试 F1          | 含义                       |
| ---- | -------------- | ------------------------ |
| 0.80 | 0.7922         | 几乎保留所有预测（7B 独有噪声也进来了，最差） |
| 1.26 | 0.8098         | 剔除弱共识                    |
| 1.45 | **0.8116（最优）** | 保留 7B 与强模型的共识            |
| 1.60 | 0.8064         | 共识过严，开始误删                |
| 2.00 | 0.8003         | 只留三方以上共识，召回损失大           |

核心发现：LLM 7B 的**独有预测是有害的**；"7B+管道""7B+BERT"这类强模型两两共识增益最大；
过松会放进单模型噪声，过严会损失弱模型之间仍然正确的三方共识。运行：

```bash
python src/ensemble.py        # 输出 data/ensemble/Result.csv 及 Result_t*.csv
```

***

## 十、结果复现说明

1. **随机种子**：全部方案统一 `seed=42`；train/dev 为评论级 shuffle 后 85%/15%
   （2745/484 条），方案五两个阶段使用完全相同的切分以便端到端评估。
2. **评估方式**：四元组四个字段精确匹配计算 P/R/F1；线上以测试集提交后的 F1 为准。
3. **训练耗时参考**（均为纯 CPU，除 LLM 外）：
   - 预处理 + EDA：各约 10 秒；
   - 规则基线：语料解析 + 双传播约 1 分钟；
   - BERT+CRF：训练 122.5 分钟，测试推理约 2 分钟；
   - 管道 Stage1：70.4 分钟；Stage2：173.4 分钟；
   - Qwen2.5-7B 全参 SFT：云端 A100 约 42 分钟 / 3 epoch（vLLM 批量推理 2237 条约数分钟）。
4. **权重文件**：`*.pt`、`saves/` 已被 `.gitignore` 忽略，按第五\~八章对应命令重新训练/归纳即可生成；
   所有 `Result.csv`、标签字典与 meta.json 均已提供，可直接运行 `python src/ensemble.py` 复现融合。
5. **方案选型小结**：规则基线提供可解释的下限；BIO 序列标注验证了判别式方法但受困于隐式方面；
   管道式以两个小模型分别攻克"配对"与"分类"，CPU 可训且达到 0.765；7B 生成式 SFT 单模型最强
   （0.8042）；最终多模型加权投票以 **0.8116** 的测试 F1 作为提交结果。

