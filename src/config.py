# -*- coding: utf-8 -*-
"""
集中路径配置(preprocess / eda / model_baseline 三个脚本共用)
==============================================================

所有脚本的数据与产物路径统一由本模块提供, 修改目录结构只需改动此文件。
目录约定: 原始数据在 data/raw, 基线产物在 data/baseline, 可再生成的中间
产物(清洗数据/图表/日志)统一在 outputs/{processed, figures, logs}。
说明: 原三个脚本中 OUT_DIR 同名但指向不同目录, 故此处按用途拆分为
PROCESSED_DIR 与 BASELINE_DIR, 各脚本导入时以 `as OUT_DIR` 别名保持
原有变量名与引用不变。
"""

from pathlib import Path

# 项目根目录(courseproject/courseproject), 基于本文件位置定位,
# 避免工作目录/反斜杠转义问题
BASE_DIR = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------
# 产物目录(outputs 三层结构: processed=清洗数据 / figures=图表 / logs=运行日志)
# ----------------------------------------------------------------------
PROCESSED_DIR = BASE_DIR / "outputs" / "processed"    # preprocess.py 清洗数据输出
FIGURES_DIR = BASE_DIR / "outputs" / "figures"        # eda.py 图表输出
BASELINE_DIR = BASE_DIR / "data" / "baseline"         # model_baseline.py 输出
BERT_DIR = BASE_DIR / "data" / "bert"                # model_bert.py 输出(权重/Result)
LLM_DIR = BASE_DIR / "data" / "llm"                  # LLM SFT 数据与预测结果

# ----------------------------------------------------------------------
# 原始数据路径
# ----------------------------------------------------------------------
TRAIN_REVIEWS_PATH = BASE_DIR / "data" / "raw" / "TRAIN" / "Train_reviews.csv"
TRAIN_LABELS_PATH = BASE_DIR / "data" / "raw" / "TRAIN" / "Train_labels.csv"
TEST_REVIEWS_PATH = BASE_DIR / "data" / "raw" / "TEST" / "Test_reviews.csv"

# ----------------------------------------------------------------------
# 预处理产物路径(outputs/processed 下, model_bert.py 直接复用, 无需重跑预处理)
#   - 标签里的字符偏移是对原始 Reviews 文本的索引, 故 BERT 输入也用 Reviews 列
# ----------------------------------------------------------------------
TRAIN_REVIEWS_PROC_PATH = PROCESSED_DIR / "Train_reviews_processed.csv"
TRAIN_LABELS_PARSED_PATH = PROCESSED_DIR / "Train_labels_parsed.csv"
TEST_REVIEWS_PROC_PATH = PROCESSED_DIR / "Test_reviews_processed.csv"

# eda.py 图表目录
EDA_FIG_DIR = FIGURES_DIR

# ----------------------------------------------------------------------
# 运行日志(脚本文字报告统一存放于 outputs/logs)
# ----------------------------------------------------------------------
LOG_DIR = BASE_DIR / "outputs" / "logs"
PREPROCESS_LOG_PATH = LOG_DIR / "preprocess_report.txt"
EDA_REPORT_PATH = LOG_DIR / "eda_report.txt"
BASELINE_LOG_PATH = LOG_DIR / "baseline_report.txt"
BERT_LOG_PATH = LOG_DIR / "bert_report.txt"           # model_bert.py 训练日志
LLM_LOG_PATH = LOG_DIR / "llm_report.txt"             # LLM 训练/推理日志

# ----------------------------------------------------------------------
# LLM SFT 数据与产物路径(data/llm 下)
#   - sft_train.json:       LLaMA-Factory 训练数据(alpaca 格式)
#   - dataset_info.json:    LLaMA-Factory 数据集注册
#   - Result.csv:           LLM 推理后拍平的提交结果
# ----------------------------------------------------------------------
LLM_SFT_TRAIN_PATH = LLM_DIR / "sft_train.json"
LLM_DATASET_INFO_PATH = LLM_DIR / "dataset_info.json"
LLM_RESULT_PATH = LLM_DIR / "Result.csv"

# ----------------------------------------------------------------------
# 配置目录(手工编辑的超参 YAML + 训练完自动生成的快照 YAML)
# ----------------------------------------------------------------------
CONFIG_DIR = BASE_DIR / "config"
BERT_CONFIG_PATH = CONFIG_DIR / "model_bert.yaml"             # 手工编辑的超参
BERT_TRAINED_SNAPSHOT_PATH = CONFIG_DIR / "model_bert_trained.yaml"  # 训练完自动 dump 的快照
LLM_QWEN7B_FULL_CONFIG_PATH = CONFIG_DIR / "llm_qwen7b_full.yaml"    # Qwen-2.5-7B/3-8B 全参 SFT
LLM_QWEN14B_LORA_CONFIG_PATH = CONFIG_DIR / "llm_qwen14b_lora.yaml"  # Qwen-3-14B LoRA
