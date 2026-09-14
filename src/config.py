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
# 原始数据路径
# ----------------------------------------------------------------------
TRAIN_REVIEWS_PATH = BASE_DIR / "data" / "raw" / "TRAIN" / "Train_reviews.csv"
TRAIN_LABELS_PATH = BASE_DIR / "data" / "raw" / "TRAIN" / "Train_labels.csv"
TEST_REVIEWS_PATH = BASE_DIR / "data" / "raw" / "TEST" / "Test_reviews.csv"

# ----------------------------------------------------------------------
# 产物目录(outputs 三层结构: processed=清洗数据 / figures=图表 / logs=运行日志)
# ----------------------------------------------------------------------
PROCESSED_DIR = BASE_DIR / "outputs" / "processed"    # preprocess.py 清洗数据输出
FIGURES_DIR = BASE_DIR / "outputs" / "figures"        # eda.py 图表输出
BASELINE_DIR = BASE_DIR / "data" / "baseline"         # model_baseline.py 输出

# eda.py 图表目录
EDA_FIG_DIR = FIGURES_DIR

# ----------------------------------------------------------------------
# 运行日志(三个脚本的文字报告统一存放于 outputs/logs)
# ----------------------------------------------------------------------
LOG_DIR = BASE_DIR / "outputs" / "logs"
PREPROCESS_LOG_PATH = LOG_DIR / "preprocess_report.txt"
EDA_REPORT_PATH = LOG_DIR / "eda_report.txt"
BASELINE_LOG_PATH = LOG_DIR / "baseline_report.txt"
