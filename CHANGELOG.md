# 课程项目 Notebook 开发日志（2026-10-30）

## 目标
为课程项目的 **6 套实验** 各生成一个独立的 Jupyter Notebook，能够**从零开始复现整套代码**：
数据预处理 → EDA → 规则基线 → BERT+CRF → 管道式 Stage1/Stage2 → LLM 全参 SFT → LLM LoRA → 多模型融合。

## 交付物

新增 6 个 Notebook，放在 `notebook/` 目录：

| 文件 | 方案 | 必须 GPU | 主要产物（写入 `src/config.py` 规定位置） |
| --- | --- | --- | --- |
| `00-preprocess-eda.ipynb` | 文本清洗 + jieba 分词 + 标签解析 + 探索性分析 + 5 张图 | ❌ CPU 即可 | `outputs/processed/*`、`outputs/figures/*.png`、`outputs/logs/preprocess_report.txt`、`eda_report.txt` |
| `01-baseline.ipynb` | Double-Propagation 规则基线（spaCy 中文依存 + 双向传播 R1~R4） | ❌ CPU 即可 | `data/baseline/Result.csv`、`lexicon_aspect.csv`、`lexicon_opinion.csv`、`outputs/logs/baseline_report.txt` |
| `02-bert-crf.ipynb` | BERT+CRF 多任务序列标注（逐字 BIO + 类别/极性 MLP 分类头） | ✅ | `data/bert/bert_crf.pt`、`label_vocab.json`、`Result.csv`、`outputs/logs/bert_report.txt` |
| `03-pipeline.ipynb` | Stage1 (Aspect,Opinion) 对抽取（BERT span + 双仿射配对）+ Stage2 交叉编码分类 | ✅ | `data/pipeline/pair_extractor.pt`、`pair_classifier.pt`、`pair_meta.json`、`pair_clf_meta.json`、`Result.csv` |
| `04-ensemble.ipynb` | 5 模型加权投票融合（权重 = 各模型测试 F1） | ❌ 纯 pandas | `data/ensemble/Result_t{阈值}.csv`（默认 1.30 / 1.35 / 1.45） |
| `05-llm-14b-lora.ipynb` | Qwen3-14B-Instruct LoRA 微调（rank 64 / alpha 128） | ✅（≥ 24GB） | `saves/qwen14b_lora/*`、`data/llm/Result.csv`、`data/llm/sft_train.json` |

原有 `llm-7b.ipynb` 同步刷新为同一模板。

## 关键设计决定

### A. src/ 脚本保持不动，路径由 `config.py` 统一控制
一开始曾尝试给 `src/config.py` 加环境变量开关 `ACOS_ARTIFACT_ROOT`，让每个 Notebook 把产物写到 `notebook/<name>/`。但用户要求 **src/ 完全不动**，于是回滚，Notebook 只做两件事：

1. SETUP cell 把 `PROJECT_ROOT = Path.cwd().parent` 定位好（兼容从 `notebook/` 目录启动）；
2. `run(cmd)` 统一通过 `subprocess.run(..., cwd=PROJECT_ROOT)` 调用 `src/` 下脚本；
3. 所有产物读写都指向 `PROJECT_ROOT`，与 `config.py` 原生路径一致。

这样 Notebook 是**薄的**：它只负责串联脚本和展示结果，核心逻辑仍由已稳定的 `src/` 提供。

### B. Notebook SETUP 模板（所有 7 个 Notebook 共用）
```python
import os, sys, subprocess
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebook":
    PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

ARTIFACT_DIR = PROJECT_ROOT                       # 与 src/config.py 原生路径一致
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

def run(cmd):
    print(">>", " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True, env=os.environ.copy())
```

### C. GPU 检查 cell（需要 GPU 的 Notebook 开头有一个）
```python
import torch
assert torch.cuda.is_available(), "未检测到 CUDA GPU"
props = torch.cuda.get_device_properties(0)
print("GPU:", props.name, "显存: %.1f GB" % (props.total_memory/1e9))
print("bf16 支持:", torch.cuda.is_bf16_supported())
```

bf16 不支持的显卡（V100、T4 等）Notebook 会自动把 yaml 的 `bf16: true` 改成 `fp16: true`。

### D. 规则基线依赖加速安装
规则基线需要额外安装 spaCy 中文模型 `zh_core_web_sm`（不在 PyPI，`python -m spacy download` 很慢且不稳定）。总结出**方案 1** 作为默认：

```bash
# 1. 清华镜像（一次性）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 2. 主依赖
pip install -r requirements.txt

# 3. spaCy 中文模型直装 wheel（不走 spacy download）
pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.8.0/zh_core_web_sm-3.8.0-py3-none-any.whl
```

已同步更新 `requirements.txt` 注释、所有 7 个 Notebook 的 Step 0 安装说明 markdown / 代码 cell。

## 调试记录

### 问题 1：`ARTIFACT_DIR` 未定义（其实是顺序误解）
用户在 Jupyter 里点 cell 顺序乱跳，把 SETUP cell（定义 `ARTIFACT_DIR`）跳过后直接跑后续读产物 cell。**正确行为是从上到下依次运行**，Notebook 里没有魔术 cell（`!` / `%`），所有依赖均显式在前。

### 问题 2：FileNotFoundError `/mnt/.../notebook/00-preprocess-eda/outputs/logs/preprocess_report.txt`
根因是之前错误设置了 `ACOS_ARTIFACT_ROOT` 让子脚本往 `notebook/<name>/` 写，但 `config.py` 没同步改。**修复**：彻底回滚 `config.py`，Notebook 不再设置任何 `ACOS_*` 环境变量，产物读写全部回到 `PROJECT_ROOT`。

### 问题 3：Notebook 语法错误
原始 `llm-7b.ipynb` 大量使用 `!nvidia-smi`、`%pip install`、`%cd` 等 Jupyter 魔术命令。在纯 Python compile 下会报 SyntaxError。**修复**：把魔术命令改成 Notebook 内定义的 `run()` / `subprocess` 调用。注意 Jupyter 环境里两种都能跑，但转 `.py` 时魔术命令会出问题。

## 遗留与后续
- 融合脚本 `ensemble.py` 当前仍硬编码 5 个模型的路径在 `data/{baseline,bert,llm,pipeline}/`；如果用户要在完全隔离的环境里跑每个 Notebook，可以考虑让 ensemble.py 也吃一组环境变量（`ACOS_*_RESULT`），但按当前需求（src 不动）先不做。
- 云端 GPU 实例的 HuggingFace / ModelScope 镜像路径与本地 Windows 路径差异较大，每个 LLM Notebook 在 Download cell 里用 `snapshot_download(cache_dir=ARTIFACT_DIR / "models")` 统一处理。
