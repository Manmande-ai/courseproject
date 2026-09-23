# -*- coding: utf-8 -*-
"""多模型四元组预测融合（加权投票）。

策略：
1. 读取多个模型的结果 CSV（无表头，每行一个四元组 id,AspectTerms,OpinionTerms,Categories,Polarities），
   每个模型的权重 = 其验证集 F1 分数；
2. 字段归一化（strip；空值/'_'/'无' 等统一为 '_'），按 (id, A, O, C, P) 去重；
3. 加权投票：统计每个四元组的权重和，>= 阈值则保留。
   权重和是分带的（pair 权和仅 9 个离散值，最小三方和 1.6609），
   阈值落在不同"模型组合带"内结果才不同；
4. 空回退：融合后某 id 四元组为空，则保留分数最高模型的该 id 完整预测；
5. 每个阈值各输出一个无表头 CSV（Result_t{阈值}.csv），格式与提交一致。

已验证测试分数: 0.8→0.7922 | 1.26→0.8098 |1.30→0.8095|1.35→0.8083 | 1.45→0.8116(最优) |1.45→0.8112 | 1.6→0.8064 | 2.0→0.8003
"""

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent  # courseproject/
DATA_DIR = BASE_DIR / 'data'

# 候选结果文件与对应验证分数（作为投票权重），按分数降序排列
PRED_FILES = [
    (DATA_DIR / 'llm' / 'Result 7b-full-2epocns.csv', 0.8042),
    (DATA_DIR / 'pipeline' / 'Result.csv', 0.7651),
    (DATA_DIR / 'bert' / 'Result.csv', 0.6198),
    (DATA_DIR / 'llm' / 'Result.csv', 0.5486),
    (DATA_DIR / 'baseline' / 'Result.csv', 0.4925),
]

# 加权投票阈值列表，每个阈值生成一个 Result_t{阈值}.csv，便于分别提交对比。
# pair 权和排序: llm+base 1.0411 / bert+base 1.1123 / bert+llm 1.1684 / pipe+base 1.2576 /
#               7b+base 1.2967 / pipe+llm 1.3137 / pipe+bert 1.3849 / 7b+bert 1.4240 / 7b+pipe 1.5693
# 最小三方和 bert+llm+base = 1.6609（低于它的阈值都能保留全部三方共识）
# 待测带: 1.30 = 1.26版-7b+base | 1.35 = 1.4版+pipe+bert | 1.45 = 1.4版-7b+bert
WEIGHT_THRESHOLDS = [1.30, 1.35, 1.45]
OUTPUT_DIR = DATA_DIR / 'ensemble'

COLUMNS = ['AspectTerms', 'OpinionTerms', 'Categories', 'Polarities']
NULL_TOKENS = {'', '_', '无', 'nan', 'None', 'NULL', 'null'}


def norm(value: str) -> str:
    """字段归一化：去首尾空白，空值统一为 '_'。"""
    value = str(value).strip()
    return '_' if value in NULL_TOKENS else value


def load_pred(path: Path) -> dict:
    """读取单个结果文件，返回 {id: {(A, O, C, P), ...}}。"""
    df = pd.read_csv(path, header=None, names=['id'] + COLUMNS, dtype=str)
    df['id'] = df['id'].str.strip()
    for col in COLUMNS:
        df[col] = df[col].map(norm)
    return {rid: set(map(tuple, grp[COLUMNS].values)) for rid, grp in df.groupby('id')}


def main():
    preds, weights = {}, {}
    for path, weight in PRED_FILES:
        name = f'{path.parent.name}/{path.name}'
        preds[name] = load_pred(path)
        weights[name] = weight
        n_quads = sum(len(v) for v in preds[name].values())
        print(f'已加载 {name}  (权重 {weight}): {len(preds[name])} 个 id, {n_quads} 个四元组')

    best_name = max(weights, key=weights.get)
    best_pred = preds[best_name]
    print(f'\n最优模型（空回退用）: {best_name}')

    # 加权投票：每个 (id, 四元组) 的权重和（各阈值共用，只算一次）
    weight_sum = Counter()
    for name, pred in preds.items():
        for rid, quads in pred.items():
            for quad in quads:
                weight_sum[(rid, quad)] += weights[name]

    # 按 id 聚合，避免逐 id 遍历全部计数
    by_id = defaultdict(list)
    for (rid, quad), w in weight_sum.items():
        by_id[rid].append((quad, w))

    all_ids = sorted(by_id, key=lambda x: int(x) if x.isdigit() else 10 ** 9)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for threshold in WEIGHT_THRESHOLDS:
        fused, fallback_ids = {}, []
        for rid in all_ids:
            kept = [(quad, w) for quad, w in by_id[rid] if w >= threshold]
            if kept:
                # 权重和降序，同权重按四元组字典序，保证输出确定性
                kept.sort(key=lambda x: (-x[1], x[0]))
                fused[rid] = [quad for quad, _ in kept]
            else:
                fallback_ids.append(rid)
                fused[rid] = sorted(best_pred.get(rid, set()))

        # 写出（无表头，与提交格式一致）
        out_path = OUTPUT_DIR / f'Result_t{threshold:g}.csv'
        rows = [[rid, *quad] for rid in all_ids for quad in fused[rid]]
        pd.DataFrame(rows).to_csv(out_path, header=False, index=False, encoding='utf-8')

        n_quads = len(rows)
        note = f', 空回退: {len(fallback_ids)} 个id' if fallback_ids else ''
        print(f'\n阈值 {threshold:g} -> {out_path.name}: 四元组 {n_quads}, '
              f'密度 {n_quads / len(all_ids):.2f}/条{note}')


if __name__ == '__main__':
    main()
