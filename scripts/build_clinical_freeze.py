#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 ds005713 的临床变量从 participants.xlsx 冻结为可复现的 TSV，并生成数据字典与清点报告。

背景：当前快照（DOI 10.18112/openneuro.ds005713.v2.0.3，许可 CC0）根目录下不存在
participants.tsv，临床变量仅以 participants.xlsx 提供（且该文件被 .bidsignore 排除）。
因此本项目自行冻结一份 TSV，作为全部分析的唯一临床变量来源。

用法：python scripts/build_clinical_freeze.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data' / 'raw_metadata' / 'participants.xlsx'
OUT_DIR = ROOT / 'data' / 'frozen'
MANIFEST_DIR = ROOT / 'data' / 'manifests'
SCHEMA = ROOT / 'data' / 'raw_metadata' / 'participants.json'


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(SRC)
    df.columns = [str(c).strip() for c in df.columns]

    # 统一主键列名
    id_cols = [c for c in df.columns if c.lower() in ('participant_id', 'bids_id')]
    if id_cols:
        df = df.rename(columns={id_cols[0]: 'participant_id'})

    df.to_csv(OUT_DIR / 'participants_frozen.tsv', sep='\t', index=False, encoding='utf-8')

    lines = ['# ds005713 临床变量清点报告',
             '',
             f'- 源文件: data/raw_metadata/participants.xlsx',
             f'- 冻结产物: data/frozen/participants_frozen.tsv',
             f'- 行数: {len(df)}',
             f'- 列数: {df.shape[1]}',
             '']
    lines.append('## 列清单与缺失情况')
    lines.append('')
    lines.append('| 列名 | 非空数 | 缺失率 | 取值示例/分布 |')
    lines.append('|---|---|---|---|')
    for c in df.columns:
        nn = int(df[c].notna().sum())
        miss = 1 - nn / max(len(df), 1)
        series = df[c].dropna()
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 8:
            sample = f'min={series.min():.3g}, median={series.median():.3g}, max={series.max():.3g}'
        else:
            vals = series.astype(str).value_counts().head(6)
            sample = '; '.join(f'{k} ({v})' for k, v in vals.items())
        lines.append(f'| {c} | {nn} | {miss:.1%} | {sample[:180]} |')

    # group 列交叉核对
    lines += ['', '## 关键分组核对']
    gcol = next((c for c in df.columns if c.lower() == 'group'), None)
    if gcol:
        lines.append('')
        lines.append('| group | n | 平均年龄 | 女性比例 |')
        lines.append('|---|---|---|---|')
        for g, sub in df.groupby(gcol):
            sexcol = next((c for c in df.columns if c.lower() == 'sex'), None)
            fem = ''
            if sexcol:
                s = pd.to_numeric(sub[sexcol], errors='coerce')
                fem = f'{(s == 0).mean():.1%}' if s.notna().any() else ''
            lines.append(f'| {g} | {len(sub)} | {sub["age"].mean():.1f} | {fem} |' if 'age' in df.columns
                         else f'| {g} | {len(sub)} | | {fem} |')
    else:
        lines.append('')
        lines.append('**警告**：未找到 group 列，无法自动核对分组，需人工确认。')

    # 术式 / 随访可用性
    for col, title in [('Surgery_type', '术式分布'), ('Surgery_outcome', '手术结局分布'),
                       ('First_follow-up', '第一次随访可用性'), ('Second_follow-up', '第二次随访可用性')]:
        real = next((c for c in df.columns if c.lower().replace('_', '-') == col.lower()), None)
        if real:
            lines += ['', f'## {title}（{real}）', '']
            vc = df[real].astype(str).value_counts()
            for k, v in vc.items():
                lines.append(f'- {k}: {v}')

    # 与 schema 对照
    if SCHEMA.exists():
        schema = json.loads(SCHEMA.read_text(encoding='utf-8'))
        missing_in_data = [k for k in schema if k not in df.columns]
        extra_in_data = [c for c in df.columns if c not in schema]
        lines += ['', '## 与 participants.json 的一致性', '',
                  f'- schema 字段数: {len(schema)}；数据字段数: {df.shape[1]}',
                  f'- 数据中缺失的 schema 字段: {missing_in_data or "无"}',
                  f'- 数据中多出的字段: {extra_in_data or "无"}']

    (OUT_DIR / 'clinical_inventory.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:60]))
    print(f'\n已写出: {OUT_DIR / "participants_frozen.tsv"}')
    print(f'已写出: {OUT_DIR / "clinical_inventory.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
