#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在冻结的临床变量上派生规范列并输出分组统计，用于确定统计口径。

派生列（不修改原始列，全部以 _norm/_derived 结尾）：
- group_norm        : patient / control，由 Clinical_finding 归一化得到
                      （注意：participants.xlsx 中没有 group 列，schema 里却有）
- sex_label         : schema 定义 0=女性、1=男性
- pain_side_norm    : R / L / Bilateral
- pain_type_norm    : Paroxysmal / Persistent
- nvc_side          : 由 MRI_finding 文本解析出的 NVC 侧别
- fazekas_grade     : 由 MRI_finding 文本解析出的 Fazekas 分级
- surgery_type_norm : MVD / PI / Outpatient（去除空格）
- surgery_outcome_norm, complete_relief
- first_fu_available, second_fu_available

用法：python scripts/derive_clinical.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from tnwt.stats_lite import chi2_contingency, welch_ttest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data' / 'frozen' / 'participants_frozen.tsv'
OUT = ROOT / 'data' / 'frozen' / 'participants_derived.tsv'
REPORT = ROOT / 'data' / 'frozen' / 'group_stats.md'


def norm_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()


def main() -> int:
    df = pd.read_csv(SRC, sep='\t')

    cf = norm_text(df['Clinical_finding']).str.lower()
    df['group_norm'] = cf.apply(
        lambda x: 'control' if x.startswith('normal') else ('patient' if x.startswith('tn') else 'other')
    )
    df['sex_label'] = pd.to_numeric(df['sex'], errors='coerce').map({0: 'F', 1: 'M'})
    df['pain_side_norm'] = norm_text(df['Pain_side'])
    df['pain_type_norm'] = norm_text(df['Pain_type'])
    df['surgery_type_norm'] = norm_text(df['Surgery_type'])
    df['surgery_outcome_norm'] = norm_text(df['Surgery_outcome'])

    red = pd.to_numeric(df['Postsurgery_pain_reduction (%)'], errors='coerce')
    df['pain_reduction_pct'] = red
    df['complete_relief'] = red.apply(lambda v: 1.0 if v == 100 else (0.0 if pd.notna(v) else float('nan')))

    mri = norm_text(df['MRI_finding']).str.lower()
    df['nvc_side'] = mri.apply(
        lambda x: 'L' if 'left nvc' in x else ('R' if 'right nvc' in x else ('none' if 'normal' in x else 'na'))
    )
    df['fazekas_grade'] = mri.apply(
        lambda x: (re.search(r'fazekas\s*(\d)', x).group(1) if re.search(r'fazekas\s*(\d)', x) else '0/none')
    )
    df['first_fu_available'] = norm_text(df['First follow-up']).str.lower().eq('yes')
    df['second_fu_available'] = norm_text(df['Second follow-up (telephone interview)']).str.lower().eq('yes')

    lines = ['# ds005713 分组统计与口径确认报告', '',
             f'源: {SRC.name}（172 行 = 患者 119 + 对照 53，见下）', '']

    # ---- 表 0：分组构成 ----
    lines += ['## 1. 分组构成（由 Clinical_finding 归一化）', '',
              '| group_norm | n |', '|---|---|']
    for k, v in df['group_norm'].value_counts().items():
        lines.append(f'| {k} | {v} |')

    # ---- 表 1：年龄 / 性别 / BMI 分组比较 ----
    lines += ['', '## 2. 年龄、性别、BMI 的分组比较（决定是否需要匹配/分层）', '',
              '| 变量 | 患者 | 对照 | 检验 |', '|---|---|---|---|']
    for col in ['age', 'BMI']:
        if col not in df.columns:
            continue
        a = pd.to_numeric(df.loc[df.group_norm == 'patient', col], errors='coerce').dropna()
        b = pd.to_numeric(df.loc[df.group_norm == 'control', col], errors='coerce').dropna()
        if len(a) > 2 and len(b) > 2:
            r = welch_ttest(a, b)
            lines.append(f'| {col} | {r["mean_a"]:.1f} ± {a.std():.1f} (n={r["n_a"]}) | '
                         f'{r["mean_b"]:.1f} ± {b.std():.1f} (n={r["n_b"]}) | Welch t={r["t"]:.2f}, p={r["p"]:.3g} |')
    sex = pd.crosstab(df['group_norm'], df['sex_label'])
    if sex.shape == (2, 2):
        r = chi2_contingency(sex.values)
        lines.append(f'| 性别(F/M) | {sex.loc["patient", "F"]}/{sex.loc["patient", "M"]} | '
                     f'{sex.loc["control", "F"]}/{sex.loc["control", "M"]} | chi2={r["chi2"]:.2f}, p={r["p"]:.3g} |')

    # ---- 表 2：患者临床特征 ----
    pat = df[df.group_norm == 'patient']
    lines += ['', '## 3. 患者临床特征（可用 n 决定分析口径）', '',
              '| 变量 | 非空 n | 描述 |', '|---|---|---|']
    for col, label in [('Pain_side', '疼痛侧别'), ('Pain_type', '疼痛类型'),
                       ('Pain_severity (average score)', '基线疼痛强度(0-10)'),
                       ('Disease_duration (years)', '病程(年)'), ('Sindou_grade', 'Sindou 分级'),
                       ('Surgery_type', '术式'), ('Surgery_outcome', '手术结局'),
                       ('Postsurgery_pain_reduction (%)', '术后疼痛缓解(%)'),
                       ('Second follow-up pain NRS (0-10)', '末次随访 NRS'),
                       ('Surgical_complications', '手术并发症')]:
        if col not in pat.columns:
            continue
        s = pat[col].dropna()
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 8:
            desc = f'median={s.median():.3g}, IQR={s.quantile(.25):.3g}–{s.quantile(.75):.3g}, range={s.min():.3g}–{s.max():.3g}'
        else:
            vc = norm_text(s).value_counts().head(5)
            desc = '; '.join(f'{k} ({v})' for k, v in vc.items())
        lines.append(f'| {label} | {len(s)}/{len(pat)} | {desc[:150]} |')

    # ---- 表 3：术后缓解分布（决定能否做完全缓解 vs 部分缓解） ----
    lines += ['', '## 4. 术后疼痛缓解率分布（决定临床分层终点）', '']
    vc = pat['pain_reduction_pct'].dropna().value_counts().sort_index(ascending=False)
    for k, v in vc.items():
        lines.append(f'- {k:g}%: {v}')
    n_comp = int((pat['complete_relief'] == 1).sum())
    n_partial = int((pat['complete_relief'] == 0).sum())
    lines.append(f'- 完全缓解(=100%): {n_comp}；未完全缓解(<100%): {n_partial}')

    # ---- 表 4：影像与随访可用性 ----
    lines += ['', '## 5. 随访与影像可用性（决定纵向分析 n）', '',
              f'- 第一次随访（MRI）可用: {int(pat["first_fu_available"].sum())}',
              f'- 第二次随访（电话 NRS）可用: {int(pat["second_fu_available"].sum())}',
              f'- 术式: ' + '; '.join(f'{k}={v}' for k, v in pat['surgery_type_norm'].value_counts().items())]
    fu_time = pd.to_numeric(pat['First follow-up time (in days)'], errors='coerce').dropna()
    if len(fu_time):
        lines.append(f'- 第一次随访间隔(天): median={fu_time.median():.0f}, range={fu_time.min():.0f}–{fu_time.max():.0f}'
                     '（注意：实际中位约 202 天，而非"6 个月"）')

    # ---- 表 5：NVC / Fazekas ----
    lines += ['', '## 6. NVC 与白质病变（可用于分层）', '',
              '| group_norm | NVC 侧别分布 | Fazekas 分布 |', '|---|---|---|']
    for g, sub in df.groupby('group_norm'):
        a = '; '.join(f'{k}={v}' for k, v in sub['nvc_side'].value_counts().items())
        b = '; '.join(f'{k}={v}' for k, v in sub['fazekas_grade'].value_counts().items())
        lines.append(f'| {g} | {a} | {b} |')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep='\t', index=False, encoding='utf-8')
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'\n已写出: {OUT}')
    print(f'已写出: {REPORT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
