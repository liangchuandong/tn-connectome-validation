#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成投稿用图表（阶段十）。零依赖，输出矢量 SVG。

读取各阶段**真实输出**，缺失的模块会被跳过并在清单中标注（不生成占位图）。

数据来源与图件对应
------------------
- ``phenotypes/contrast_*.csv``          → 图 1（三模态效应量空间分布）
- ``phenotypes/effect_map_*.csv``        → 图 2（共线性诊断）
- ``phenotypes/run_summary.json``        → 图 1 的被试计数、图 6 的流程注记
- ``transcriptome/pls_*.csv``            → 图 3（PLS1 权重分布 + 稳定性）
- ``transcriptome/convergence.json``     → 图 4（权重余弦 + 收敛矩阵）
- ``enrichment/ora_*.csv``               → 图 5（富集条形图）
- ``enrichment/ewce_*.json``             → 图 5（细胞类型）
- ``longitudinal/paired_*.csv``          → 图 6（术后变化）
- ``longitudinal/interaction_*.csv``     → 图 6
- ``figures/subject_flow.json``（可选）  → 图 1（纳入流程，人工填写后生成）

用法::

    python scripts/make_figures.py --out-dir build/figures \\
        --phenotypes build/phenotypes_out --longitudinal build/longitudinal_out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tnwt import svgplots as fg  # noqa: E402

PHENO_LABELS = {
    'A1_streamlines': 'A1 WM 连接强度（流线数）',
    'A1_fa': 'A1 WM 连接强度（FA 加权）',
    'A2_coupling': 'A2 结构-功能耦合',
    'A3_gwc': 'A3 灰白质对比度',
    'A3_t1t2': 'A3 T1w/T2w 比值',
}


def _parcel_tick_labels(parcels, every: int = 10) -> list[str]:
    """为分区轴生成刻度标签（每 every 个显示一次，其余留空）。

    分区标签可能是 ``P1``/``1``/``7Networks_LH_Vis_1`` 等形式，
    因此**不能**直接 ``int()``——只取尾部数字部分，取不到则原样显示。
    """
    out = []
    for i, p in enumerate(parcels):
        if i % every:
            out.append('')
            continue
        s = str(p)
        digits = ''.join(ch for ch in s if ch.isdigit())
        out.append(digits if digits else s)
    return out


def _read_csv(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None


def _read_json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


# ==========================================================================
# 图 1：三模态效应量空间分布 + 被试计数
# ==========================================================================
def fig1_effect_profiles(pheno_dir: Path, out: Path, manifest: dict) -> bool:
    contrasts = {}
    for name in PHENO_LABELS:
        df = _read_csv(pheno_dir / f'contrast_{name}.csv')
        if df is not None and 'hedges_g' in df.columns:
            contrasts[name] = df
    if not contrasts:
        return False

    width, height = fg.DOUBLE_COL, 150
    panels = []
    n = len(contrasts)
    pw = width / n
    for i, (name, df) in enumerate(contrasts.items()):
        g = df['hedges_g'].to_numpy(dtype=float)
        sig = (df['fdr_significant'].to_numpy(dtype=bool)
               if 'fdr_significant' in df.columns else np.zeros(len(g), dtype=bool))
        colors = [fg.PALETTE['vermillion'] if s else fg.PALETTE['grey'] for s in sig]
        panels.append((pw * i, 0, fg.bar_chart(
            pw - 6, height, np.abs(g), _parcel_tick_labels(df['parcel']),
            ylabel='|Hedges g|' if i == 0 else '', colors=colors,
            panel_label=chr(65 + i),
            note=f'{PHENO_LABELS[name]}；FDR 显著 {int(sig.sum())}/{len(g)}')))
    content = fg.composite(width, height, panels,
                           title='图 1  三模态组间效应量空间分布')
    fg.write_svg(out / 'figure1_effect_profiles.svg', content,
                 {'panels': list(contrasts), 'note': '红色 = FDR 显著分区'})
    manifest['figure1_effect_profiles'] = {
        'file': 'figure1_effect_profiles.svg',
        'phenotypes': list(contrasts),
        'n_parcels': int(len(next(iter(contrasts.values())))),
    }
    return True


# ==========================================================================
# 图 2：共线性诊断（相关矩阵 + VIF）
# ==========================================================================
def fig2_collinearity(pheno_dir: Path, out: Path, manifest: dict) -> bool:
    maps = {}
    for name in PHENO_LABELS:
        df = _read_csv(pheno_dir / f'effect_map_{name}.csv')
        if df is not None and 'effect' in df.columns:
            maps[name] = df['effect'].to_numpy(dtype=float)
    if len(maps) < 2:
        return False
    from tnwt.stats_imaging import collinearity_diagnostics
    diag = collinearity_diagnostics(maps)
    names = list(maps)
    corr = np.array(diag['corr'])
    vif = [diag['vif'][k] for k in names]

    pw = fg.DOUBLE_COL * 0.55
    panels = [
        (0, 0, fg.heatmap_chart(pw, 150, corr, names, names, panel_label='A',
                                cbar_label='Pearson r',
                                title='A1/A2/A3 空间共线性')),
        (pw + 6, 0, fg.bar_chart(fg.DOUBLE_COL - pw - 6, 150, vif, names,
                                 ylabel='VIF', panel_label='B',
                                 note=f"条件数 = {diag['condition_number']:.1f}；"
                                      "判据：两两 |r| < 0.7 方可做收敛检验")),
    ]
    content = fg.composite(fg.DOUBLE_COL, 150, panels,
                           title='图 2  共线性诊断与收敛前置检验')
    fg.write_svg(out / 'figure2_collinearity.svg', content,
                 {'vif': diag['vif'], 'condition_number': diag['condition_number'],
                  'pairwise_r': diag['corr'], 'names': names})
    manifest['figure2_collinearity'] = {
        'file': 'figure2_collinearity.svg', 'names': names,
        'condition_number': diag['condition_number'],
    }
    return True


# ==========================================================================
# 图 3：PLS1 结果（权重分布 + 稳定性）
# ==========================================================================
def fig3_pls(transcriptome_dir: Path, out: Path, manifest: dict) -> bool:
    if not transcriptome_dir or not transcriptome_dir.exists():
        return False
    weights_files = sorted(transcriptome_dir.glob('pls_weights_*.csv'))
    stab_files = sorted(transcriptome_dir.glob('pls_stability_*.csv'))
    if not weights_files:
        return False
    name = weights_files[0].stem.replace('pls_weights_', '')
    w = pd.read_csv(weights_files[0])
    z = w['z'].to_numpy(dtype=float) if 'z' in w.columns else w.iloc[:, 1].to_numpy(dtype=float)
    panels = [(0, 0, fg.histogram_chart(
        fg.DOUBLE_COL * 0.5, 150, z, bins=60, xlabel='PLS1 基因权重（Z 分数）',
        panel_label='A', threshold=2.0, threshold_label='|Z|=2',
        annotate=f'{name}：{len(z)} 个基因'))]
    sf = next((p for p in stab_files if name in p.name), None)
    if sf is not None:
        s = pd.read_csv(sf)
        if 'selection_freq' in s.columns and 'z' in s.columns:
            panels.append((fg.DOUBLE_COL * 0.5 + 6, 0, fg.scatter_chart(
                fg.DOUBLE_COL * 0.5 - 6, 150, s['z'].to_numpy(dtype=float),
                s['selection_freq'].to_numpy(dtype=float),
                xlabel='PLS1 权重 Z', ylabel='重采样入选率', fit=False,
                panel_label='B',
                annotate='仅入选率 > 0.8 者列为候选基因')))
    content = fg.composite(fg.DOUBLE_COL, 150, panels,
                           title='图 3  PLS1 基因权重分布与稳定性')
    fg.write_svg(out / 'figure3_pls.svg', content, {'phenotype': name})
    manifest['figure3_pls'] = {'file': 'figure3_pls.svg', 'phenotype': name,
                              'n_genes': int(len(z)),
                              'has_stability_panel': len(panels) > 1}
    return True


# ==========================================================================
# 图 4：跨模态收敛
# ==========================================================================
def fig4_convergence(transcriptome_dir: Path, out: Path, manifest: dict) -> bool:
    if not transcriptome_dir or not transcriptome_dir.exists():
        return False
    conv = _read_json(transcriptome_dir / 'convergence.json')
    if not conv:
        return False
    mods = conv.get('modalities') or []
    cos = conv.get('weight_cosine') or {}
    if len(mods) < 2:
        return False
    M = np.full((len(mods), len(mods)), np.nan)
    for i, a in enumerate(mods):
        for j, b in enumerate(mods):
            if i == j:
                M[i, j] = 1.0
            else:
                v = cos.get(f'{a}|{b}', cos.get(f'{b}|{a}'))
                if v is not None:
                    M[i, j] = v
    panels = [(0, 0, fg.heatmap_chart(
        fg.DOUBLE_COL * 0.55, 150, M, mods, mods, panel_label='A',
        cbar_label='PLS1 权重余弦相似度', value_range=(-1, 1),
        title='模态间基因权重一致性'))]
    passed = conv.get('converged')
    crit = conv.get('criteria') or {}
    # 注意：森林图的总高 = 绘图区高度 + 轴标签/标题占用的下边距（约 30pt），
    # 因此片段高度必须小于组合图高度，否则 x 轴标签会被裁剪（实测踩到）。
    panels.append((fg.DOUBLE_COL * 0.55 + 6, 0, fg.forest_chart(
        fg.DOUBLE_COL * 0.45 - 6, 118,
        list(crit.keys()) or ['判据未记录'],
        [float(v) for v in crit.values()] if crit else [float('nan')],
        xlabel='事前判据阈值', panel_label='B', null_value=0.0,
        annotate=('结论：收敛' if passed else
                  ('结论：不收敛/不可分离（报告模态特异性）' if passed is False
                   else '结论：未判定')))))
    content = fg.composite(fg.DOUBLE_COL, 150, panels,
                           title='图 4  跨模态收敛检验')
    fg.write_svg(out / 'figure4_convergence.svg', content, conv)
    manifest['figure4_convergence'] = {'file': 'figure4_convergence.svg',
                                       'converged': passed, 'modalities': mods}
    return True


# ==========================================================================
# 图 5：富集与细胞类型
# ==========================================================================
def fig5_enrichment(enrich_dir: Path, out: Path, manifest: dict) -> bool:
    if not enrich_dir or not enrich_dir.exists():
        return False
    ora_files = sorted(enrich_dir.glob('ora_*.csv'))
    ewce_files = sorted(enrich_dir.glob('ewce_*.json'))
    panels, used = [], {}
    if ora_files:
        df = pd.read_csv(ora_files[0])
        if 'p' in df.columns and len(df):
            top = df.nsmallest(min(12, len(df)), 'p')
            labels = [str(s).split('|')[-1][:38] for s in top.iloc[:, 0]]
            panels.append((0, 0, fg.forest_chart(
                fg.DOUBLE_COL * 0.62, 160,
                labels, -np.log10(top['p'].clip(lower=1e-300).to_numpy(dtype=float)),
                xlabel='-log10(p)', panel_label='A',
                annotate=f'Top {len(top)} 富集条目（{ora_files[0].stem}）')))
            used['ora'] = ora_files[0].name
    if ewce_files:
        e = _read_json(ewce_files[0])
        res = (e or {}).get('results') or []
        if res:
            ct = [r['cell_type'] for r in res]
            fc = [float(r.get('fold_change', np.nan)) for r in res]
            panels.append((fg.DOUBLE_COL * 0.62 + 6, 0, fg.bar_chart(
                fg.DOUBLE_COL * 0.38 - 6, 160, fc, ct, ylabel='fold change',
                panel_label='B', highlight=[r.get('fdr_significant', False) for r in res],
                note='细胞类型富集（EWCE）；红色 = FDR 显著')))
            used['ewce'] = ewce_files[0].name
    if not panels:
        return False
    content = fg.composite(fg.DOUBLE_COL, 160, panels,
                           title='图 5  功能富集与细胞类型')
    fg.write_svg(out / 'figure5_enrichment.svg', content, used)
    manifest['figure5_enrichment'] = {'file': 'figure5_enrichment.svg', **used}
    return True


# ==========================================================================
# 图 6：纵向变化
# ==========================================================================
def fig6_longitudinal(long_dir: Path, out: Path, manifest: dict) -> bool:
    if not long_dir or not long_dir.exists():
        return False
    paired = sorted(long_dir.glob('paired_*.csv'))
    if not paired:
        return False
    n = min(len(paired), 5)
    pw = fg.DOUBLE_COL / 2
    ph = 140
    panels = []
    names = []
    for i, f in enumerate(paired[:n]):
        name = f.stem.replace('paired_', '')
        df = pd.read_csv(f)
        if 'mean_change' not in df.columns:
            continue
        d = df['mean_change'].to_numpy(dtype=float)
        sig = (df['fdr_significant'].to_numpy(dtype=bool)
               if 'fdr_significant' in df.columns else np.zeros(len(d), dtype=bool))
        col = i % 2
        row = i // 2
        panels.append((pw * col, row * ph, fg.bar_chart(
            pw - 6, ph, d, _parcel_tick_labels(df['parcel']),
            ylabel='Δ（术后−术前）' if col == 0 else '',
            colors=[fg.PALETTE['vermillion'] if s else fg.PALETTE['grey'] for s in sig],
            panel_label=chr(65 + i),
            note=f'{PHENO_LABELS.get(name, name)}；FDR 显著 {int(sig.sum())}/{len(d)}')))
        names.append(name)
    if not panels:
        return False
    rows = (len(panels) + 1) // 2
    content = fg.composite(fg.DOUBLE_COL, rows * ph, panels,
                           title='图 6  术后影像变化')
    fg.write_svg(out / 'figure6_longitudinal.svg', content,
                 {'phenotypes': names})
    manifest['figure6_longitudinal'] = {'file': 'figure6_longitudinal.svg',
                                        'phenotypes': names}
    return True


# ==========================================================================
# 图 7：受试者纳入流程（需人工提供计数）
# ==========================================================================
def fig7_flow(flow_spec: Path, out: Path, manifest: dict) -> bool:
    if not flow_spec.exists():
        return False
    spec = json.loads(flow_spec.read_text(encoding='utf-8'))
    stages = spec.get('stages') or []
    if not stages:
        return False
    height = 40 + len(stages) * 46
    content = fg.flow_diagram(fg.DOUBLE_COL * 0.8, height, stages,
                              title=spec.get('title', '受试者纳入流程'))
    fg.write_svg(out / 'figure7_subject_flow.svg', content, spec)
    manifest['figure7_subject_flow'] = {'file': 'figure7_subject_flow.svg',
                                        'n_stages': len(stages)}
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='生成投稿用图表（SVG，零依赖）')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'figures')
    ap.add_argument('--phenotypes', type=Path, default=ROOT / 'data' / 'phenotypes')
    ap.add_argument('--transcriptome', type=Path, default=ROOT / 'data' / 'transcriptome')
    ap.add_argument('--enrichment', type=Path, default=ROOT / 'data' / 'enrichment')
    ap.add_argument('--longitudinal', type=Path, default=ROOT / 'data' / 'longitudinal')
    ap.add_argument('--flow-spec', type=Path,
                    default=ROOT / 'figures' / 'subject_flow.json')
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {'out_dir': str(args.out_dir), 'generated': [], 'skipped': []}
    steps = [
        ('figure1_effect_profiles', lambda: fig1_effect_profiles(args.phenotypes, args.out_dir, manifest)),
        ('figure2_collinearity', lambda: fig2_collinearity(args.phenotypes, args.out_dir, manifest)),
        ('figure3_pls', lambda: fig3_pls(args.transcriptome, args.out_dir, manifest)),
        ('figure4_convergence', lambda: fig4_convergence(args.transcriptome, args.out_dir, manifest)),
        ('figure5_enrichment', lambda: fig5_enrichment(args.enrichment, args.out_dir, manifest)),
        ('figure6_longitudinal', lambda: fig6_longitudinal(args.longitudinal, args.out_dir, manifest)),
        ('figure7_subject_flow', lambda: fig7_flow(args.flow_spec, args.out_dir, manifest)),
    ]
    for key, fn in steps:
        try:
            ok = fn()
        except Exception as exc:  # noqa: BLE001
            print(f'  [ERROR] {key}: {type(exc).__name__}: {exc}', file=sys.stderr)
            manifest['skipped'].append({'figure': key, 'reason': f'错误：{exc}'})
            continue
        if ok:
            manifest['generated'].append(key)
            print(f'  [ok]   {key} → {manifest[key]["file"]}')
        else:
            manifest['skipped'].append({'figure': key, 'reason': '输入数据尚未生成'})
            print(f'  [skip] {key}（输入数据尚未生成）')

    fg.figure_manifest(manifest, args.out_dir / 'figure_manifest.json')
    print(f'\n已生成 {len(manifest["generated"])} 张图；'
          f'跳过 {len(manifest["skipped"])} 张（输入缺失）')
    print(f'清单：{args.out_dir / "figure_manifest.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
