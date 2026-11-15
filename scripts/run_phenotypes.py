#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""影像表型提取与组间比较（阶段四–五）运行入口。

输入（按 docs/derived_data_contract.md 交付）::

    <derivatives>/
    ├── a1_streamlines/sub-XXX.csv     A1 白质连接强度（流线数）
    ├── a1_fa/sub-XXX.csv              A1（FA 加权）
    ├── a3_gwc/sub-XXX.csv             A3 灰白质对比度
    ├── a3_t1t2/sub-XXX.csv            A3 平行髓鞘指标
    ├── a2/sc/sub-XXX.npy              结构连接矩阵
    ├── a2/fc/sub-XXX.npy              功能连接矩阵
    └── qc/subjects_qc.csv             QC 表

输出（``<output-dir>``）::

    phenotype_A1_streamlines.csv / phenotype_A1_fa.csv
    phenotype_A2_coupling.csv / phenotype_A3_gwc.csv / phenotype_A3_t1t2.csv
    phenotype_*_residual.csv           混淆校正后的矩阵
    contrast_<表型>.csv                每个分区的 g / t / p / q
    effect_map_<表型>.csv              仅含有效应量图谱（供 PLS 使用）
    sensitivity_nmatched_<表型>.json   样本量匹配重采样汇总
    sensitivity_<协变量>_matched_<表型>.json  年龄/FD 匹配敏感性
    run_summary.json / run_report.md

**本脚本只输出真实数据计算所得数值。** 若派生数据缺失，脚本会明确失败并提示，
不会生成占位值。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tnwt import config_lite as cl          # noqa: E402
from tnwt import phenotypes as ph           # noqa: E402
from tnwt.enrichment import fdr_bh          # noqa: E402

PHENOTYPE_SOURCES = {
    'A1_streamlines': 'a1_streamlines',
    'A1_fa': 'a1_fa',
    'A3_gwc': 'a3_gwc',
    # 原始 T1w/T2w：保留用于"标定前 vs 标定后"的分布对照（写入论文 QC 证据）。
    # ⚠️ 该版本存在逐被试缩放异质性（中位数相差 9.3 倍、双峰），
    #    其组间比较与转录组关联**不可解释**，不得用于结论。
    'A3_t1t2': 'a3_t1t2',
    # 白质内参标定版：双峰已消除（CV 0.571 → 0.083），是唯一可用于统计的版本。
    'A3_t1t2_std': 'a3_t1t2_std',
}


def _load_clinical(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'找不到冻结临床表：{path}')
    return pd.read_csv(path, sep='\t')


def _merge_qc_confounds(clinical: pd.DataFrame, qc_path: Path | None) -> pd.DataFrame:
    """把 QC 表中的影像质量协变量并入临床表。

    为什么需要：冻结临床表只有人口学与临床变量（年龄、性别、Fazekas 等），
    头动（``bold_mean_fd``）属于影像质量指标，存在 QC 表里。混淆校正若拿不到头动，
    设计矩阵会退化为"只有年龄性别"，而这恰是方案强调必须控制的一项（见 N14）。

    合并键：``participant_id``，只取基线行；只补充临床表**缺失**的列，不覆盖已有列。
    """
    if qc_path is None or not Path(qc_path).exists():
        return clinical
    qc = pd.read_csv(qc_path)
    if 'participant_id' not in qc.columns:
        return clinical
    if 'session' in qc.columns:
        qc = qc[qc['session'].astype(str).str.lower() == 'baseline']
    qc = qc.drop_duplicates(subset='participant_id', keep='first')
    candidates = [c for c in ('bold_mean_fd', 'bold_pct_fd_gt_0.2', 'bold_tsnr',
                              'dwi_mean_fd', 't1_qc_rating', 'dwi_qc_rating',
                              'bold_qc_rating', 'seg_qc_rating', 'exclusion_flag')
                  if c in qc.columns and c not in clinical.columns]
    if not candidates:
        return clinical
    take = qc[['participant_id'] + candidates]
    merged = clinical.merge(take, on='participant_id', how='left')
    print(f'  已从 QC 表并入协变量：{candidates}'
          f'（{candidates[0]} 命中 {int(merged[candidates[0]].notna().sum())}/{len(merged)} 行）')
    return merged


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='影像表型提取与组间比较（阶段四–五）')
    ap.add_argument('--derivatives', type=Path, required=True,
                    help='派生数据目录（见契约）')
    ap.add_argument('--clinical', type=Path, default=None,
                    help='冻结临床表；默认取配置中的 dataset.clinical_derived')
    ap.add_argument('--qc', type=Path, default=None,
                    help='QC 表（含 bold_mean_fd 等影像质量指标）；'
                         '不提供时若临床表缺头动会给出警告')
    ap.add_argument('--config', type=Path, default=ROOT / 'config' / 'analysis.yaml')
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'phenotypes')
    ap.add_argument('--n-parcels', type=int, default=None)
    ap.add_argument('--n-modes', type=int, default=None, help='A2 谐波模态数；默认 n/10')
    ap.add_argument('--n-matched-rep', type=int, default=1000)
    ap.add_argument('--match-on', nargs='*', default=['age', 'bold_mean_fd'],
                    help='需要做的匹配敏感性分析（列名）')
    ap.add_argument('--no-residualize', action='store_true')
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args(argv)

    cfg = cl.load(args.config) if args.config.exists() else {}
    n_parcels = args.n_parcels or int(cl.dig(cfg, 'atlas.n_parcels', 200))
    seed = args.seed if args.seed is not None else int(cl.dig(cfg, 'seeds.global', 0))
    clinical_path = args.clinical or (ROOT / cl.dig(cfg, 'dataset.clinical_derived'))

    t0 = time.time()
    print(f'载入派生数据：{args.derivatives}（分区数 {n_parcels}）')
    deriv_base, deriv_fu = ph.load_derivatives_split(args.derivatives, n_parcels=n_parcels)
    if not deriv_base.vectors:
        print('未找到任何分区向量目录；请按 docs/derived_data_contract.md 交付', file=sys.stderr)
        return 2
    n_fu_vectors = sum(v.shape[0] for v in deriv_fu.vectors.values())
    print(f'基线被试目录 {len(next(iter(deriv_base.vectors.values())).index)} 个；'
          f'随访向量 {n_fu_vectors} 条')

    clinical = _load_clinical(Path(clinical_path))
    print(f'临床表：{clinical_path}（{len(clinical)} 行）')
    if args.qc is not None:
        clinical = _merge_qc_confounds(clinical, args.qc)
    else:
        if 'bold_mean_fd' not in clinical.columns:
            print('  [warn] 临床表无头动指标；建议用 --qc 指定 QC 表以并入 '
                  'bold_mean_fd（混淆校正必需）', file=sys.stderr)

    # ---------- 1. 组装表型矩阵 ----------
    def assemble(deriv: ph.Derivatives) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for name, sub in PHENOTYPE_SOURCES.items():
            if sub in deriv.vectors:
                out[name] = deriv.vectors[sub]
        if 'a2/sc' in deriv.matrices and 'a2/fc' in deriv.matrices:
            a2 = ph.compute_a2(deriv, n_modes=args.n_modes)
            if not a2.empty:
                out['A2_coupling'] = a2
        return out

    matrices = assemble(deriv_base)
    matrices_fu = assemble(deriv_fu)
    for name, df in matrices.items():
        print(f'  {name}: 基线 {df.shape}'
              + (f'，随访 {matrices_fu[name].shape}' if name in matrices_fu else '，无随访'))
    if not matrices:
        print('没有任何可用表型', file=sys.stderr)
        return 2

    # ---------- 2. 分组信息 ----------
    clin = clinical.set_index('participant_id')
    group = clin['group_norm'] if 'group_norm' in clin.columns else None
    if group is None:
        print('临床表缺少 group_norm（应由 scripts/derive_clinical.py 生成）', file=sys.stderr)
        return 2

    summaries, contrasts, extra = [], {}, {}

    for name, mat in matrices.items():
        # 只保留有分组信息的被试（基线）
        common = [s for s in mat.index if s in group.index]
        mat = mat.loc[common]
        if mat.shape[0] < 6:
            print(f'  [skip] {name}：可用被试仅 {mat.shape[0]}', file=sys.stderr)
            continue
        _save(mat, args.output_dir / f'phenotype_{name}.csv')
        summaries.append(ph.phenotype_summary(mat, name))

        # 随访矩阵：单独写出，**不做混淆校正**（配对分析在阶段九统一处理协变量），
        # 但同样要求被试出现在临床表中。
        fu_df = matrices_fu.get(name)
        if fu_df is not None and not fu_df.empty:
            # ⚠️ 随访被试的 ID 形如 `sub-024fu`，而临床表的 participant_id 只有
            # 基线 ID（`sub-024`，共 172 行，不含任何 fu）。首版这里写成
            #     fu_common = [s for s in fu_df.index if s in group.index]
            # 于是 fu_common 恒为空 → `phenotype_*_fu.csv` **从未写出** →
            # 阶段九 run_longitudinal.py 找不到随访矩阵，会以"无随访数据"跳过
            # 全部模态，纵向分析整体空转。
            # 现按 tnwt.longitudinal.pair_ids 的同一约定（剥 `_ses-2`/`_ses-1`/`fu`
            # 后缀）折算成基线 ID 再查临床表。
            def _to_base(_s: str) -> str:
                for _suf in ('_ses-2', '_ses-1', 'fu'):
                    if _s.endswith(_suf):
                        return _s[: -len(_suf)]
                return _s

            fu_common = [s for s in fu_df.index if _to_base(s) in group.index]
            fu_df = fu_df.loc[fu_common]
            if not fu_df.empty:
                _save(fu_df, args.output_dir / f'phenotype_{name}_fu.csv')
                print(f'    → 随访矩阵 {fu_df.shape} 已写出')

        # 混淆校正
        used = mat
        if not args.no_residualize:
            used = ph.residualize(mat, clinical)
            _save(used, args.output_dir / f'phenotype_{name}_residual.csv')

        # 组间比较
        res = ph.group_contrast(used, group)
        rej, q = fdr_bh(res.p)
        tbl = pd.DataFrame({
            'parcel': res.parcels, 'hedges_g': res.g, 'welch_t': res.t,
            'p': res.p, 'q': q, 'fdr_significant': rej,
        })
        _save(tbl, args.output_dir / f'contrast_{name}.csv')
        _save(pd.DataFrame({'parcel': res.parcels, 'effect': res.g}),
              args.output_dir / f'effect_map_{name}.csv')
        contrasts[name] = {
            'n_patient': res.n_patient, 'n_control': res.n_control,
            'n_fdr_significant': int(np.sum(rej)),
            'max_abs_g': float(np.nanmax(np.abs(res.g))) if np.isfinite(res.g).any() else None,
            'min_p': float(np.nanmin(res.p)) if np.isfinite(res.p).any() else None,
        }
        print(f'  {name}: n={res.n_patient}+{res.n_control}, '
              f'FDR 显著分区={contrasts[name]["n_fdr_significant"]}, '
              f'max|g|={contrasts[name]["max_abs_g"]}')

        # ---------- 3. 敏感性分析 ----------
        if res.n_patient >= 5 and res.n_control >= 5:
            nm = ph.n_matched_resampling(used, group, n_rep=args.n_matched_rep,
                                         seed=seed)
            extra[f'nmatched_{name}'] = {
                'n_rep': nm['n_rep'], 'n_small': nm['n_small'],
                'g_mean_median': float(np.nanmedian(nm['g_mean'])) if nm['n_rep'] else None,
                'sign_consistency_median': (float(np.nanmedian(nm['sign_consistency']))
                                            if nm['n_rep'] else None),
                'sign_consistency_min': (float(np.nanmin(nm['sign_consistency']))
                                         if nm['n_rep'] else None),
            }
            if nm['n_rep']:
                _save(pd.DataFrame({'parcel': res.parcels, 'g_mean': nm['g_mean'],
                                    'g_sd': nm['g_sd'],
                                    'sign_consistency': nm['sign_consistency']}),
                      args.output_dir / f'sensitivity_nmatched_{name}.csv')

        for cov in args.match_on:
            try:
                matched = ph.matched_subsample(used, clinical, match_on=cov, group=group)
            except KeyError as exc:
                print(f'  [warn] 匹配变量 {cov} 不可用：{exc}', file=sys.stderr)
                continue
            if len(matched) < 6:
                print(f'  [warn] {cov} 匹配后仅 {len(matched)} 例，跳过', file=sys.stderr)
                continue
            sub = used.loc[matched]
            r2 = ph.group_contrast(sub, group)
            # 与全样本效应量的空间一致性
            finite = np.isfinite(r2.g) & np.isfinite(res.g)
            r_spatial = (float(np.corrcoef(r2.g[finite], res.g[finite])[0, 1])
                         if finite.sum() > 3 else None)
            extra[f'{cov}_matched_{name}'] = {
                'n_matched': len(matched), 'n_patient': r2.n_patient,
                'n_control': r2.n_control,
                'spatial_r_with_full_sample': r_spatial,
                'max_abs_g': float(np.nanmax(np.abs(r2.g))) if np.isfinite(r2.g).any() else None,
            }
            _save(pd.DataFrame({'parcel': r2.parcels, 'hedges_g': r2.g,
                                'welch_t': r2.t, 'p': r2.p}),
                  args.output_dir / f'sensitivity_{cov}_matched_{name}.csv')

    # ---------- 4. 报告 ----------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'derivatives_dir': str(args.derivatives),
        'clinical': str(clinical_path),
        'n_parcels': n_parcels,
        'n_modes_A2': args.n_modes,
        'residualized': not args.no_residualize,
        'phenotypes': summaries,
        'contrasts': contrasts,
        'sensitivity': extra,
        'total_seconds': round(time.time() - t0, 1),
    }
    (args.output_dir / 'run_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# 影像表型提取与组间比较报告', '',
             f'- 派生数据：`{args.derivatives}`',
             f'- 临床表：`{clinical_path}`',
             f'- 分区数：{n_parcels}；混淆校正：{"否" if args.no_residualize else "是"}', '',
             '## 表型覆盖', '',
             '| 表型 | 被试数 | 分区数 | 有效值比例 | 全缺失分区 |', '|---|---|---|---|---|']
    for s in summaries:
        lines.append(f"| {s['phenotype']} | {s['n_subjects']} | {s['n_parcels']} | "
                     f"{s['frac_finite']:.3f} | {s['parcels_all_nan']} |")
    lines += ['', '## 组间比较（Welch + Hedges\' g，FDR-BH）', '',
              '| 表型 | 患者 | 对照 | FDR 显著分区 | max\\|g\\| | 最小 p |', '|---|---|---|---|---|---|']
    for name, c in contrasts.items():
        lines.append(f"| {name} | {c['n_patient']} | {c['n_control']} | "
                     f"{c['n_fdr_significant']} | {c['max_abs_g']} | {c['min_p']} |")
    lines += ['', '## 敏感性分析', '']
    if extra:
        lines += ['| 分析 | 指标 | 值 |', '|---|---|---|']
        for k, v in extra.items():
            for kk, vv in v.items():
                lines.append(f'| {k} | {kk} | {vv} |')
    else:
        lines.append('（未执行，或可用样本不足）')
    lines += ['', '> 说明：本报告全部数值由真实派生数据计算。表型矩阵中的 NaN 表示该分区/'
              '被试缺少数据，未做任何插补。']
    (args.output_dir / 'run_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'\n总耗时 {summary["total_seconds"]}s')
    print(f'报告：{args.output_dir / "run_report.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
