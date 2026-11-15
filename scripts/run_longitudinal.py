#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""纵向分析与临床分层（阶段九）运行入口。

对应研究方案 v2.1 第 5.8、5.9 节。

输入
----
- 术前与术后的表型矩阵（``<phenotypes-dir>/phenotype_<name>.csv``，
  行 = 被试 ID，含 ``sub-XXXfu`` 形式的随访行）
- 冻结临床表（含 ``Surgery_type``、``pain_reduction_pct``、``complete_relief``、
  ``First follow-up time (in days)`` 等）

输出（``<output-dir>``）
----------------------
- ``paired_<表型>.csv``              配对比较（均值变化 / t / p / Cohen's dz）
- ``stratified_<表型>_<术式>.csv``   按术式的配对比较
- ``interaction_<表型>.csv``         MVD vs 经皮的交互检验（对变化量的 Welch t）
- ``moderation_<表型>.csv``          疼痛缓解百分比的调节分析（每分区 beta/p/q）
- ``stratification_<表型>.json``     完全缓解 vs 未完全缓解的区分能力（AUC/PR-AUC/置换）
- ``mice_sensitivity_<表型>.json``   多重插补 vs 完整病例的敏感性对比
- ``run_summary.json`` / ``run_report.md``

**全部数值来自真实输入数据。** 缺少必要列或样本不足时明确报告并跳过，不生成占位值。
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
from tnwt import config_lite as cl                      # noqa: E402
from tnwt import longitudinal as lo                     # noqa: E402
from tnwt.enrichment import fdr_bh                      # noqa: E402

FOLLOWUP_SUFFIXES = ('fu', '_ses-2')


def _split_baseline_followup(matrix: pd.DataFrame, fu_ids: list[str]
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把混合矩阵拆成基线与随访两部分（行索引后缀识别）。"""
    is_fu = pd.Index([str(i).endswith(FOLLOWUP_SUFFIXES) for i in matrix.index])
    return matrix[~is_fu], matrix[is_fu]


def _load_clinical(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'找不到临床表：{path}')
    df = pd.read_csv(path, sep='\t')
    return df.drop_duplicates(subset='participant_id', keep='first')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='纵向分析与临床分层（阶段九）')
    ap.add_argument('--phenotypes-dir', type=Path, required=True,
                    help='run_phenotypes.py 的输出目录')
    ap.add_argument('--clinical', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=ROOT / 'config' / 'analysis.yaml')
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'longitudinal')
    ap.add_argument('--min-pairs', type=int, default=10)
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--n-imputations', type=int, default=5)
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args(argv)

    cfg = cl.load(args.config) if args.config.exists() else {}
    seed = args.seed if args.seed is not None else int(cl.dig(cfg, 'seeds.global', 0))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clin = _load_clinical(args.clinical)
    t0 = time.time()
    results: dict = {'phenotypes': {}, 'warnings': []}

    pheno_files = sorted(args.phenotypes_dir.glob('phenotype_*.csv'))
    pheno_files = [p for p in pheno_files
                   if '_residual' not in p.stem and not p.stem.endswith('_fu')]
    if not pheno_files:
        print(f'{args.phenotypes_dir} 下未找到 phenotype_*.csv', file=sys.stderr)
        return 2
    print(f'找到 {len(pheno_files)} 个基线表型矩阵')

    for pf in pheno_files:
        name = pf.stem.replace('phenotype_', '')
        mat = pd.read_csv(pf, index_col=0)
        fu_path = args.phenotypes_dir / f'phenotype_{name}_fu.csv'
        if fu_path.exists():
            # 优先使用 run_phenotypes.py 单独导出的随访矩阵
            base = mat
            fu = pd.read_csv(fu_path, index_col=0)
            src = '单独随访文件'
        else:
            # 兼容：随访行混在同一矩阵中
            base, fu = _split_baseline_followup(mat, list(clin['participant_id']))
            src = '同一矩阵内的随访行'
        if fu.empty:
            results['warnings'].append(
                f'{name}: 未找到随访数据（既无 *_fu.csv，也无同矩阵随访行），跳过纵向分析')
            print(f'  [skip] {name}: 无随访数据')
            continue
        fu_aligned = lo.align_followup(fu, base)
        if fu_aligned.empty:
            results['warnings'].append(f'{name}: 随访 ID 无法映射到基线 ID')
            print(f'  [skip] {name}: 随访 ID 无法映射')
            continue

        entry: dict = {'n_baseline': int(base.shape[0]), 'n_followup_rows': int(fu.shape[0]),
                       'n_pairs': int(fu_aligned.shape[0]), 'followup_source': src}
        print(f'  {name}: 基线 {base.shape[0]} / 随访行 {fu.shape[0]} / 配对 {fu_aligned.shape[0]}'
              f'（随访来源：{src}）')

        # ---------- 1. 配对比较 ----------
        pr = lo.paired_contrast(base, fu_aligned, min_pairs=args.min_pairs)
        if pr.n_pairs >= args.min_pairs:
            rej, q = fdr_bh(pr.p)
            pd.DataFrame({
                'parcel': pr.parcels, 'mean_change': pr.mean_change, 'paired_t': pr.t,
                'p': pr.p, 'q': q, 'cohens_dz': pr.cohens_dz,
                'n_increase': pr.n_positive, 'n_decrease': pr.n_negative,
                'fdr_significant': rej,
            }).to_csv(args.output_dir / f'paired_{name}.csv', index=False, encoding='utf-8')
            entry['paired'] = {
                'n_pairs': pr.n_pairs,
                'n_fdr_significant': int(np.sum(rej)),
                'max_abs_dz': float(np.nanmax(np.abs(pr.cohens_dz)))
                              if np.isfinite(pr.cohens_dz).any() else None,
            }
        else:
            entry['paired'] = {'note': f'配对数 {pr.n_pairs} 不足 {args.min_pairs}'}

        # ---------- 2. 术式分层与交互 ----------
        try:
            surgery = lo.stratify_by_surgery(clin, list(base.index))
            strat = lo.stratified_paired_contrast(base, fu_aligned, surgery,
                                                  min_pairs=max(3, args.min_pairs // 3))
            summ = {}
            for k, v in strat.items():
                if k == 'interaction':
                    p_int = v['p']
                    rej_i, q_i = fdr_bh(p_int)
                    pd.DataFrame({'parcel': list(base.columns), 'welch_t': v['welch_t'],
                                  'p': p_int, 'q': q_i, 'fdr_significant': rej_i}) \
                        .to_csv(args.output_dir / f'interaction_{name}.csv',
                                index=False, encoding='utf-8')
                    summ['interaction'] = {
                        'groups': v['groups'],
                        'n_fdr_significant': int(np.sum(rej_i)),
                        'min_p': float(np.nanmin(p_int)) if np.isfinite(p_int).any() else None,
                    }
                elif 'mean_change' in v:
                    pd.DataFrame({'parcel': list(base.columns), 'mean_change': v['mean_change'],
                                  'paired_t': v['t'], 'p': v['p']}) \
                        .to_csv(args.output_dir / f'stratified_{name}_{k}.csv',
                                index=False, encoding='utf-8')
                    summ[k] = {'n_pairs': v['n_pairs'],
                               'max_abs_change': float(np.nanmax(np.abs(v['mean_change'])))
                                                 if np.isfinite(v['mean_change']).any() else None}
                else:
                    summ[k] = {'n_pairs': v.get('n_pairs'), 'note': v.get('note')}
            entry['stratified'] = summ
        except KeyError as exc:
            results['warnings'].append(f'{name}: 术式分层失败（{exc}）')
            entry['stratified'] = {'note': str(exc)}

        # ---------- 3. 缓解百分比的调节分析 ----------
        moder_col = 'pain_reduction_pct'
        if moder_col in clin.columns:
            moder = clin.set_index('participant_id')[moder_col]
            change = fu_aligned - base.loc[fu_aligned.index]
            cov = None
            interval_col = 'First follow-up time (in days)'
            if interval_col in clin.columns:
                cov = clin.set_index('participant_id')[[interval_col]]
                cov = cov.rename(columns={interval_col: 'followup_interval_days'})
            mdf = lo.moderation_map(change, moder, covariates=cov)
            mdf.to_csv(args.output_dir / f'moderation_{name}.csv', index=False, encoding='utf-8')
            entry['moderation'] = {
                'moderator': moder_col,
                'n_with_moderator': int(np.isfinite(moder.reindex(fu_aligned.index)
                                                    .to_numpy(dtype=float)).sum()),
                'n_fdr_significant': int(mdf['fdr_significant'].sum()),
                'min_p': float(mdf['p'].min()) if mdf['p'].notna().any() else None,
                'covariate': 'followup_interval_days' if cov is not None else None,
            }

        # ---------- 4. 临床分层（完全缓解 vs 未完全缓解） ----------
        if 'complete_relief' in clin.columns:
            outcome = clin.set_index('participant_id')['complete_relief']
            # 用术前的 A1 做术前预测（若该表型是 A1 系），否则用配对变化量
            import pandas as pd_local  # noqa: F401
            for label, scores in (
                ('baseline_mean', base.loc[fu_aligned.index].mean(axis=1)),
                ('change_mean', (fu_aligned - base.loc[fu_aligned.index]).mean(axis=1)),
            ):
                sr = lo.stratify_outcome(scores, outcome, n_splits=args.n_splits,
                                         n_perm=args.n_perm, seed=seed)
                entry.setdefault('stratification', {})[label] = {
                    'auc': sr.auc, 'pr_auc': sr.pr_auc,
                    'n_positive': sr.n_positive, 'n_negative': sr.n_negative,
                    'auc_perm_p': sr.auc_perm_p, 'n_perm': sr.n_perm,
                }

        # ---------- 5. 多重插补敏感性 ----------
        wide = pd.concat([base.loc[fu_aligned.index].add_suffix('_pre'),
                          fu_aligned.add_suffix('_post')], axis=1)
        n_missing = int(wide.isna().to_numpy().sum())
        if n_missing > 0:
            sets = lo.mice_impute(wide, n_imputations=args.n_imputations, seed=seed)
            est, var = [], []
            for s in sets:
                pre = s[[c for c in s.columns if c.endswith('_pre')]].to_numpy(dtype=float)
                post = s[[c for c in s.columns if c.endswith('_post')]].to_numpy(dtype=float)
                d = (post - pre).mean(axis=1)
                est.append(float(d.mean()))
                var.append(float(d.var(ddof=1) / len(d)) if len(d) > 1 else np.nan)
            pooled = lo.rubin_pool(est, var)
            complete = (fu_aligned - base.loc[fu_aligned.index]).mean(axis=1).dropna()
            entry['mice_sensitivity'] = {
                'n_missing_cells': n_missing,
                'complete_case_mean_change': float(complete.mean()) if len(complete) else None,
                'complete_case_n': int(len(complete)),
                'pooled': pooled,
            }
            (args.output_dir / f'mice_sensitivity_{name}.json').write_text(
                json.dumps(entry['mice_sensitivity'], ensure_ascii=False, indent=2,
                           default=float), encoding='utf-8')
        else:
            entry['mice_sensitivity'] = {'note': '无缺失，无需插补'}

        results['phenotypes'][name] = entry

    results['total_seconds'] = round(time.time() - t0, 1)
    (args.output_dir / 'run_summary.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding='utf-8')

    # ---------- 报告 ----------
    lines = ['# 纵向分析与临床分层报告', '',
             f'- 表型目录：`{args.phenotypes_dir}`',
             f'- 临床表：`{args.clinical}`',
             f'- 最小配对数：{args.min_pairs}；置换次数：{args.n_perm}', '',
             '## 配对比较（术后 − 术前）', '',
             '| 表型 | 配对数 | FDR 显著分区 | max\\|dz\\| |', '|---|---|---|---|']
    for name, e in results['phenotypes'].items():
        p = e.get('paired', {})
        lines.append(f"| {name} | {p.get('n_pairs', '—')} | "
                     f"{p.get('n_fdr_significant', '—')} | {p.get('max_abs_dz', '—')} |")
    lines += ['', '## 术式分层与交互', '']
    for name, e in results['phenotypes'].items():
        s = e.get('stratified', {})
        lines.append(f'### {name}')
        for k, v in s.items():
            lines.append(f'- {k}: {json.dumps(v, ensure_ascii=False, default=float)}')
        lines.append('')
    lines += ['## 疼痛缓解的调节分析', '']
    for name, e in results['phenotypes'].items():
        m = e.get('moderation')
        if m:
            lines.append(f"- {name}: 可用 n={m['n_with_moderator']}, "
                         f"FDR 显著={m['n_fdr_significant']}, min p={m['min_p']}, "
                         f"协变量={m['covariate']}")
    lines += ['', '## 临床分层（完全缓解 vs 未完全缓解）', '',
              '| 表型 | 指标 | AUC | PR-AUC | 正/负例 | 置换 p |', '|---|---|---|---|---|---|']
    for name, e in results['phenotypes'].items():
        st = e.get('stratification', {})
        for label, v in st.items():
            lines.append(f"| {name} | {label} | {v['auc']:.3f} | {v['pr_auc']:.3f} | "
                         f"{v['n_positive']}/{v['n_negative']} | {v['auc_perm_p']:.4f} |")
    lines += ['', '## 缺失数据敏感性（多重插补，Rubin 合并）', '']
    for name, e in results['phenotypes'].items():
        ms = e.get('mice_sensitivity', {})
        lines.append(f"- {name}: {json.dumps(ms, ensure_ascii=False, default=float)}")
    if results['warnings']:
        lines += ['', '## 警告', '']
        for w in results['warnings']:
            lines.append(f'- {w}')
    lines += ['', '> 全部数值来自真实输入数据；配对分析只使用同时具备术前与术后数据的被试。']
    (args.output_dir / 'run_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'\n总耗时 {results["total_seconds"]}s')
    print(f'报告：{args.output_dir / "run_report.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
