#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""影像转录组关联与跨模态收敛（阶段六–八）运行入口。

这是此前缺失的"胶水"脚本：把 AHBA 表达矩阵（阶段三输出）与影像效应量图谱
（阶段四–五输出）接起来，产出后续出图与论文所需的全部文件。

输入
----
- ``<ahba-dir>/expression_<tag>.npy``  分区 × 基因 表达矩阵（run_ahba.py 输出）
- ``<ahba-dir>/genes_<tag>.json``      基因名
- ``<ahba-dir>/lodo_<tag>.npy``        可选，LODO 拆分（供稳健性分析）
- ``<phenotypes-dir>/effect_map_*.csv`` 各模态效应量图谱

输出（``<out-dir>``）
--------------------
- ``pls_weights_<modal>.csv``        每模态 PLS1 权重（gene, z）
- ``pls_stability_<modal>.csv``     基因入选率（gene, z, selection_freq）
- ``pls_summary.json``              显著性、解释方差、候选基因数、敏感性
- ``convergence.json``              收敛判据与结论（方案 4.3）
- ``convergence_report.md``         可读报告
- ``robustness.json``               坐标/归一化/阈值/LODO 的重叠率（方案 5.7 第一级）
- ``external_consistency.json``     与外部权重向量的一致性（需 --external-weights）

设计原则
--------
**只使用真实输入数据计算。** 输入缺失时报错退出并说明缺什么，不生成占位结果。
富集分析由 ``scripts/run_enrichment.py`` 独立执行（输入为本脚本的候选基因列表）。
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
from tnwt import config_lite as cl                    # noqa: E402
from tnwt import convergence as cv                    # noqa: E402
from tnwt.stats_imaging import pls_regression, pls_stability  # noqa: E402

MODALITY_FILES = {
    'A1_streamlines': 'effect_map_A1_streamlines.csv',
    'A1_fa': 'effect_map_A1_fa.csv',
    'A2_coupling': 'effect_map_A2_coupling.csv',
    'A3_gwc': 'effect_map_A3_gwc.csv',
    # 原始 T1w/T2w：仅作"标定前"对照，用于证明缩放缺陷会产生伪关联。
    # ⚠️ 其结果不得作为发现，也不得进入收敛判定。
    'A3_t1t2': 'effect_map_A3_t1t2.csv',
    # 白质内参标定版：唯一可用于统计与论文结论的 T1w/T2w 版本。
    'A3_t1t2_std': 'effect_map_A3_t1t2_std.csv',
}


def _load_expression(ahba_dir: Path, tag: str | None = None
                     ) -> tuple[np.ndarray, list[str], Path]:
    """载入表达矩阵与基因名（自动挑选 tag）。"""
    npy = sorted(ahba_dir.glob('expression_*.npy'))
    if not npy:
        raise FileNotFoundError(
            f'{ahba_dir} 下没有 expression_*.npy；请先运行 scripts/run_ahba.py')
    if tag:
        cand = [p for p in npy if tag in p.stem]
        if not cand:
            raise FileNotFoundError(f'未找到与 tag={tag} 匹配的表达矩阵；可选：'
                                    f'{[p.stem for p in npy]}')
        chosen = cand[0]
    else:
        # 优先 primary_*
        prim = [p for p in npy if 'primary' in p.stem]
        chosen = prim[0] if prim else npy[0]
    expr = np.load(chosen)
    gene_file = chosen.with_name(chosen.name.replace('expression_', 'genes_')).with_suffix('.json')
    if not gene_file.exists():
        raise FileNotFoundError(f'缺少基因名文件：{gene_file}')
    genes = json.loads(gene_file.read_text(encoding='utf-8'))
    if len(genes) != expr.shape[1]:
        raise ValueError(f'基因名数 {len(genes)} 与表达矩阵列数 {expr.shape[1]} 不一致')
    return np.asarray(expr, dtype=float), list(genes), chosen


def _load_effect_map(pheno_dir: Path, fname: str) -> np.ndarray | None:
    p = pheno_dir / fname
    if not p.exists():
        return None
    df = pd.read_csv(p)
    col = 'effect' if 'effect' in df.columns else df.columns[-1]
    return df[col].to_numpy(dtype=float)


def _load_spins(coords_path: Path, n_perm: int, seed: int) -> np.ndarray | None:
    if not coords_path.exists():
        return None
    df = pd.read_csv(coords_path)
    cols = {c.strip().upper(): c for c in df.columns}
    x = cols.get('R') or cols.get('X')
    y = cols.get('A') or cols.get('Y')
    z = cols.get('S') or cols.get('Z')
    if not (x and y and z):
        return None
    coords = df[[x, y, z]].to_numpy(dtype=float)
    from tnwt.spin import gen_spinsamples
    return gen_spinsamples(coords, n_perm=n_perm, seed=seed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='影像转录组关联与跨模态收敛（阶段六–八）')
    ap.add_argument('--ahba-dir', type=Path, required=True,
                    help='run_ahba.py 的输出目录（含 expression_*.npy）')
    ap.add_argument('--phenotypes-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'data' / 'transcriptome')
    ap.add_argument('--config', type=Path, default=ROOT / 'config' / 'analysis.yaml')
    ap.add_argument('--atlas-centroids', type=Path,
                    default=ROOT / 'data' / 'atlas'
                            / 'Schaefer2018_200Parcels_7Networks_centroids_MNI.csv')
    ap.add_argument('--expression-tag', default=None,
                    help='指定用哪个表达矩阵（默认优先 primary_*）')
    ap.add_argument('--modalities', nargs='*', default=None,
                    help='参与分析的模态（默认全部存在的 effect_map_*.csv）')
    ap.add_argument('--n-stability', type=int, default=None,
                    help='稳定性重采样次数（默认取配置 pls.stability.n_resamples）')
    ap.add_argument('--n-perm', type=int, default=None, help='spin 置换次数')
    ap.add_argument('--external-weights', type=Path, default=None,
                    help='外部基因权重 CSV（两列：gene, z），用于第二级验证')
    ap.add_argument('--spin-mode', choices=('fixed', 'refit', 'both'),
                    default='refit',
                    help='空间置换检验方案。refit（默认）= 每次置换重新拟合 PLS，'
                         '是投稿应采用的口径；fixed = 固定权重（快，但对阴性模态'
                         'p 值偏小）；both = 两者都算并同时报告')
    ap.add_argument('--refit-perm', type=int, default=None,
                    help='refit 方案的置换次数（默认与 --n-perm 相同）。'
                         'refit 每次置换需一次完整 PLS 拟合，耗时约为 fixed 的 400 倍')
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args(argv)

    cfg = cl.load(args.config) if args.config.exists() else {}
    seed = args.seed if args.seed is not None else int(cl.dig(cfg, 'seeds.global', 0))
    n_perm = args.n_perm or int(cl.dig(cfg, 'spin.n_permutations', 5000))
    n_stab = args.n_stability or int(cl.dig(cfg, 'pls.stability.n_resamples', 1000))
    min_freq = float(cl.dig(cfg, 'pls.stability.candidate_gene_min_selection_freq', 0.8))
    threshold = float(cl.dig(cfg, 'pls.gene_threshold_primary', 'abs_z_gt_2').replace(
        'abs_z_gt_', '')) if isinstance(cl.dig(cfg, 'pls.gene_threshold_primary'), str) else 2.0
    refit_perm = args.refit_perm or n_perm
    spin_mode = args.spin_mode
    criteria = cv.ConvergenceCriteria.from_config(cfg)

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    expr, genes, expr_path = _load_expression(args.ahba_dir, args.expression_tag)
    print(f'表达矩阵：{expr_path.name} → {expr.shape[0]} 分区 × {expr.shape[1]} 基因')

    # ---------- 载入效应量图谱 ----------
    names = args.modalities or [k for k, f in MODALITY_FILES.items()
                                if (args.phenotypes_dir / f).exists()]
    maps: dict[str, np.ndarray] = {}
    for name in names:
        v = _load_effect_map(args.phenotypes_dir, MODALITY_FILES[name])
        if v is None:
            print(f'  [skip] {name}：缺少 {MODALITY_FILES[name]}')
            continue
        if len(v) != expr.shape[0]:
            print(f'  [skip] {name}：分区数 {len(v)} 与表达矩阵 {expr.shape[0]} 不一致')
            continue
        maps[name] = v
    if not maps:
        print('没有任何可用模态的效应量图谱；请先运行 scripts/run_phenotypes.py',
              file=sys.stderr)
        return 2
    print(f'参与模态：{list(maps)}')

    spins_per_modality: dict[str, np.ndarray] = {}
    spins = _load_spins(args.atlas_centroids, n_perm, seed)
    if spins is None:
        print(f'  [warn] 无法从 {args.atlas_centroids} 读取质心，spin 检验将跳过',
              file=sys.stderr)
    else:
        print(f'旋转置换：{spins.shape[0]} 次（seed={seed}）')
        for name in maps:
            spins_per_modality[name] = spins

    # ---------- 各模态 PLS + 稳定性 ----------
    summary: dict = {'expression_source': str(expr_path), 'n_parcels': int(expr.shape[0]),
                     'n_genes': int(expr.shape[1]), 'n_perm': n_perm,
                     'n_stability_resamples': n_stab,
                     'candidate_gene_min_selection_freq': min_freq,
                     'spin_mode': spin_mode,
                     'n_perm_refit': refit_perm,
                     'modalities': {}}
    candidate_genes: dict[str, list[str]] = {}
    spin_p_values: dict[str, float] = {}
    stable_weights: dict[str, np.ndarray] = {}

    for name, y in maps.items():
        m = np.isfinite(y) & np.all(np.isfinite(expr), axis=1)
        if m.sum() < 20:
            print(f'  [skip] {name}：有效分区仅 {int(m.sum())}')
            continue
        X = expr[m]
        yv = np.asarray(y, dtype=float)[m]
        r = pls_regression(X, yv, n_components=1)
        z = r.x_weights[:, 0]
        raw = r.x_weights_raw[:, 0]
        pd.DataFrame({'gene': genes, 'z': z, 'weight': raw}) \
            .to_csv(args.out_dir / f'pls_weights_{name}.csv', index=False, encoding='utf-8')

        print(f'  {name}: PLS1 var_y={r.var_y[0]:.3f}，开始 {n_stab} 次稳定性重采样…')
        stab = pls_stability(X, yv, raw, n_resamples=n_stab, seed=seed,
                             mode='bootstrap', top_frac=0.05)
        pd.DataFrame({'gene': genes, 'z': z,
                      'selection_freq': stab['selection_freq'],
                      'sign_consistency': stab['sign_consistency']}) \
            .to_csv(args.out_dir / f'pls_stability_{name}.csv', index=False, encoding='utf-8')

        cand_mask = stab['selection_freq'] >= min_freq
        cand = [g for g, k in zip(genes, cand_mask) if k]
        candidate_genes[name] = cand
        stable_weights[name] = z
        (args.out_dir / f'candidate_genes_{name}.json').write_text(
            json.dumps({'modal': name, 'min_selection_freq': min_freq,
                        'n_candidates': len(cand), 'genes': cand}, ensure_ascii=False,
                       indent=2), encoding='utf-8')

        # ---------- 空间置换检验 ----------
        # 约定：统一旋转**影像图谱** y（与 spin_test_refit 的内部约定一致）。
        spins = spins_per_modality.get(name)
        spin_info: dict = {}
        if spins is not None:
            x_full = np.full(len(np.asarray(y, dtype=float)), np.nan, dtype=float)
            x_full[m] = X @ raw
            y_full = np.asarray(y, dtype=float)

            def _corr(a, b):
                sa, sb = a.std(ddof=0), b.std(ddof=0)
                if sa == 0 or sb == 0:
                    return float('nan')
                return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))

            if spin_mode in ('fixed', 'both'):
                from tnwt.spin import spin_test as _st
                fa = _st(y_full, spins, x_full)
                spin_info['fixed'] = {'p': fa['p_two_sided'],
                                      'observed': fa['observed'],
                                      'null_mean': float(fa['null'].mean()),
                                      'null_sd': float(fa['null'].std(ddof=1)),
                                      'n_perm': fa['n_perm']}
            if spin_mode in ('refit', 'both'):
                from tnwt.spin import spin_test_refit as _srf

                def _stat(yv, _m=m, _X=X):
                    yy = yv[_m]
                    # ⚠️ 旋转会**移动** NaN 的位置：`_m` 是按**原始** y 的有效性选的，
                    # 旋转后某些原本有效的位置可能变成 NaN（反之亦然）。
                    # 而 `pls_regression` 内部会再按 `isfinite(X) & isfinite(Y)`
                    # 静默丢掉这些行（见 stats_imaging.pls_regression），
                    # 导致 x_scores 比 yy 少几行 —— 实测 A2_coupling 是 191 vs 192，
                    # 直接抛 `operands could not be broadcast together`。
                    # 因此这里必须显式对齐：用同一个有限性掩码同时作用于 X 和 yy。
                    ok = np.isfinite(yy)
                    if not ok.all():
                        yy = yy[ok]
                        _X = _X[ok]
                    rr = pls_regression(_X, yy, n_components=1)
                    return _corr(rr.x_scores[:, 0], yy)

                rf = _srf(_stat, y_full, spins[:refit_perm])
                spin_info['refit'] = {'p': rf['p_two_sided'],
                                      'observed': rf['observed'],
                                      'null_mean': rf['null_mean'],
                                      'null_sd': rf['null_sd'],
                                      'n_perm': rf['n_perm']}
            chosen = spin_info.get(spin_mode) or spin_info.get('refit') \
                or spin_info.get('fixed')
            if chosen:
                spin_p_values[name] = chosen['p']
            print(f"    → spin({spin_mode}): "
                  + '；'.join(f"{k} p={v['p']:.5f}" for k, v in spin_info.items()))

        summary['modalities'][name] = {
            'n_used_parcels': int(m.sum()),
            'var_y': float(r.var_y[0]),
            'var_x': float(r.var_x[0]),
            'stability_n_valid': int(stab['n_valid']),
            'stability_cosine_mean': (float(np.mean(stab['cosine']))
                                      if len(stab['cosine']) else None),
            'n_abs_z_gt_2': int(np.sum(np.abs(z) >= 2.0)),
            'n_candidate_genes': len(cand),
            'spin': spin_info,
        }
        print(f'    → 候选基因 {len(cand)} 个（入选率 ≥ {min_freq}）；'
              f'|Z|≥2 者 {summary["modalities"][name]["n_abs_z_gt_2"]} 个')

    if not summary['modalities']:
        print('所有模态均不可分析', file=sys.stderr)
        return 2

    # ---------- 收敛检验 ----------
    # D11：组间图谱未通过 FDR 的模态不参与收敛判定。
    # 判定依据是阶段四–五的 contrast_<模态>.csv 中 fdr_significant 列；
    # 若该文件缺失，则不剔除任何模态（并在报告中标注无法判定）。
    print('跨模态收敛检验…')
    excluded: dict[str, str] = {}
    contrast_info: dict[str, dict] = {}
    for name in maps:
        cf = args.phenotypes_dir / f'contrast_{name}.csv'
        if not cf.exists():
            continue
        try:
            cdf = pd.read_csv(cf)
        except Exception:  # noqa: BLE001
            continue
        if 'fdr_significant' not in cdf.columns:
            continue
        n_sig = int(cdf['fdr_significant'].astype(bool).sum())
        contrast_info[name] = {'n_fdr_significant': n_sig,
                               'n_parcels': int(len(cdf))}
        if n_sig == 0:
            excluded[name] = (f'组间效应图谱 FDR 校正后无显著分区（0/{len(cdf)}），'
                              '按 D11 不参与收敛判定')
    if contrast_info:
        parts = [f"{k}={v['n_fdr_significant']}/{v['n_parcels']}"
                 for k, v in contrast_info.items()]
        print('  组间结果（FDR 显著分区/总分区）：' + '；'.join(parts))
    if excluded:
        print(f'  按 D11 剔除的模态：{list(excluded)}')

    conv = cv.convergence_test(maps, expr, genes, criteria=criteria,
                              spins_per_modality=spins_per_modality or None,
                              excluded_modalities=excluded,
                              spin_p_precomputed=spin_p_values or None,
                              spin_mode=spin_mode)
    (args.out_dir / 'convergence.json').write_text(
        json.dumps(conv, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    (args.out_dir / 'convergence_report.md').write_text(
        cv.render_convergence_report(conv), encoding='utf-8')
    print(f"  → 可分离={conv.get('separable')}；最小权重余弦="
          f"{conv.get('min_pairwise_cosine'):+.3f}；结论={'收敛' if conv.get('converged') else '不收敛'}")

    # ---------- 稳健性：LODO ----------
    robustness: dict = {}
    lodo_file = next(iter(sorted(args.ahba_dir.glob('lodo_*.npy'))), None)
    if lodo_file is not None and stable_weights:
        stack = np.load(lodo_file)          # (n_donors, parcels, genes)
        if stack.ndim == 3 and stack.shape[1:] == expr.shape:
            ref_name = next(iter(stable_weights))
            variants: dict[str, np.ndarray] = {}
            for i in range(stack.shape[0]):
                m = np.isfinite(maps[ref_name]) & np.all(np.isfinite(stack[i]), axis=1)
                if m.sum() < 20:
                    continue
                r = pls_regression(stack[i][m], np.asarray(maps[ref_name], dtype=float)[m],
                                   n_components=1)
                variants[f'lodo_donor{i+1}'] = r.x_weights[:, 0]
            robustness['lodo'] = cv.robustness_matrix(stable_weights[ref_name], genes,
                                                      variants, threshold=2.0)
            print(f"  → LODO 稳健性（{ref_name}）：最小重叠系数 "
                  f"{robustness['lodo']['min_overlap_coefficient']:.3f}")
    else:
        robustness['lodo'] = {'note': '缺少 lodo_*.npy 或表达矩阵，未做 LODO 稳定性'}

    # ---------- 二级验证：外部权重一致性 ----------
    if args.external_weights is not None and args.external_weights.exists():
        ext = pd.read_csv(args.external_weights)
        gcol = 'gene' if 'gene' in ext.columns else ext.columns[0]
        zcol = 'z' if 'z' in ext.columns else ext.columns[-1]
        ext_map = dict(zip(ext[gcol].astype(str), ext[zcol].astype(float)))
        cons = {}
        for name, z in stable_weights.items():
            # 先按基因对齐，再调用一致性检验
            common = [g for g in genes if g in ext_map]
            if len(common) < 10:
                cons[name] = {'n_common_genes': len(common),
                              'note': '共同基因不足 10 个'}
                continue
            idx = [genes.index(g) for g in common]
            cons[name] = cv.external_consistency(
                np.array([z[i] for i in idx]),
                np.array([ext_map[g] for g in common]), common)
        robustness['external_consistency'] = cons
        (args.out_dir / 'external_consistency.json').write_text(
            json.dumps(cons, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    else:
        robustness['external_consistency'] = {
            'note': '未提供 --external-weights；方案 5.7 第二级验证待补'}

    (args.out_dir / 'robustness.json').write_text(
        json.dumps(robustness, ensure_ascii=False, indent=2, default=float),
        encoding='utf-8')

    # ---------- 汇总 ----------
    summary['converged'] = conv.get('converged')
    summary['max_pairwise_abs_r'] = conv.get('max_pairwise_abs_r')
    summary['min_pairwise_cosine'] = conv.get('min_pairwise_cosine')
    summary['separable'] = conv.get('separable')
    # 兼容 write_paper.py 期望的字段名
    summary['p_spin'] = (min(conv['spin_p'].values()) if conv.get('spin_p') else None)
    summary['var_y'] = max((v['var_y'] for v in summary['modalities'].values()),
                           default=None)
    summary['n_candidate_genes'] = {k: len(v) for k, v in candidate_genes.items()}
    summary['candidate_genes_by_modality'] = candidate_genes
    summary['candidate_genes_union'] = sorted({g for v in candidate_genes.values()
                                               for g in v})
    summary['n_candidate_genes_union'] = len(summary['candidate_genes_union'])
    summary['total_seconds'] = round(time.time() - t0, 1)
    (args.out_dir / 'pls_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding='utf-8')

    # 富集分析的输入：各模态候选基因
    (args.out_dir / 'enrichment_input.json').write_text(
        json.dumps({'candidate_genes_by_modality': candidate_genes,
                    'union': summary['candidate_genes_union'],
                    'background_note': '富集分析请使用 AHBA 检出基因 ∩ 注释可注释基因作为背景'},
                   ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'\n总耗时 {summary["total_seconds"]}s')
    print(f'候选基因（并集）{summary["n_candidate_genes_union"]} 个')
    print(f'输出：{args.out_dir}')
    print('下一步：python scripts/run_enrichment.py --transcriptome '
          f'{args.out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
