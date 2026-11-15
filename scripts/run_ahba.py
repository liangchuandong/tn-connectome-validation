#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AHBA 处理运行入口（阶段四：abagen 等价管线）。

数据准备
--------
AHBA 微阵列数据约 4GB，需从 https://human.brain-map.org/static/download 获取，
或用 abagen（若可安装）执行：

    import abagen
    files = abagen.fetch_microarray(donors='all', verbose=0)

目录结构要求（与 abagen 一致）::

    <ahba_dir>/
    ├── normalized_microarray_donor9861/
    │   ├── MicroarrayExpression.csv
    │   ├── Probes.csv
    │   ├── SampleAnnot.csv
    │   ├── PACall.csv          # 可选但强烈建议
    │   └── Ontology.csv
    ├── normalized_microarray_donor10021/
    └── ...

CIC 坐标集（方案 D2 的主分析口径）
----------------------------------
默认坐标来自各供体 SampleAnnot.csv 的 MNI 列。CIC 坐标集可显著提高样本解剖定位精度，
可通过 ``--cic-coords-dir`` 指定由 CoBrALab 提供的坐标文件目录，文件名形如
``H0351.1009_SampleAnnot_RAS_mni_nonlin.csv``。

用法
----
  # 用 CIC 坐标
  python scripts/run_ahba.py --ahba-dir <dir> --cic-coords-dir <dir> --output-dir data/ahba

  # 敏感性分析：全部坐标集 × 归一化 × 半径
  python scripts/run_ahba.py --ahba-dir <dir> --sensitivity

  # 无真实数据时用合成供体自检全流程
  python scripts/run_ahba.py --synthetic --output-dir build/ahba_synth
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tnwt import ahba  # noqa: E402
from tnwt import config_lite as cl  # noqa: E402

# CIC 坐标文件名（CoBrALab/AllenHumanGeneMNI）中供体编号 → 我们的供体目录名
CIC_FILENAME_TEMPLATE = 'H0351.{donor}_SampleAnnot_RAS_mni_nonlin.csv'

DONOR_TO_CIC = {
    '9861': 'H0351.2001',
    '10021': 'H0351.2002',
    '12876': 'H0351.1009',
    '14380': 'H0351.1012',
    '15496': 'H0351.1015',
    '15697': 'H0351.1016',
}


def load_coords_from_manifest(path: Path) -> np.ndarray:
    """读取冻结的 Schaefer200 质心表（scripts/fetch_atlas.py 的产物）。"""
    rows = list(csv.DictReader(path.open('r', encoding='utf-8')))
    cols = rows[0].keys()
    xcol = next((c for c in cols if c.strip().upper() in ('R', 'X', 'MNI_X')), None)
    ycol = next((c for c in cols if c.strip().upper() in ('A', 'Y', 'MNI_Y')), None)
    zcol = next((c for c in cols if c.strip().upper() in ('S', 'Z', 'MNI_Z')), None)
    if not (xcol and ycol and zcol):
        raise ValueError(f'质心表缺少 R/A/S 列，实际列：{list(cols)}')
    return np.array([[float(r[xcol]), float(r[ycol]), float(r[zcol])] for r in rows], dtype=float)


def apply_cic_coords(donors, cic_dir: Path, verbose: bool = True):
    """把 SampleAnnot 坐标替换为 CIC 坐标集。

    匹配策略：按 ``sample_id`` 对齐；CIC 文件的列名不一致时，取前三列可解析为数值的列。
    无法匹配的供体保持原坐标并在报告中记录。
    """
    applied, skipped = [], []
    for dd in donors:
        d = ahba.read_donor(Path(dd))
        short = d.donor_id
        # 我们的目录名是 9861/10021/...，CIC 用 H0351.1009 这类完整编号
        # Donor ids (9861/10021/...) do NOT appear in CIC filenames, which use
        # H0351 numbers.  Consult the explicit map first, then fall back.
        cic_name = DONOR_TO_CIC.get(str(short))
        candidates = []
        if cic_name:
            candidates = list(cic_dir.glob(f'{cic_name}*mni*.csv'))
        if not candidates:
            candidates = list(cic_dir.glob(f'*{short}*mni*.csv'))
        if not candidates:
            skipped.append((short, '未找到匹配的 CIC 文件'))
            continue
        try:
            reordered = _read_cic_coords(candidates[0], d.sample_ids)
        except Exception as exc:  # noqa: BLE001
            skipped.append((short, f'解析失败：{exc}'))
            continue
        if reordered is None:
            skipped.append((short, 'sample_id 无法对齐'))
            continue
        applied.append(short)
        yield dd, reordered
    if verbose:
        print(f'  CIC 坐标已应用：{len(applied)} 个供体 {applied}')
        for sid, why in skipped:
            print(f'  [warn] 供体 {sid} 未使用 CIC 坐标：{why}')


def _read_cic_coords(path: Path, sample_ids: list[str]) -> np.ndarray | None:
    """Read CIC coordinates aligned to ``sample_ids``; returns (n, 3) or None.

    The real CIC files (CoBrALab/AllenHumanGeneMNI, ``transformed-points/``)
    carry **three numeric columns and no sample id**::

        mni_nlin_x,mni_nlin_y,mni_nlin_z

    so we align **by position** (CIC row order matches SampleAnnot order) and
    require the row count to equal the sample count.  If a file does happen to
    provide a sample id column we still match on it, for compatibility with
    other coordinate sources.
    """
    rows = list(csv.DictReader(path.open('r', encoding='utf-8')))
    if not rows:
        return None
    cols = list(rows[0].keys())
    idcol = next((c for c in cols
                  if str(c).strip().lower() in ('sample_id', 'sampleid', 'sample')), None)
    numeric = []
    for c in cols:
        if c == idcol:
            continue
        try:
            float(rows[0][c])
            numeric.append(c)
        except (TypeError, ValueError):
            continue
    if len(numeric) < 3:
        return None

    def _row(r):
        return [float(r[numeric[0]]), float(r[numeric[1]]), float(r[numeric[2]])]

    # Case A: an id column exists -> match on it
    if idcol is not None:
        xyz = {str(r[idcol]): _row(r) for r in rows}
        if all(s in xyz for s in sample_ids):
            return np.array([xyz[s] for s in sample_ids], dtype=float)

    # Case B: no id column (the real CIC layout) -> align by position
    if len(rows) != len(sample_ids):
        return None
    return np.array([_row(r) for r in rows], dtype=float)

def build_synthetic(root: Path, n_donors: int = 3, n_parcels: int = 30):
    coords = np.random.default_rng(2026).normal(0, 30, size=(n_parcels, 3))
    donors = [ahba.make_synthetic_donor(root, str(9861 + i * 100), n_samples=80,
                                        n_genes=60, probes_per_gene=2,
                                        seed=i, coords=coords) for i in range(n_donors)]
    return donors, coords


def summarize(expr: np.ndarray, genes: list[str], report, tag: str) -> dict:
    finite = np.isfinite(expr)
    return {
        'tag': tag,
        'n_parcels': int(expr.shape[0]),
        'n_genes': len(genes),
        'n_parcels_with_data': report.n_parcels_with_data,
        'coverage_min': float(np.min(report.parcel_donor_coverage)) if len(report.parcel_donor_coverage) else None,
        'coverage_median': float(np.median(report.parcel_donor_coverage)) if len(report.parcel_donor_coverage) else None,
        'frac_finite': float(finite.mean()) if expr.size else 0.0,
        'value_median': float(np.nanmedian(expr)) if finite.any() else None,
        'value_iqr': [float(np.nanpercentile(expr, 25)) if finite.any() else None,
                      float(np.nanpercentile(expr, 75)) if finite.any() else None],
        'n_probes_raw': report.n_probes_raw,
        'n_probes_filtered': report.n_probes_filtered,
        'notes': report.notes,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='AHBA 处理管线（abagen 等价实现）')
    ap.add_argument('--ahba-dir', type=Path, default=None,
                    help='AHBA 微阵列数据目录（含 normalized_microarray_donor* 子目录）')
    ap.add_argument('--cic-coords-dir', type=Path, default=None,
                    help='CoBrALab CIC 坐标目录（方案 D2 主分析口径）')
    ap.add_argument('--atlas-centroids', type=Path,
                    default=ROOT / 'data' / 'atlas' / 'Schaefer2018_200Parcels_7Networks_centroids_MNI.csv')
    ap.add_argument('--config', type=Path, default=ROOT / 'config' / 'analysis.yaml')
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'data' / 'ahba')
    ap.add_argument('--radius', type=float, default=2.0)
    ap.add_argument('--atlas', type=Path,
                    default=ROOT / 'data' / 'atlas' /
                    'Schaefer2018_200Parcels_7Networks_MNI_1mm.nii.gz',
                    help='Schaefer 体积图集；--match-method voxel 时必需')
    ap.add_argument('--match-method', choices=('voxel', 'centroid'), default='voxel',
                    help='样本->分区匹配方式。voxel(默认)=按图集体素归属并按 '
                         '--radius 膨胀；centroid=旧行为（质心+半径，实测覆盖率极低）')
    ap.add_argument('--donor-coverage-min', type=int, default=2)
    ap.add_argument('--synthetic', action='store_true', help='用合成供体自检全流程')
    ap.add_argument('--sensitivity', action='store_true', help='跑坐标集/归一化/半径敏感性')
    args = ap.parse_args(argv)

    cfg = cl.load(args.config) if args.config.exists() else {}
    ahba_cfg = cl.dig(cfg, 'ahba', {}) or {}
    primary_norm = (ahba_cfg.get('normalization') or {}).get('primary', 'sigmoid')
    variants = (ahba_cfg.get('normalization') or {}).get('sensitivity',
                                                         ['robust_sigmoid', 'zscore'])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.synthetic:
        print('=== 合成供体自检 ===')
        donors, coords = build_synthetic(args.output_dir / 'synthetic_donors')
        print(f'供体 {len(donors)} 个，分区 {len(coords)} 个')
    else:
        if args.ahba_dir is None or not Path(args.ahba_dir).is_dir():
            print('需要 --ahba-dir 指向 AHBA 数据目录（或用 --synthetic 自检）', file=sys.stderr)
            return 2
        donors = sorted(p for p in Path(args.ahba_dir).glob('normalized_microarray_donor*')
                        if p.is_dir())
        if not donors:
            print(f'{args.ahba_dir} 下未找到 normalized_microarray_donor* 目录', file=sys.stderr)
            return 2
        if not args.atlas_centroids.exists():
            print(f'缺少图集质心表：{args.atlas_centroids}；请先运行 scripts/fetch_atlas.py',
                  file=sys.stderr)
            return 2
        coords = load_coords_from_manifest(args.atlas_centroids)
        print(f'供体 {len(donors)} 个，分区 {len(coords)} 个')

    results: list[dict] = []

    MATCHER = None
    if args.match_method == 'voxel':
        try:
            MATCHER = ahba.make_voxel_matcher(args.atlas, tolerance=args.radius)
            print(f'样本匹配：体素归属 + {args.radius}mm 膨胀（{args.atlas.name}）')
        except Exception as exc:                      # noqa: BLE001
            print(f'[warn] 体素匹配不可用（{type(exc).__name__}: {exc}），回退到质心匹配',
                  file=sys.stderr)
            MATCHER = None
    else:
        print('样本匹配：质心 + 半径（旧行为）')

    def run(tag: str, coords_arg, sample_norm, gene_norm, radius, lodo=False):
        t = time.time()
        expr, genes, report, extra = ahba.run_pipeline(
            donors, coords_arg, radius=radius, missing_threshold=0.5,
            sample_norm=sample_norm, gene_norm=gene_norm,
            donor_coverage_min=args.donor_coverage_min, lodo=lodo,
            matcher=MATCHER)
        info = summarize(expr, genes, report, tag)
        info['seconds'] = round(time.time() - t, 1)
        info['sample_norm'] = sample_norm
        info['gene_norm'] = gene_norm
        info['radius_mm'] = radius
        info['match_method'] = args.match_method
        results.append(info)
        # 持久化
        np.save(args.output_dir / f'expression_{tag}.npy', expr)
        (args.output_dir / f'genes_{tag}.json').write_text(
            json.dumps(genes, ensure_ascii=False), encoding='utf-8')
        if 'lodo' in extra:
            np.save(args.output_dir / f'lodo_{tag}.npy',
                    np.stack([extra['lodo'][k] for k in sorted(extra['lodo'])]))
        print(f'  [{tag}] 分区×基因={expr.shape} 可用分区={report.n_parcels_with_data} '
              f'覆盖中位={info["coverage_median"]} 有限值比例={info["frac_finite"]:.3f} '
              f'耗时={info["seconds"]}s')
        return info

    # 主分析（CIC 坐标 + 方案主归一化）
    if args.cic_coords_dir:
        print('应用 CIC 坐标集…')
        donors = list(dd for dd, _ in apply_cic_coords(donors, args.cic_coords_dir))
        if not donors:
            print('所有供体均未能应用 CIC 坐标，回退到原始坐标', file=sys.stderr)
            donors = sorted(p for p in Path(args.ahba_dir).glob('normalized_microarray_donor*')
                            if p.is_dir())

    print(f'\n主分析：归一化={primary_norm}, 半径={args.radius}mm')
    run(f'primary_{primary_norm}', coords, primary_norm, primary_norm, args.radius, lodo=True)

    if args.sensitivity:
        print('\n敏感性分析：')
        for norm in variants:
            run(f'norm_{norm}', coords, norm, norm, args.radius)
        for radius in (1.0, 3.0):
            if abs(radius - args.radius) > 1e-9:
                run(f'radius_{radius:g}mm', coords, primary_norm, primary_norm, radius)

    (args.output_dir / 'ahba_run_summary.json').write_text(
        json.dumps({'results': results, 'total_seconds': round(time.time() - t0, 1)},
                   ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# AHBA 处理报告', '',
             f'- 供体数：{len(donors)}', f'- 分区数：{len(coords)}',
             f'- 主归一化：{primary_norm}', f'- 匹配半径：{args.radius} mm',
             f'- 供体覆盖阈值：{args.donor_coverage_min}', '',
             '| 运行 | 分区×基因 | 可用分区 | 覆盖中位 | 有限值比例 | 值中位 | 耗时(s) |',
             '|---|---|---|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['tag']} | {r['n_parcels']}×{r['n_genes']} | {r['n_parcels_with_data']} | "
                     f"{r['coverage_median']} | {r['frac_finite']:.3f} | {r['value_median']} | {r['seconds']} |")
    low = [r for r in results if r['frac_finite'] < 0.5]
    if low:
        lines += ['', '**警告**：以下运行的有效数据比例低于 50%，需检查坐标匹配与供体覆盖：']
        for r in low:
            lines.append(f"- {r['tag']}：{r['frac_finite']:.3f}")
    (args.output_dir / 'ahba_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(f'\n总耗时 {round(time.time() - t0, 1)}s')
    print(f'报告：{args.output_dir / "ahba_report.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
