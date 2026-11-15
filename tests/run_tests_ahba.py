#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""零依赖测试（三）：AHBA 管线（读取/过滤/选探针/匹配/归一化/聚合/LODO）。

运行：python tests/run_tests_ahba.py
"""
from __future__ import annotations

import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tnwt import ahba  # noqa: E402

TESTS: list[tuple[str, callable]] = []
TMP = ROOT / 'build' / 'ahba_selftest'


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


def _fresh_tmp():
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


# ==========================================================================
# 归一化
# ==========================================================================
@test
def test_normalization_formulas():
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    # center: 均值为 0
    c = ahba.normalize(x, 'center')
    assert abs(c.mean()) < 1e-12
    # zscore: 均值 0、样本标准差 1
    z = ahba.normalize(x, 'zscore')
    assert abs(z.mean()) < 1e-12 and abs(z.std(ddof=1) - 1) < 1e-12
    # minmax: 落在 [0,1] 且端点为 0/1
    m = ahba.normalize(x, 'minmax')
    assert abs(m.min()) < 1e-12 and abs(m.max() - 1) < 1e-12
    # sigmoid: 落在 (0,1)
    s = ahba.normalize(x, 'sigmoid')
    assert s.min() > 0 and s.max() < 1 and s.max() < 1
    assert abs(s[0, 2] - 0.5) < 1e-12, '中位数应映射到 0.5'
    # scaled_sigmoid 端点 0/1
    ss = ahba.normalize(x, 'scaled_sigmoid')
    assert abs(ss.min()) < 1e-12 and abs(ss.max() - 1) < 1e-12
    # robust_sigmoid：中位数映射到 0.5
    rs = ahba.normalize(x, 'robust_sigmoid')
    assert abs(rs[0, 2] - 0.5) < 1e-9, rs
    # scaled_robust_sigmoid 端点 0/1
    srs = ahba.normalize(x, 'scaled_robust_sigmoid')
    assert abs(srs.min()) < 1e-12 and abs(srs.max() - 1) < 1e-12
    # None 原样返回
    assert np.allclose(ahba.normalize(x, None), x)
    # 未知方法应报错
    try:
        ahba.normalize(x, 'nope')
    except ValueError:
        pass
    else:
        raise AssertionError('未知方法应抛错')


@test
def test_normalization_handles_zero_variance_row():
    x = np.array([[3.0, 3.0, 3.0], [1.0, 2.0, 3.0]])
    for method in ('zscore', 'minmax', 'sigmoid', 'robust_sigmoid',
                   'scaled_sigmoid', 'scaled_robust_sigmoid', 'mixed_sigmoid'):
        out = ahba.normalize(x, method)
        assert out.shape == x.shape, (method, out.shape)
        assert np.all(np.isnan(out[0])), f'{method}: 零方差行应为 NaN'
        assert np.all(np.isfinite(out[1])), method


# ==========================================================================
# 读取与过滤
# ==========================================================================
@test
def test_read_donor_roundtrip():
    tmp = _fresh_tmp()
    d = ahba.make_synthetic_donor(tmp, '9999', n_samples=30, n_genes=10, probes_per_gene=2)
    donor = ahba.read_donor(d)
    assert donor.donor_id == '9999', donor.donor_id
    assert donor.n_probes == 20 and donor.n_samples == 30
    assert donor.expression.shape == (20, 30)
    assert donor.mni.shape == (30, 3)
    assert donor.detected is not None and donor.detected.shape == (20, 30)


@test
def test_read_donor_missing_file_raises():
    tmp = _fresh_tmp()
    d = ahba.make_synthetic_donor(tmp, '8888')
    (d / 'Probes.csv').unlink()
    try:
        ahba.read_donor(d)
    except FileNotFoundError:
        return
    raise AssertionError('缺文件应报 FileNotFoundError')


@test
def test_filter_drops_unannotated_and_low_detection():
    tmp = _fresh_tmp()
    d = ahba.make_synthetic_donor(tmp, '7777', n_samples=40, n_genes=10, probes_per_gene=2)
    donor = ahba.read_donor(d)
    n0 = donor.n_probes
    donor = ahba.reannotate_probes(donor)
    donor = ahba.filter_probes(donor, missing_threshold=0.5)
    assert donor.n_probes <= n0
    # 人为插入一个未注释探针，应被剔除
    donor.gene_symbols[0] = 'A_32_P00000'
    donor.entrez_ids[0] = ''
    n1 = donor.n_probes
    donor = ahba.filter_probes(donor, missing_threshold=0.5, drop_unannotated=True)
    assert donor.n_probes == n1 - 1, (n1, donor.n_probes)


@test
def test_select_probes_picks_one_per_gene():
    tmp = _fresh_tmp()
    d = ahba.make_synthetic_donor(tmp, '6666', n_samples=50, n_genes=12, probes_per_gene=3)
    donor = ahba.filter_probes(ahba.reannotate_probes(ahba.read_donor(d)))
    sel = ahba.select_probes(donor)
    assert len(sel) == 12, len(sel)
    assert len(set(sel.values())) == 12, '每个基因应选中不同探针'
    genes, mat = ahba.gene_by_sample(donor, sel)
    assert mat.shape == (12, 50), mat.shape
    assert genes == sorted(sel)
    # 被选探针应与其同基因其它探针平均相关最高（抽样验证一个基因）
    from collections import defaultdict
    groups = defaultdict(list)
    for i, g in enumerate(donor.gene_symbols):
        groups[g].append(i)
    for g, rows in list(groups.items())[:3]:
        if len(rows) < 2:
            continue
        sub = donor.expression[rows]
        corr = np.nan_to_num(np.corrcoef(sub))
        np.fill_diagonal(corr, np.nan)
        scores = np.nanmean(corr, axis=1)
        assert sel[g] == rows[int(np.nanargmax(scores))], g


# ==========================================================================
# 匹配与聚合
# ==========================================================================
@test
def test_match_samples_radius_and_k():
    coords = np.array([[0.0, 0, 0], [50.0, 0, 0]])
    samples = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.5, 0, 0], [50.0, 0, 0]])
    a = ahba.match_samples(coords, samples, radius=2.0)
    assert a[0] == [0, 1] and a[1] == [3], a
    b = ahba.match_samples(coords, samples, k=1)
    assert b[0] == [0] and b[1] == [3], b
    # 无样本落入半径应返回空列表
    c = ahba.match_samples(np.array([[999.0, 0, 0]]), samples, radius=1.0)
    assert c == [[]], c


@test
def test_parcel_expression_means_and_nan():
    genes = ['A', 'B']
    mat = np.array([[1.0, 3.0, 10.0], [2.0, 4.0, 20.0]])   # 基因 x 样本
    assign = [[0, 1], [], [2]]
    out = ahba.parcel_expression(genes, mat, assign)
    assert out.shape == (3, 2)
    assert np.allclose(out[0], [2.0, 3.0])
    assert np.all(np.isnan(out[1]))
    assert np.allclose(out[2], [10.0, 20.0])


# ==========================================================================
# 端到端
# ==========================================================================
@test
def test_pipeline_end_to_end_and_lodo():
    tmp = _fresh_tmp()
    n_parcels = 20
    rng = np.random.default_rng(5)
    coords = rng.normal(0, 30, size=(n_parcels, 3))
    # 关键：所有供体共享同一组分区质心（真实 AHBA 中所有供体在同一 MNI 空间）
    donors = [ahba.make_synthetic_donor(tmp, str(1000 + i), n_samples=60, n_genes=30,
                                        probes_per_gene=2, seed=i, coords=coords)
              for i in range(3)]
    expr, genes, report, extra = ahba.run_pipeline(
        donors, coords, radius=2.0, sample_norm='sigmoid', gene_norm='sigmoid',
        donor_coverage_min=2, lodo=True)
    assert expr.shape == (n_parcels, len(genes)), expr.shape
    assert len(genes) > 0 and report.n_donors == 3
    assert report.n_probes_raw == 3 * 60
    assert report.parcel_donor_coverage.shape == (n_parcels,)
    assert report.n_parcels_with_data > 0, report.parcel_donor_coverage
    assert 'lodo' in extra and len(extra['lodo']) == 3
    low = report.parcel_donor_coverage < 2
    if low.any():
        assert np.all(np.isnan(expr[low])), '覆盖不足的分区应为 NaN'
    assert np.isfinite(expr[~low]).any()


@test
def test_pipeline_normalization_variants_differ():
    tmp = _fresh_tmp()
    coords = np.random.default_rng(7).normal(0, 30, size=(15, 3))
    donors = [ahba.make_synthetic_donor(tmp, str(2000 + i), n_samples=40, n_genes=15,
                                        probes_per_gene=2, seed=10 + i, coords=coords)
              for i in range(2)]
    outs = {}
    for norm in ('sigmoid', 'zscore', 'scaled_robust_sigmoid'):
        expr, genes, _, _ = ahba.run_pipeline(donors, coords, sample_norm=norm,
                                             gene_norm=norm, donor_coverage_min=1)
        assert np.isfinite(expr).any(), f'{norm} 未产出任何有限值'
        outs[norm] = expr[np.isfinite(expr)]
    assert not np.allclose(np.median(outs['sigmoid']), np.median(outs['zscore']))
    assert abs(np.nanmean(outs['zscore'])) < 0.5, np.nanmean(outs['zscore'])


@test
def test_pipeline_is_deterministic():
    tmp = _fresh_tmp()
    coords = np.random.default_rng(9).normal(0, 30, size=(15, 3))
    donors = [ahba.make_synthetic_donor(tmp, str(3000 + i), n_samples=40, n_genes=15,
                                        probes_per_gene=2, seed=20 + i, coords=coords)
              for i in range(2)]
    a, ga, _, _ = ahba.run_pipeline(donors, coords, donor_coverage_min=1)
    b, gb, _, _ = ahba.run_pipeline(donors, coords, donor_coverage_min=1)
    assert ga == gb
    assert np.allclose(a, b, equal_nan=True), '相同输入应完全可复现'


def main() -> int:
    failures = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f'PASS  {name}')
        except Exception:  # noqa: BLE001
            failures += 1
            print(f'FAIL  {name}')
            traceback.print_exc()
    print(f'\n{len(TESTS) - failures}/{len(TESTS)} 通过')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
