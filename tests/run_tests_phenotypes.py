#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""零依赖测试（五）：影像表型提取与混淆控制（阶段四–五）。

覆盖：派生数据读取、A2 图谐波耦合、侧别处理、混淆残差化、
Welch/Hedges' g 组间比较、样本量匹配重采样、年龄/FD 匹配。

运行：python tests/run_tests_phenotypes.py
"""
from __future__ import annotations

import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tnwt import phenotypes as ph  # noqa: E402

TESTS: list[tuple[str, callable]] = []
TMP = ROOT / 'build' / 'pheno_selftest'
N_PARCELS = 20           # 小规模图集，左右各 10


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


def _fresh_tmp():
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


def _write_vectors(root: Path, kind: str, data: dict[str, np.ndarray]):
    d = root / kind
    d.mkdir(parents=True, exist_ok=True)
    for sid, vals in data.items():
        pd.DataFrame({'parcel_index': np.arange(1, len(vals) + 1), 'value': vals}) \
            .to_csv(d / f'{sid}.csv', index=False)


def _write_matrices(root: Path, kind: str, data: dict[str, np.ndarray]):
    d = root / kind
    d.mkdir(parents=True, exist_ok=True)
    for sid, m in data.items():
        np.save(d / f'{sid}.npy', m)


def _make_derivatives(root: Path, n_subjects: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    subs = [f'sub-{i+1:03d}' for i in range(n_subjects)]
    for kind in ('a1_streamlines', 'a1_fa', 'a3_gwc', 'a3_t1t2'):
        _write_vectors(root, kind, {s: rng.normal(1, 0.2, N_PARCELS) for s in subs})
    # 结构连接：稀疏对称非负
    sc = {}
    for s in subs:
        m = rng.random((N_PARCELS, N_PARCELS))
        m = np.triu(m, 1)
        m = m + m.T
        sc[s] = m
    fc = {}
    for s in subs:
        base = rng.normal(size=(N_PARCELS, 30))
        c = np.corrcoef(base)
        np.fill_diagonal(c, 1.0)
        fc[s] = c
    _write_matrices(root, 'a2/sc', sc)
    _write_matrices(root, 'a2/fc', fc)
    qc = pd.DataFrame({
        'participant_id': subs, 'session': 'baseline',
        'has_t1w': 1, 'has_dwi': 1, 'has_bold': 1,
        'bold_mean_fd': rng.uniform(0.05, 0.4, n_subjects),
        'exclusion_flag': 0,
    })
    (root / 'qc').mkdir(parents=True, exist_ok=True)
    qc.to_csv(root / 'qc' / 'subjects_qc.csv', index=False)
    return subs


def _clinical(subs: list[str], seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(subs)
    return pd.DataFrame({
        'participant_id': subs,
        'group_norm': ['patient'] * (n // 2) + ['control'] * (n - n // 2),
        'age': rng.normal(58, 10, n),
        'sex_label': rng.choice(['F', 'M'], n),
        'bold_mean_fd': rng.uniform(0.05, 0.4, n),
        'fazekas_grade': rng.integers(0, 2, n).astype(float),
    })


# ==========================================================================
# 读取
# ==========================================================================
@test
def test_load_derivatives_shape_and_index():
    tmp = _fresh_tmp()
    subs = _make_derivatives(tmp)
    d = ph.load_derivatives(tmp, n_parcels=N_PARCELS)
    assert set(d.vectors) == {'a1_streamlines', 'a1_fa', 'a3_gwc', 'a3_t1t2'}
    assert d.vectors['a1_fa'].shape == (len(subs), N_PARCELS)
    assert list(d.vectors['a1_fa'].index) == subs
    assert len(d.matrices['a2/sc']) == len(subs)
    assert d.matrices['a2/sc'][subs[0]].shape == (N_PARCELS, N_PARCELS)
    assert not d.qc.empty


@test
def test_read_vector_rejects_wrong_length():
    tmp = _fresh_tmp()
    d = tmp / 'a1_fa'
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'parcel_index': [1, 2], 'value': [1.0, 2.0]}).to_csv(d / 'sub-001.csv', index=False)
    try:
        ph.load_derivatives(tmp, n_parcels=N_PARCELS)
    except ValueError as exc:
        assert '分区数' in str(exc)
        return
    raise AssertionError('分区数不符应抛 ValueError')


# ==========================================================================
# A2 图谐波耦合
# ==========================================================================
@test
def test_laplacian_properties():
    rng = np.random.default_rng(2)
    m = rng.random((10, 10))
    sc = np.triu(m, 1)
    sc = sc + sc.T
    L = ph.structure_laplacian(sc, normalize=False)
    assert np.allclose(L, L.T), '拉普拉斯应对称'
    assert np.allclose(L.sum(axis=1), 0, atol=1e-9), '行和应为 0'
    Ln = ph.structure_laplacian(sc, normalize=True)
    ev = np.linalg.eigvalsh(Ln)
    assert ev.min() > -1e-9 and ev.max() <= 2 + 1e-9, (ev.min(), ev.max())
    assert abs(ev.min()) < 1e-9, '归一化拉普拉斯应有零特征值'
    # 非对称输入应被强制对称化
    asym = sc.copy()
    asym[0, 1] = 0.0
    L2 = ph.structure_laplacian(asym, normalize=False)
    assert np.allclose(L2, L2.T)


@test
def test_a2_high_when_function_follows_structure():
    """当功能连接矩阵由结构谐波基生成时，R² 应显著高于随机功能连接。

    注意构造方式：图谐波耦合对**功能连接矩阵的每一行**（即每个分区与其他分区的
    功能连接剖面）用结构谐波基做最小二乘重建。因此要让 R² 高，必须让 FC 的**行空间**
    落在前若干低频谐波基张成的子空间内，即

        FC_row[i] = Σ_k c_{ik} B[k]

    其中 B 为低频谐波基。等价地 ``FC = C @ B.T``（C 为 分区×模态 系数）。
    """
    rng = np.random.default_rng(3)
    n = 60
    m = rng.random((n, n))
    sc = np.triu(m, 1)
    sc = sc + sc.T
    L = ph.structure_laplacian(sc)
    ev, U = np.linalg.eigh(L)
    k = 6
    B = U[:, np.argsort(ev)[:k]]              # (n, k) 低频谐波基
    C = rng.normal(size=(n, k))
    fc_struct = C @ B.T                       # 行空间落在谐波基张成的子空间
    r2_hi = ph.harmonic_coupling(fc_struct, sc, n_modes=k)
    fc_rand = rng.normal(size=(n, n))
    r2_lo = ph.harmonic_coupling(fc_rand, sc, n_modes=k)
    assert np.nanmean(r2_hi) > 0.9, np.nanmean(r2_hi)
    assert np.nanmean(r2_hi) > np.nanmean(r2_lo) + 0.5, (np.nanmean(r2_hi), np.nanmean(r2_lo))
    assert np.isfinite(r2_hi).all()


@test
def test_a2_excludes_self_and_handles_nan():
    rng = np.random.default_rng(4)
    n = 30
    sc = rng.random((n, n))
    sc = np.triu(sc, 1)
    sc = sc + sc.T
    fc = np.corrcoef(rng.normal(size=(n, 20)))
    fc[0, :] = np.nan
    fc[:, 0] = np.nan
    fc[0, 0] = 1.0
    out = ph.harmonic_coupling(fc, sc, n_modes=3)
    assert out.shape == (n,)
    assert np.isnan(out[0])
    assert np.isfinite(out[1:]).any()


@test
def test_a2_equals_timeseries_path():
    """由时间序列算 FC 再算耦合，应与直接传入 FC 等价。"""
    rng = np.random.default_rng(5)
    n, t = 25, 200
    ts = rng.normal(size=(n, t))
    fc = np.corrcoef(ts)
    sc = rng.random((n, n))
    sc = np.triu(sc, 1)
    sc = sc + sc.T
    a = ph.harmonic_coupling(fc, sc, n_modes=4)
    b = ph.coupling_from_timeseries(ts, sc, n_modes=4)
    assert np.allclose(a, b, equal_nan=True)


@test
def test_compute_a2_uses_intersection_of_subjects():
    tmp = _fresh_tmp()
    subs = _make_derivatives(tmp, n_subjects=6)
    d = ph.load_derivatives(tmp, n_parcels=N_PARCELS)
    # 人为删掉一个被试的 FC
    d.matrices['a2/fc'].pop(subs[0])
    a2 = ph.compute_a2(d, n_modes=3)
    assert a2.shape == (len(subs) - 1, N_PARCELS)
    assert subs[0] not in a2.index


# ==========================================================================
# 侧别处理
# ==========================================================================
@test
def test_hemisphere_split_and_contralateral():
    v = np.arange(20, dtype=float)          # 0..9 左，10..19 右
    left, right = ph.split_hemispheres(20)
    assert list(left) == list(range(10)) and list(right) == list(range(10, 20))
    # 右侧疼痛 → 取左半球
    assert np.allclose(ph.contralateral_map(v, 'R'), v[:10])
    # 左侧疼痛 → 取右半球
    assert np.allclose(ph.contralateral_map(v, 'L'), v[10:])
    # 双侧 → 平均
    assert np.allclose(ph.contralateral_map(v, 'Bilateral'), (v[:10] + v[10:]) / 2)
    # 大小写与空白容错
    assert np.allclose(ph.contralateral_map(v, ' r '), v[:10])


@test
def test_mirror_symmetric_is_symmetric():
    rng = np.random.default_rng(6)
    v = rng.normal(size=20)
    m = ph.mirror_symmetric(v)
    assert m.shape == v.shape
    assert np.allclose(m[:10], m[10:]), '镜像后左右半球应相同'


# ==========================================================================
# 混淆控制
# ==========================================================================
@test
def test_residualize_removes_age_effect():
    """若信号完全由年龄驱动，残差化后与年龄的相关应接近 0。"""
    rng = np.random.default_rng(7)
    n = 60
    subs = [f'sub-{i+1:03d}' for i in range(n)]
    clinical = _clinical(subs, seed=8)
    age = clinical['age'].to_numpy()
    # 每个分区都是年龄的线性函数 + 常数
    mat = pd.DataFrame(np.outer(age, rng.normal(1, 0.2, 5)) + rng.normal(0, 0.01, (n, 5)),
                       index=subs)
    resid = ph.residualize(mat, clinical)
    for c in mat.columns:
        r = np.corrcoef(resid[c].to_numpy(), age)[0, 1]
        assert abs(r) < 1e-6, (c, r)
    # 残差量级应远小于原始量级
    assert np.nanstd(resid.to_numpy()) < 0.05 * np.nanstd(mat.to_numpy())


@test
def test_design_matrix_includes_intercept_and_terms():
    subs = [f'sub-{i+1:03d}' for i in range(10)]
    clinical = _clinical(subs, seed=9)
    X, names = ph.build_design(clinical, subs, with_age_squared=True)
    assert X.shape[0] == 10
    assert names[0] == 'intercept'
    assert np.allclose(X[:, 0], 1.0)
    assert 'age' in names and 'age_sq' in names and 'sex_male' in names
    assert 'mean_fd' in names


@test
def test_residualize_preserves_subject_order():
    subs = [f'sub-{i+1:03d}' for i in range(12)]
    clinical = _clinical(subs, seed=10)
    mat = pd.DataFrame(np.random.default_rng(1).normal(size=(12, 4)), index=subs)
    resid = ph.residualize(mat, clinical)
    assert list(resid.index) == subs
    assert list(resid.columns) == list(mat.columns)


# ==========================================================================
# 组间比较
# ==========================================================================
@test
def test_group_contrast_recovers_planted_effect():
    """人为给患者组加固定偏移，Hedges' g 应接近解析值且方向正确。"""
    rng = np.random.default_rng(11)
    n_pat, n_ctl, n_par = 40, 40, 6
    subs = [f'P{i+1:03d}' for i in range(n_pat)] + [f'C{i+1:03d}' for i in range(n_ctl)]
    group = pd.Series(['patient'] * n_pat + ['control'] * n_ctl, index=subs)
    mat = pd.DataFrame(rng.normal(0, 1, (len(subs), n_par)), index=subs)
    mat.iloc[:n_pat, 0] += 1.0              # 仅第 0 分区有效应
    res = ph.group_contrast(mat, group)
    assert res.n_patient == n_pat and res.n_control == n_ctl
    assert 0.7 < res.g[0] < 1.3, res.g[0]
    assert res.p[0] < 1e-4, res.p[0]
    assert np.nanmax(np.abs(res.g[1:])) < 0.5, res.g[1:]
    # Welch t 与 g 方向一致
    assert np.sign(res.t[0]) == np.sign(res.g[0])


@test
def test_group_contrast_handles_tiny_groups():
    subs = ['a', 'b', 'c', 'd']
    group = pd.Series(['patient', 'patient', 'control', 'control'], index=subs)
    mat = pd.DataFrame(np.arange(8, dtype=float).reshape(4, 2), index=subs)
    res = ph.group_contrast(mat, group)
    assert np.all(np.isnan(res.g)), '每组不足 3 例时应返回 NaN'


@test
def test_n_matched_resampling_stability():
    """样本量匹配重采样：真实效应分区的符号一致性应极高，且重采样均值接近参考值。

    说明：无效应分区的符号一致性**不会接近 0.5**。因为两组样本量不同（患者较多、
    对照较少），参考 g 由全样本估计，而重采样用较小的平衡样本；同向概率由较小样本的
    t 分布决定（本例 n=25 时约 0.97）。因此本测试只对真实效应分区下强断言，
    对无效应分区只要求统计量有限且非退化。
    """
    rng = np.random.default_rng(12)
    n_pat, n_ctl = 60, 25
    subs = [f'P{i+1:03d}' for i in range(n_pat)] + [f'C{i+1:03d}' for i in range(n_ctl)]
    group = pd.Series(['patient'] * n_pat + ['control'] * n_ctl, index=subs)
    mat = pd.DataFrame(rng.normal(0, 1, (len(subs), 8)), index=subs)
    mat.iloc[:n_pat, 0] += 1.2              # 强效应
    out = ph.n_matched_resampling(mat, group, n_rep=200, seed=13)
    assert out['n_rep'] == 200 and out['n_small'] == n_ctl
    assert out['sign_consistency'][0] > 0.95, out['sign_consistency'][0]
    assert np.all(np.isfinite(out['g_mean'])), '重采样均值应全为有限值'
    assert np.all(out['g_sd'] > 0), '重采样标准差应大于 0（非退化）'
    # 重采样均值应与参考效应方向一致
    ref = ph.group_contrast(mat, group).g
    assert np.all(np.sign(out['g_mean']) == np.sign(ref)), (out['g_mean'], ref)
    # 真实效应分区的一致性应高于无效应分区的一致性中位数
    assert out['sign_consistency'][0] > np.median(out['sign_consistency'][1:])


@test
def test_n_matched_resampling_too_small_group():
    subs = ['a', 'b', 'c']
    group = pd.Series(['patient', 'patient', 'control'], index=subs)
    mat = pd.DataFrame(np.ones((3, 2)), index=subs)
    out = ph.n_matched_resampling(mat, group, n_rep=10, seed=1)
    assert out['n_rep'] == 0 and out['n_small'] == 1


@test
def test_matched_subsample_respects_caliper():
    """年龄匹配：匹配后两组年龄差异应明显小于匹配前。"""
    rng = np.random.default_rng(14)
    n_pat, n_ctl = 50, 25
    subs = [f'P{i+1:03d}' for i in range(n_pat)] + [f'C{i+1:03d}' for i in range(n_ctl)]
    clinical = pd.DataFrame({
        'participant_id': subs,
        'group_norm': ['patient'] * n_pat + ['control'] * n_ctl,
        'age': np.concatenate([rng.normal(62, 10, n_pat), rng.normal(52, 8, n_ctl)]),
    })
    group = pd.Series(clinical['group_norm'].to_numpy(), index=subs)
    mat = pd.DataFrame(rng.normal(size=(len(subs), 3)), index=subs)
    before = abs(clinical.loc[clinical.group_norm == 'patient', 'age'].mean()
                 - clinical.loc[clinical.group_norm != 'patient', 'age'].mean())
    matched = ph.matched_subsample(mat, clinical, match_on='age', group=group)
    assert len(matched) >= 10, len(matched)
    cl = clinical.set_index('participant_id').loc[matched]
    after = abs(cl.loc[cl.group_norm == 'patient', 'age'].mean()
                - cl.loc[cl.group_norm != 'patient', 'age'].mean())
    assert after < before, (before, after)
    assert after < 0.5 * before, (before, after)


@test
def test_matched_subsample_missing_variable_raises():
    subs = ['a', 'b']
    clinical = pd.DataFrame({'participant_id': subs, 'group_norm': ['patient', 'control']})
    mat = pd.DataFrame(np.ones((2, 2)), index=subs)
    try:
        ph.matched_subsample(mat, clinical, match_on='not_a_column')
    except KeyError:
        return
    raise AssertionError('缺失匹配变量应抛 KeyError')


# ==========================================================================
# 汇总
# ==========================================================================
@test
def test_phenotype_summary_counts():
    mat = pd.DataFrame([[1.0, np.nan], [2.0, 4.0]], index=['a', 'b'])
    s = ph.phenotype_summary(mat, 'A1')
    assert s['n_subjects'] == 2 and s['n_parcels'] == 2
    assert abs(s['frac_finite'] - 0.75) < 1e-12
    assert s['parcels_all_nan'] == 0
    assert s['subjects_all_nan'] == 0
    mat2 = pd.DataFrame([[np.nan, np.nan]], index=['x'])
    s2 = ph.phenotype_summary(mat2, 'A2')
    assert s2['parcels_all_nan'] == 2 and s2['subjects_all_nan'] == 1


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
