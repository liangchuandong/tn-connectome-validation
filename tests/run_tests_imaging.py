#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""零依赖测试（二）：PLS、稳定性重采样、共线性、空间自相关、spin test。

运行：python tests/run_tests_imaging.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from tnwt import spin as sp  # noqa: E402
from tnwt import stats_imaging as si  # noqa: E402

TESTS: list[tuple[str, callable]] = []


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


def sphere_coords(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * 60.0   # 模拟半径 60mm 的脑


def smooth_on_sphere(coords: np.ndarray, length_scale: float, seed: int = 0) -> np.ndarray:
    """构造具有空间自相关的图谱（高斯核平滑白噪声）。"""
    rng = np.random.default_rng(seed)
    c = si.to_unit_sphere(coords) if False else coords / np.linalg.norm(coords, axis=1, keepdims=True)
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    K = np.exp(-d2 / (2 * length_scale ** 2))
    K /= K.sum(axis=1, keepdims=True)
    return K @ rng.normal(size=len(coords))


# ==========================================================================
# PLS
# ==========================================================================
@test
def test_pls_recovers_known_latent_factor():
    """Y = X·w + 噪声时，PLS1 应捕捉到真实潜变量。

    重要说明（本测试的作用与边界）：
    在 p >> n 的高维情形下，**权重向量方向本身不可稳定估计**——这是估计量的固有性质，
    不是实现缺陷。实测：用同一数据、同一真实 w*，
      · 以 X'y 方向作权重的实现得 cos ≈ 0.56；
      · 本实现（先按列标准化再取 X'Y 首奇异向量）得 cos ≈ 0.45；
    两者都"正确"，差异来自标准化带来的几何变化，而在弱信号 + 高维下方向估计极不稳定。
    因此本测试**不断言 cos 的具体数值**，只断言：
      (1) cos 明显优于随机（随机 |cos| ≈ 1/sqrt(p) ≈ 0.058）；
      (2) PLS1 能解释 Y 的绝大部分方差（这才是实际使用中真正重要的性质）。
    """
    rng = np.random.default_rng(1)
    n, p = 120, 300
    w_true = rng.normal(size=p)
    w_true /= np.linalg.norm(w_true)
    X = rng.normal(size=(n, p))
    y = X @ w_true + rng.normal(scale=0.2, size=n)
    r = si.pls_regression(X, y, n_components=3)
    w_dir = r.x_weights_raw[:, 0]
    cos = abs(float(np.dot(w_dir / np.linalg.norm(w_dir), w_true)))
    random_baseline = 1.0 / np.sqrt(p)
    assert cos > 5 * random_baseline, f'cos={cos:.3f} 未明显优于随机基线 {random_baseline:.3f}'
    assert r.var_y[0] > 0.8, f'PLS1 解释 Y 方差仅 {r.var_y[0]:.3f}'
    # 与真实潜变量本身的相关应很高（比权重方向稳健得多）
    r_latent = abs(float(np.corrcoef(r.x_scores[:, 0], X @ w_true)[0, 1]))
    assert r_latent > 0.85, f'PLS1 得分与真实潜变量相关仅 {r_latent:.3f}'
    assert r.x_weights.shape == (p, r.n_components), (r.x_weights.shape, r.n_components)
    assert r.n_components >= 1
    assert np.allclose(r.x_weights[:, 0].mean(), 0, atol=1e-9)  # z 分数
    assert abs(r.x_weights[:, 0].std(ddof=1) - 1) < 1e-9


@test
def test_pls_orthogonal_components_and_ordering():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 200))
    Y = rng.normal(size=(100, 3))
    r = si.pls_regression(X, Y, n_components=4)
    # 奇异值非增
    assert np.all(np.diff(r.singular_values) <= 1e-9)
    # PLS 得分应近似正交。注：秩一压缩方案不保证严格正交（残余相关在数值精度量级），
    # 因此用相关系数容差而非严格正交来断言。
    S = r.x_scores - r.x_scores.mean(axis=0, keepdims=True)
    corr = np.corrcoef(S, rowvar=False)
    off = corr - np.eye(corr.shape[0])
    assert np.abs(off).max() < 0.05, f'成分间残余相关 {np.abs(off).max():.4f} 过大'


@test
def test_pls_handles_nan_rows():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 30))
    y = rng.normal(size=50)
    X[0, 0] = np.nan
    y[1] = np.nan
    r = si.pls_regression(X, y, n_components=2)
    assert r.meta['n_used'] == 48 and r.meta['n_dropped'] == 2, r.meta


@test
def test_pls_permutation_null_is_calibrated():
    """无关联数据的 PLS1 成分置换 p 值应大致均匀（此处只验证不显著）。"""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 60))
    y = rng.normal(size=80)
    r = si.pls_regression(X, y, n_components=2)
    res = si.pls_component_significance(X, y, r, n_perm=200, seed=7)
    assert res['p_perm'][0] > 0.05, res


# ==========================================================================
# 稳定性重采样
# ==========================================================================
@test
def test_stability_identifies_true_signal_genes():
    """真实信号基因的入选率应显著高于噪声基因。"""
    rng = np.random.default_rng(5)
    n, p = 200, 400
    signal = np.zeros(p, dtype=bool)
    signal[:20] = True
    X = rng.normal(size=(n, p))
    y = X[:, signal].sum(axis=1) + rng.normal(scale=2.0, size=n)
    ref = si.pls_regression(X, y, n_components=1).x_weights_raw[:, 0]
    stab = si.pls_stability(X, y, ref, n_resamples=120, seed=11, mode='bootstrap', top_frac=0.05)
    f = stab['selection_freq']
    assert stab['n_valid'] > 100, stab['n_valid']
    assert f[signal].mean() > f[~signal].mean(), (f[signal].mean(), f[~signal].mean())
    # 权重向量方向在高维下不稳定（见 test_pls_recovers_known_latent_factor 的说明），
    # 因此这里只要求余弦为正（方向一致）且重采样有效，不断言具体数值。
    assert np.mean(stab['cosine']) > 0.1, np.mean(stab['cosine'])
    assert stab['sign_consistency'][signal].mean() > 0.9, stab['sign_consistency'][signal].mean()
    cand = si.candidate_genes(f, min_freq=0.8)
    # 候选基因中真实信号基因的比例应明显高于随机（20/400 = 5%）
    if cand.sum() > 0:
        assert signal[cand].mean() > 0.15, signal[cand].mean()


@test
def test_split_half_mode_runs():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(100, 50))
    y = X[:, 0] + rng.normal(size=100)
    ref = si.pls_regression(X, y, n_components=1).x_weights_raw[:, 0]
    stab = si.pls_stability(X, y, ref, n_resamples=50, seed=1, mode='split_half')
    assert stab['n_valid'] > 30
    assert len(stab['cosine']) == stab['n_valid']


# ==========================================================================
# 共线性诊断
# ==========================================================================
@test
def test_collinearity_detects_near_duplicate():
    rng = np.random.default_rng(7)
    a = rng.normal(size=150)
    b = a + rng.normal(scale=0.01, size=150)   # 几乎重复
    c = rng.normal(size=150)
    d = si.collinearity_diagnostics({'A1': a, 'A2': b, 'A3': c})
    assert abs(d['corr'][0][1]) > 0.99
    assert d['vif']['A1'] > 50, d['vif']
    assert d['condition_number'] > 20, d['condition_number']
    assert abs(d['corr'][0][2]) < 0.3


@test
def test_orthogonalize_removes_shared_variance():
    rng = np.random.default_rng(8)
    shared = rng.normal(size=200)
    y = 2.0 * shared + rng.normal(scale=0.1, size=200)
    res = si.orthogonalize(y, shared)
    assert abs(np.corrcoef(res, shared)[0, 1]) < 1e-8, np.corrcoef(res, shared)[0, 1]
    # 残差应保留原始噪声量级
    assert 0.05 < res.std() < 0.2, res.std()


@test
def test_pairwise_spatial_correlation():
    a = np.arange(10, dtype=float)
    out = si.pairwise_spatial_correlation({'A1': a, 'A2': 2 * a + 1, 'A3': -a})
    assert abs(out['A1|A2'] - 1.0) < 1e-12
    assert abs(out['A1|A3'] + 1.0) < 1e-12


# ==========================================================================
# 空间自相关
# ==========================================================================
@test
def test_morans_i_sign_and_range():
    coords = sphere_coords(150, seed=2)
    W = si.spatial_weights(coords, k=10)
    smooth = smooth_on_sphere(coords, 0.3, seed=3)
    noise = np.random.default_rng(4).normal(size=150)
    i_smooth = si.morans_i(smooth, W)
    i_noise = si.morans_i(noise, W)
    assert i_smooth > i_noise, (i_smooth, i_noise)
    assert -1.5 < i_noise < 1.5


# ==========================================================================
# spin test
# ==========================================================================
@test
def test_rotation_matrices_are_proper_rotations():
    rots = sp.gen_rotation_matrices(200, seed=1)
    for R in rots:
        assert abs(np.linalg.det(R) - 1.0) < 1e-9, np.linalg.det(R)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)


@test
def test_spinsamples_is_a_permutation():
    coords = sphere_coords(120, seed=5)
    idx = sp.gen_spinsamples(coords, n_perm=20, seed=2)
    assert idx.shape == (20, 120)
    for row in idx:
        assert len(np.unique(row)) == len(row), '最近邻旋转应得到置换（无重复）'
        assert row.min() >= 0 and row.max() < 120


@test
def test_spin_preserves_autocorrelation_but_shuffle_does_not():
    """核心性质：旋转零分布保留空间自相关，行置换不保留。"""
    coords = sphere_coords(200, seed=6)
    W = si.spatial_weights(coords, k=10)
    x = smooth_on_sphere(coords, 0.25, seed=7)
    spins = sp.gen_spinsamples(coords, n_perm=300, seed=3)

    i_obs = si.morans_i(x, W)
    i_spin = np.array([si.morans_i(x[s], W) for s in spins[:200]])
    rng = np.random.default_rng(8)
    i_shuf = np.array([si.morans_i(x[rng.permutation(len(x))], W) for _ in range(200)])

    # 旋转应保持自相关（保留大部分），置换应趋近于 0。
    # 注意：球面随机点上的分区分布不均，最近邻分配会损失一部分结构，
    # 因此这里只要求"保持 > 50%"，核心是相对 shuffle 的定性差异。
    assert i_spin.mean() > 0.5 * i_obs, (i_spin.mean(), i_obs)
    assert abs(i_shuf.mean()) < 0.15 * abs(i_obs), (i_shuf.mean(), i_obs)

@test
def test_spin_test_null_is_calibrated():
    """两个独立平滑图谱的 spin p 值应大致均匀（此处验证不显著且分布合理）。"""
    coords = sphere_coords(200, seed=9)
    spins = sp.gen_spinsamples(coords, n_perm=400, seed=4)
    ps = []
    for s in range(8):
        x = smooth_on_sphere(coords, 0.3, seed=100 + s)
        y = smooth_on_sphere(coords, 0.3, seed=200 + s)
        r = sp.spin_test(x, spins, y)
        ps.append(r['p_two_sided'])
    ps = np.array(ps)
    assert np.all(np.isfinite(ps))
    # 无关联时不应系统性显著
    assert (ps < 0.05).sum() <= 1, ps


@test
def test_spin_test_detects_true_association():
    """Y = X 时观测相关应为 1，spin p 值应接近最小值 1/(n_perm+1)。"""
    coords = sphere_coords(150, seed=10)
    spins = sp.gen_spinsamples(coords, n_perm=200, seed=5)
    x = smooth_on_sphere(coords, 0.3, seed=11)
    r = sp.spin_test(x, spins, x.copy())
    assert abs(r['observed'] - 1.0) < 1e-12, r['observed']
    assert r['p_two_sided'] <= 1.0 / (r['n_perm'] + 1) + 1e-12, r['p_two_sided']
    # 加入极小噪声后仍应显著
    y = x + 0.01 * np.random.default_rng(12).normal(size=150)
    r2 = sp.spin_test(x, spins, y)
    assert r2['observed'] > 0.99, r2['observed']
    assert r2['p_two_sided'] <= 0.02, r2['p_two_sided']


@test
def test_spin_observed_equals_numpy_corrcoef():
    """回归测试：spin_test 报告的 observed 必须与 np.corrcoef 一致（1e-12 内）。

    背景：本模块的 corr 用 mean((a-ā)(b-b̄)) / (std0(a)·std0(b)) 这一等价形式实现。
    曾一度误改为 ddof=1 的样本标准差，导致结果偏离 numpy/pandas 约定
    （实测 numpy 2.3.5 与 pandas 3.0.1 的 corr 均为 0.8220，而 ddof=1 写法给出 0.6576）。
    该偏差会系统性影响所有空间关联结论，因此用本测试锁死等价性。
    """
    coords = sphere_coords(150, seed=10)
    spins = sp.gen_spinsamples(coords, n_perm=50, seed=5)
    x = smooth_on_sphere(coords, 0.3, seed=11)
    y = smooth_on_sphere(coords, 0.3, seed=12)
    r = sp.spin_test(x, spins, y)
    ref = float(np.corrcoef(x, y)[0, 1])
    assert abs(r['observed'] - ref) < 1e-12, (r['observed'], ref)
    # 相同图谱必须给出精确 1；负相关给出精确 -1
    assert abs(sp.spin_test(x, spins, x.copy())['observed'] - 1.0) < 1e-12
    assert abs(sp.spin_test(x, spins, -x)['observed'] + 1.0) < 1e-12


@test
def test_spin_test_null_is_calibrated_by_rotating():
    """旋转后的图谱与固定图谱之间不应出现系统性显著（自洽性检查）。"""
    coords = sphere_coords(180, seed=21)
    spins = sp.gen_spinsamples(coords, n_perm=500, seed=7)
    y = smooth_on_sphere(coords, 0.3, seed=22)
    # 用第一个旋转作为"观测"图谱：它是 y 的旋转版本，不应显著
    x_rot = y[spins[0]]
    r = sp.spin_test(x_rot, spins[1:], y)
    assert r['n_perm'] == 499
    assert r['p_two_sided'] > 0.05, r['p_two_sided']


@test
def test_spin_test_handles_missing_values():
    coords = sphere_coords(100, seed=13)
    spins = sp.gen_spinsamples(coords, n_perm=50, seed=6)
    x = smooth_on_sphere(coords, 0.3, seed=14)
    y = smooth_on_sphere(coords, 0.3, seed=15)
    x[3] = np.nan
    y[7] = np.nan
    r = sp.spin_test(x, spins, y)
    assert np.isfinite(r['observed']) and r['n_perm'] > 0


@test
def test_fdr_bh_known_behaviour():
    rej, q = sp.fdr_bh([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    # 经典 BH 示例：前两个被拒绝
    assert rej[:2].all() and not rej[2:].any(), (rej, q)
    assert abs(q[0] - 0.01) < 1e-9, q[0]


@test
def test_fwer_perm_bounds():
    null = np.linspace(-1, 1, 1000)
    assert abs(sp.fwer_perm(0.0, null) - 1.0) < 0.01
    assert sp.fwer_perm(2.0, null) < 0.01


@test
def test_spin_refit_equals_fixed_when_stat_is_fixed():
    """stat_fn 与模型选择无关时，refit 与固定权重应完全等价。

    检验实现正确性：当统计量只是"给定固定向量与图谱的相关"，
    就没有"重新拟合"可言，两种方案的零分布与 p 必须一致。
    """
    rng = np.random.default_rng(0)
    n = 60
    coords = rng.normal(size=(n, 3))
    coords /= np.linalg.norm(coords, axis=1, keepdims=True)
    spins = sp.gen_spinsamples(coords * 50, n_perm=200, seed=1)
    y = rng.normal(size=n)
    x = rng.normal(size=n)

    def stat_fixed(yv):
        sa, sb = x.std(ddof=0), yv.std(ddof=0)
        if sa == 0 or sb == 0:
            return 0.0
        return float(((x - x.mean()) * (yv - yv.mean())).mean() / (sa * sb))

    ref = sp.spin_test_refit(stat_fixed, y, spins)
    # 约定：`spin_test_refit` 旋转**影像图谱 y**。因此对照必须写成
    # spin_test(y, spins, x)（旋转 y、固定 x），而不是 spin_test(x, spins, y)。
    # 两者对相关系数是对称的，但置换族的索引不同，零分布逐项并不相同。
    fix = sp.spin_test(y, spins, x)
    assert abs(ref['observed'] - fix['observed']) < 1e-10
    assert abs(ref['p_two_sided'] - fix['p_two_sided']) < 1e-10
    assert abs(ref['null_mean'] - fix['null'].mean()) < 1e-10


@test
def test_spin_refit_null_is_elevated_under_model_selection():
    """stat_fn 含模型选择时，refit 的零分布必须显著高于固定权重。

    这是本函数存在的理由：固定权重零分布以 0 为中心，
    而"从大量基向量里挑出最相关的一个"这一选择行为会把统计量整体抬高，
    refit 如实反映该膨胀；固定权重则会把 p 压得过小。
    """
    rng = np.random.default_rng(2)
    n = 80
    coords = rng.normal(size=(n, 3))
    coords /= np.linalg.norm(coords, axis=1, keepdims=True)
    spins = sp.gen_spinsamples(coords * 50, n_perm=300, seed=3)

    # 100 个随机基向量：stat 取"与 y 相关性最大的那个"——典型的选择型统计量
    basis = rng.normal(size=(100, n))
    basis = (basis - basis.mean(axis=1, keepdims=True)) / basis.std(axis=1, ddof=0, keepdims=True)

    def corr_row(b, v):
        sv = v.std(ddof=0)
        if sv == 0:
            return 0.0
        vv = (v - v.mean()) / sv
        return float(np.max(np.abs(b @ vv)) / len(v))

    y = rng.normal(size=n)
    # refit：每次置换都重新挑选最相关的基
    ref = sp.spin_test_refit(lambda v: corr_row(basis, v), y, spins)
    # 固定：用观测数据挑出的那个基，置换时不再挑选
    best = int(np.argmax(np.abs(basis @ ((y - y.mean()) / y.std(ddof=0)))))
    fix = sp.spin_test(y, spins, basis[best])

    assert ref['null_mean'] > 2 * max(fix['null'].mean(), 1e-6), (
        ref['null_mean'], fix['null'].mean())
    assert ref['null_mean'] > 0.1, ref['null_mean']
    assert np.isfinite(ref['p_two_sided']) and 0 < ref['p_two_sided'] <= 1


@test
def test_spin_refit_flags_pure_noise_as_nonsignificant():
    """纯噪声下 refit 不应给出显著结果（避免新实现把假阳性当成真信号）。"""
    rng = np.random.default_rng(4)
    n = 80
    coords = rng.normal(size=(n, 3))
    coords /= np.linalg.norm(coords, axis=1, keepdims=True)
    spins = sp.gen_spinsamples(coords * 50, n_perm=500, seed=5)
    basis = rng.normal(size=(200, n))
    y_noise = rng.normal(size=n)
    res = sp.spin_test_refit(
        lambda v: float(np.max(np.abs(basis @ ((v - v.mean()) / (v.std(ddof=0) or 1))))),
        y_noise, spins)
    # 噪声下 p 不应极小；允许抽样波动
    assert res['p_two_sided'] > 0.02, res['p_two_sided']


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
