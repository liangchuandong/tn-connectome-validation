#!/usr/bin/env python -W ignore
# -*- coding: utf-8 -*-
"""零依赖测试（七）：收敛判据与稳健性指标（阶段七）。

覆盖：空间可分离性、权重余弦、事前判据判定、正交化残差回退、
集合重叠率、top 基因集合提取、外部一致性、报告渲染。

运行：python tests/run_tests_convergence.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tnwt import convergence as cv  # noqa: E402

TESTS: list[tuple[str, callable]] = []


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


def _fake_data(n_par=200, n_gene=300, seed=0, shared=True):
    """构造三个模态：共享一个潜因子（用于收敛）或各自独立（用于不收敛）。"""
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(n_par, n_gene))
    w = rng.normal(size=n_gene)
    latent = E @ w
    if shared:
        maps = {f'M{i}': latent + rng.normal(0, 0.3, n_par) for i in range(3)}
    else:
        maps = {f'M{i}': E @ rng.normal(size=n_gene) + rng.normal(0, 0.3, n_par)
                for i in range(3)}
    return E, maps


# ==========================================================================
# 可分离性
# ==========================================================================
@test
def test_spatial_independence_reports_pairs():
    rng = np.random.default_rng(1)
    maps = {'A': rng.normal(size=100)}
    maps['B'] = maps['A'] + rng.normal(0, 0.01, 100)   # 高度共线
    maps['C'] = rng.normal(size=100)
    out = cv.spatial_independence(maps)
    assert out['names'] == ['A', 'B', 'C']
    assert out['pairwise_abs_r']['A|B'] > 0.99
    assert out['max_abs_r'] > 0.99
    assert abs(out['pairwise_abs_r']['A|C']) < 0.3
    assert out['vif']['A'] > 10


@test
def test_spatial_independence_single_map():
    out = cv.spatial_independence({'A': np.arange(20, dtype=float)})
    assert out['pairwise_abs_r'] == {}
    assert not np.isfinite(out['max_abs_r'])


# ==========================================================================
# 权重余弦
# ==========================================================================
@test
def test_weight_cosine_basic():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(cv.weight_cosine(a, a) - 1.0) < 1e-12
    assert abs(cv.weight_cosine(a, -a) + 1.0) < 1e-12
    b = np.array([1.0, -2.0, 3.0, -4.0])
    assert abs(cv.weight_cosine(a, b)) < 1.0
    # 含 NaN 时按共同有效值计算
    c = a.copy()
    c[0] = np.nan
    assert np.isfinite(cv.weight_cosine(a, c))
    # 有效值过少返回 NaN
    d = np.full(4, np.nan)
    assert not np.isfinite(cv.weight_cosine(a, d))


# ==========================================================================
# 收敛检验与判据
# ==========================================================================
@test
def test_convergence_passes_when_shared_and_separable():
    E, maps = _fake_data(seed=2, shared=True)
    # 让三个模态彼此略作区分，避免触发"不可分离"
    rng = np.random.default_rng(22)
    maps = {k: v + rng.normal(0, 1.2, v.size) for k, v in maps.items()}
    crit = cv.ConvergenceCriteria(max_pairwise_abs_r=0.7, min_weight_cosine=0.5,
                                 max_spin_p_fdr=0.05)
    out = cv.convergence_test(maps, E, [f'g{i}' for i in range(E.shape[1])],
                             criteria=crit)
    assert out['n_modalities_analysed'] == 3
    assert np.isfinite(out['min_pairwise_cosine'])
    assert out['min_pairwise_cosine'] > 0.5, out['min_pairwise_cosine']
    assert isinstance(out['converged'], bool)
    # 无 spin 输入时 spin 判据不满足 → 整体不应判为收敛
    assert out['criteria_met']['spin_p_fdr'] is False
    assert out['converged'] is False


@test
def test_convergence_fails_when_maps_collinear():
    """三个模态几乎相同 → 不可分离，应触发回退路径，并标注正交化病态。

    注意：**不能**断言"正交化后残差之间相关为 0"。当模态高度共线时（正是触发该回退
    路径的情形），回归设计矩阵病态，残差之间仍会有非零相关（实测 |r| 可达 0.5）。
    正确做法是报告条件数并声明残差分析不可采信。
    """
    E = np.random.default_rng(3).normal(size=(200, 300))
    w = np.random.default_rng(4).normal(size=300)
    base = E @ w
    maps = {f'M{i}': base + np.random.default_rng(10 + i).normal(0, 0.01, 200)
            for i in range(3)}
    crit = cv.ConvergenceCriteria()
    out = cv.convergence_test(maps, E, [f'g{i}' for i in range(300)], criteria=crit)
    assert out['separable'] is False
    assert out['converged'] is False
    assert 'fallback' in out
    fb = out['fallback']
    # 残差相关应被**报告**（数值可能非零，这正是病态的证据）
    assert len(fb['residual_pairwise_r']) == 3
    assert all(np.isfinite(v) for v in fb['residual_pairwise_r'].values())
    # 解释率应接近 1（彼此高度共线，几乎完全可被对方解释）
    for v in fb['residual_variance_explained'].values():
        assert v > 0.9, v
    # 关键：必须标注病态与不可采信。注意条件数按构造必为 inf（截距列方差为 0），
    # 因此不能用 "> 1e6" 这种会因 inf 比较而失效的写法来检验——实际踩到过：
    # 早期实现用 np.nanmax 求最大条件数，inf 被折成 NaN，病态保护静默失效。
    assert fb['ill_conditioned'] is True, fb['max_condition_number']
    assert (not np.isfinite(fb['max_condition_number'])
            or fb['max_condition_number'] > 1e6), fb['max_condition_number']
    assert 'interpretation_warning' in fb
    # 报告中也应出现该警告
    assert '病态' in cv.render_convergence_report(out)


@test
def test_convergence_with_spin_gives_significance():
    from tnwt.spin import gen_spinsamples
    E, maps = _fake_data(seed=5, shared=True)
    rng = np.random.default_rng(55)
    maps = {k: v + rng.normal(0, 1.5, v.size) for k, v in maps.items()}
    n = E.shape[0]
    coords = np.random.default_rng(6).normal(size=(n, 3))
    coords /= np.linalg.norm(coords, axis=1, keepdims=True)
    spins = gen_spinsamples(coords * 60, n_perm=300, seed=7)
    crit = cv.ConvergenceCriteria()
    out = cv.convergence_test(maps, E, [f'g{i}' for i in range(E.shape[1])],
                             criteria=crit, spins_per_modality={k: spins for k in maps})
    assert 'spin_p' in out and len(out['spin_p']) == 3
    assert 'spin_q' in out and 'spin_fdr_significant' in out
    # 潜因子真实存在，spin 应显著
    assert all(out['spin_fdr_significant'].values()), out['spin_fdr_significant']
    assert np.isfinite(out['min_pairwise_cosine'])
    assert out['criteria_met']['spin_p_fdr'] is True


@test
def test_convergence_skips_insufficient_parcels():
    E = np.random.default_rng(8).normal(size=(5, 50))
    maps = {'M': np.arange(5, dtype=float)}
    out = cv.convergence_test(maps, E, [f'g{i}' for i in range(50)],
                             criteria=cv.ConvergenceCriteria())
    assert out['n_modalities_analysed'] == 0
    assert out.get('warnings')


# ==========================================================================
# 集合重叠率
# ==========================================================================
@test
def test_overlap_coefficients_known_values():
    a = {'x', 'y', 'z', 'w'}
    b = {'x', 'y'}
    out = cv.overlap_coefficients(a, b)
    assert out['n_intersection'] == 2
    assert abs(out['jaccard'] - 2 / 4) < 1e-12
    assert abs(out['overlap_coefficient'] - 2 / 2) < 1e-12
    # 空集：overlap coefficient 无定义（NaN），jaccard 为 0
    e = cv.overlap_coefficients(set(), b)
    assert e['jaccard'] == 0.0
    assert not np.isfinite(e['overlap_coefficient'])


@test
def test_top_gene_set_modes_agree():
    z = np.array([3.0, -2.5, 1.0, 0.5, -0.2, 2.1])
    genes = [f'g{i}' for i in range(6)]
    assert cv.top_gene_set(z, genes, threshold=2.0) == {'g0', 'g1', 'g5'}
    assert cv.top_gene_set(z, genes, top_n=3) == {'g0', 'g1', 'g5'}
    assert len(cv.top_gene_set(z, genes, top_frac=0.5)) == 3
    try:
        cv.top_gene_set(z, genes)
    except ValueError:
        pass
    else:
        raise AssertionError('未给阈值应报错')


@test
def test_top_gene_set_handles_nan():
    z = np.array([3.0, np.nan, 1.0])
    s = cv.top_gene_set(z, ['a', 'b', 'c'], threshold=2.0)
    assert s == {'a'}, s


@test
def test_robustness_matrix_identical_variants():
    rng = np.random.default_rng(9)
    z = rng.normal(size=500)
    variants = {'same': z.copy(), 'noisy': z + rng.normal(0, 0.1, 500)}
    out = cv.robustness_matrix(z, [f'g{i}' for i in range(500)], variants, threshold=2.0)
    assert out['variants']['same']['overlap_coefficient'] == 1.0
    assert out['variants']['noisy']['overlap_coefficient'] > 0.8
    assert out['min_overlap_coefficient'] >= 0.8


@test
def test_robustness_matrix_handles_length_mismatch():
    z = np.random.default_rng(10).normal(size=100)
    out = cv.robustness_matrix(z, [f'g{i}' for i in range(100)],
                               {'bad': np.zeros(50)}, threshold=2.0)
    assert 'note' in out['variants']['bad']


# ==========================================================================
# 外部一致性
# ==========================================================================
@test
def test_external_consistency_detects_agreement():
    rng = np.random.default_rng(11)
    z = rng.normal(size=400)
    genes = [f'g{i}' for i in range(400)]
    same = z + rng.normal(0, 0.1, 400)
    out = cv.external_consistency(z, same, genes)
    assert out['n_common_genes'] == 400
    assert out['pearson_r'] > 0.9
    assert out['cosine'] > 0.9
    assert out['sign_agreement_rate'] > 0.9


@test
def test_external_consistency_flags_insufficient_overlap():
    """共同基因少于 5 个时应返回说明而不是伪造相关。
    注意：本检验要求外部权重长度与基因名一致，长度不足时会在函数入口先报错，
    因此这里通过**较短向量**触发"共同基因不足"的分支。"""
    z = np.arange(4, dtype=float)
    out = cv.external_consistency(z, z, [f'g{i}' for i in range(4)])
    assert 'note' in out, out
    assert out['n_common_genes'] == 4


@test
def test_external_consistency_length_check():
    try:
        cv.external_consistency(np.arange(5.0), np.arange(3.0), ['a', 'b', 'c', 'd', 'e'])
    except ValueError:
        return
    raise AssertionError('长度不一致应报错')


# ==========================================================================
# 报告
# ==========================================================================
@test
def test_render_convergence_report_contains_key_fields():
    E, maps = _fake_data(seed=12, shared=True)
    rng = np.random.default_rng(120)
    maps = {k: v + rng.normal(0, 1.5, v.size) for k, v in maps.items()}
    out = cv.convergence_test(maps, E, [f'g{i}' for i in range(E.shape[1])],
                             criteria=cv.ConvergenceCriteria())
    md = cv.render_convergence_report(out)
    for token in ('跨模态收敛检验', '可分离性', '权重余弦', '判据逐条核对', '结论'):
        assert token in md, token
    assert '## ' in md


@test
def test_d11_excludes_null_modality_from_convergence():
    """D11：组间图谱未过 FDR 的模态不参与收敛判定。

    被剔除的模态仍需算出权重（其转录组关联作探索性结果），
    但不得进入 ``converged`` 判定，且两两余弦只统计参与判定的模态。
    """
    E, maps = _fake_data(n_par=200, n_gene=300, seed=31, shared=True)
    rng = np.random.default_rng(310)
    # 三个模态共享潜因子且彼此略作区分 → 若全部参与，本应可能收敛
    maps = {f'M{i}': v + rng.normal(0, 1.2, v.size) for i, v in enumerate(maps.values())}
    maps = {f'M{i}': v for i, v in enumerate(maps.values())}
    crit = cv.ConvergenceCriteria()
    excluded = {'M2': 'FDR 后 0 个显著分区'}

    out = cv.convergence_test(maps, E, [f'g{i}' for i in range(E.shape[1])],
                             criteria=crit, excluded_modalities=excluded)
    assert out['excluded_modalities'] == excluded
    assert out['participating_modalities'] == ['M0', 'M1']
    assert 'M2' in out['modalities']
    assert 'exclusion_note' in out
    # 两两余弦只应包含 M0|M1
    assert set(out['weight_cosine']) == {'M0|M1'}, out['weight_cosine']
    # M2 仍应被计算权重（探索性报告需要）
    assert 'M2' in out.get('var_y', {})
    # 结论中必须提及 D11 与剔除原因
    md = cv.render_convergence_report(out)
    assert 'D11' in md and 'M2' in md
    assert '探索性' in md


@test
def test_d11_insufficient_participants_blocks_convergence():
    """剔除后只剩 1 个模态时，不得判为收敛，并须明确说明无法判定。"""
    E, maps = _fake_data(n_par=150, n_gene=200, seed=41, shared=True)
    crit = cv.ConvergenceCriteria()
    excluded = {k: 'FDR 后 0 个显著分区' for k in ('M1', 'M2')}
    out = cv.convergence_test(maps, E, [f'g{i}' for i in range(200)],
                             criteria=crit, excluded_modalities=excluded)
    assert out['participating_modalities'] == ['M0']
    assert out['converged'] is False
    assert out.get('insufficient_participants') is True
    assert out['criteria_met']['enough_participants'] is False
    assert '无法' in out.get('note', '')
    md = cv.render_convergence_report(out)
    assert '⚠️' in md


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
