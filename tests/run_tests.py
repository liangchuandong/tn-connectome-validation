#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""零依赖测试：配置解析与统计工具。

运行：python tests/run_tests.py
退出码 0 表示全部通过。

统计函数的参考值来自解析解或 scipy 公开参考值（本环境无 scipy，
故使用可手算或已知精度的常数作为基准，并在注释中给出出处）。
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from tnwt import config_lite as cl  # noqa: E402
from tnwt import stats_lite as sl  # noqa: E402

TESTS: list[tuple[str, callable]] = []


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


# --------------------------------------------------------------------------
# config_lite
# --------------------------------------------------------------------------
@test
def test_parse_scalars():
    assert cl.parse_scalar('1') == 1 and isinstance(cl.parse_scalar('1'), int)
    assert cl.parse_scalar('1.0') == 1.0 and isinstance(cl.parse_scalar('1.0'), float)
    assert cl.parse_scalar('"1.8.0"') == '1.8.0'          # 引号保护版本号
    assert cl.parse_scalar('1.8.0') == '1.8.0'            # 非数字字面量退化为字符串
    assert cl.parse_scalar('true') is True and cl.parse_scalar('false') is False
    assert cl.parse_scalar('null') is None and cl.parse_scalar('~') is None
    assert cl.parse_scalar('[a, b, 3]') == ['a', 'b', 3]
    assert cl.parse_scalar('{a: 1, b: x}') == {'a': 1, 'b': 'x'}
    assert cl.parse_scalar('[]') == [] and cl.parse_scalar('{}') == {}


@test
def test_parse_nested_and_block_list():
    text = (
        'top:\n'
        '  child: 5\n'
        '  items:\n'
        '    - a\n'
        '    - name: x\n'
        '      value: 2\n'
        'list:\n'
        '  - 1\n'
        '  - 2\n'
    )
    got = cl.parse(text)
    assert got['top']['child'] == 5, got
    assert got['top']['items'][0] == 'a', got
    assert got['top']['items'][1] == {'name': 'x', 'value': 2}, got
    assert got['list'] == [1, 2], got


@test
def test_comments_and_blank_lines_ignored():
    got = cl.parse('# 头部注释\n\na: 1  # 行尾注释\n\n  # 缩进注释\nb: "x # not comment"\n')
    assert got == {'a': 1, 'b': 'x # not comment'}, got


@test
def test_reject_tab_indent():
    try:
        cl.parse('a:\n\tb: 1\n')
    except cl.ConfigError:
        return
    raise AssertionError('tab 缩进应被拒绝')


@test
def test_reject_duplicate_key():
    try:
        cl.parse('a: 1\na: 2\n')
    except cl.ConfigError:
        return
    raise AssertionError('重复键应被拒绝')


@test
def test_reject_anchors_and_tags():
    for bad in ('a: &anchor 1\n', 'a: *ref\n', 'a: !tag 1\n'):
        try:
            cl.parse(bad)
        except cl.ConfigError:
            continue
        raise AssertionError(f'应拒绝：{bad!r}')


@test
def test_load_project_config():
    cfg = cl.load(ROOT / 'config' / 'analysis.yaml')
    assert cfg['dataset']['snapshot_tag'] == 'v2.0.3', cfg['dataset']['snapshot_tag']
    assert cfg['dataset']['license'] == 'CC0'
    assert cfg['dataset']['bids_version'] == '1.8.0'
    assert cl.dig(cfg, 'cohort.patients') == 119
    assert cl.dig(cfg, 'cohort.controls') == 53
    assert cl.dig(cfg, 'cohort.exclude_no_clinical') == ['sub-23', 'sub-31']
    assert cl.dig(cfg, 'cohort.surgery_groups.MVD') == 79
    assert cl.dig(cfg, 'cohort.surgery_groups.PI') == 31
    assert cl.dig(cfg, 'atlas.primary').startswith('Schaefer2018_200')
    assert len(cl.dig(cfg, 'atlas.sensitivity')) == 2
    assert cl.dig(cfg, 'confounds.age_mismatch_p') == 0.0167
    assert cl.dig(cfg, 'spin.n_permutations') == 5000
    assert cl.dig(cfg, 'convergence.max_pairwise_abs_r') == 0.7
    assert cl.dig(cfg, 'pls.stability.candidate_gene_min_selection_freq') == 0.8
    assert cl.dig(cfg, 'phenotypes.A3.myelination_enrichment_role') == 'positive_control'
    assert cl.dig(cfg, 'phenotypes.A3.circularity_controls') == [
        'parallel_metric_replication', 'demyelination_residual_pls']
    assert cl.dig(cfg, 'longitudinal.covariates') == ['followup_interval_days']
    assert cl.dig(cfg, 'clinical_stratification.metrics') == ['auc', 'pr_auc', 'ci', 'permutation_null']
    assert cl.dig(cfg, 'seeds.spin') == 20261018
    assert cl.dig(cfg, 'does.not.exist', 'fallback') == 'fallback'


# --------------------------------------------------------------------------
# stats_lite：与解析解 / 已知参考值对齐
# --------------------------------------------------------------------------
@test
def test_chi2_sf_known_values():
    # chi2.sf(3.841459, 1) = 0.05（scipy 参考）
    assert abs(sl.chi2_sf(3.841459, 1) - 0.05) < 1e-4, sl.chi2_sf(3.841459, 1)
    # chi2.sf(5.991465, 2) = 0.05
    assert abs(sl.chi2_sf(5.991465, 2) - 0.05) < 1e-4, sl.chi2_sf(5.991465, 2)
    # chi2.sf(0, k) = 1
    assert sl.chi2_sf(0.0, 3) == 1.0


@test
def test_t_sf_known_values():
    # 2*t.sf(2.228139, 10) = 0.05
    assert abs(2 * sl.t_sf(2.228139, 10) - 0.05) < 1e-3, 2 * sl.t_sf(2.228139, 10)
    # 2*t.sf(1.959964, 1e6) ≈ 0.05（趋近正态）
    assert abs(2 * sl.t_sf(1.959964, 1_000_000) - 0.05) < 1e-3
    # df=inf 必须按标准正态处理，不能返回 NaN
    # （历史 bug：inf/inf = NaN → p 静默变为 NaN，上层 min(1.0, ...) 后表现为 p=1.0）
    # 注意：本实现把 inf 映射为 1e12，故结果与真正态有约 8e-6 的偏差
    assert abs(2 * sl.t_sf(1.959964, float('inf')) - 0.05) < 1e-4
    assert abs(2 * sl.t_sf(5.0, float('inf')) - 5.733e-7) < 1e-8
    assert not np.isnan(sl.t_sf(5.0, float('inf')))
    # df=nan / df<=0 也应稳健
    assert not np.isnan(sl.t_sf(5.0, float('nan')))
    assert not np.isnan(sl.t_sf(5.0, 0.0))

@test
def test_welch_matches_manual_formula():
    a = [1.2, 1.5, 1.9, 2.2, 2.6, 3.0, 3.4]
    b = [0.8, 1.1, 1.3, 1.6, 1.9, 2.1]
    r = sl.welch_ttest(a, b)
    import numpy as np
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    na, nb = len(a), len(b)
    se2 = va / na + vb / nb
    t_manual = (np.mean(a) - np.mean(b)) / np.sqrt(se2)
    df_manual = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    assert abs(r['t'] - t_manual) < 1e-12
    assert abs(r['df'] - df_manual) < 1e-12
    # 已知参考：t=2.1852, df=10.1351, p=0.0534
    assert abs(r['t'] - 2.1852322553) < 1e-6
    assert abs(r['p'] - 0.0534188186) < 1e-6, r['p']
    # 交换两组：|t| 不变、符号相反
    r2 = sl.welch_ttest(b, a)
    assert abs(r2['t'] + r['t']) < 1e-12


@test
def test_welch_edge_cases():
    r = sl.welch_ttest([1.0], [2.0, 3.0])
    assert r['n_a'] == 1 and r['p'] != r['p']  # NaN
    r2 = sl.welch_ttest([1.0, float('nan'), 3.0], [1.0, 2.0, 4.0])
    assert r2['n_a'] == 2, r2  # NaN 应被剔除


@test
def test_chi2_contingency_known():
    # 2x2 表，scipy: chi2=0.0 当完全独立
    r = sl.chi2_contingency([[10, 10], [10, 10]])
    assert abs(r['chi2']) < 1e-12 and abs(r['p'] - 1.0) < 1e-9
    # 强关联表应显著
    r2 = sl.chi2_contingency([[30, 5], [5, 30]])
    assert r2['p'] < 1e-6, r2


@test
def test_hedges_g_shrinkage():
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [0.5, 1.5, 2.5, 3.5, 4.5]
    d = sl.cohen_d(a, b)
    g = sl.hedges_g(a, b)
    n = len(a) + len(b)
    j = 1 - 3 / (4 * n - 9)
    assert abs(g - d * j) < 1e-12
    assert abs(g) < abs(d)  # 收缩


@test
def test_betainc_identities():
    # I_x(a,b) + I_{1-x}(b,a) = 1
    for a, b, x in [(0.5, 0.5, 0.3), (2.0, 3.0, 0.7), (5.0, 1.0, 0.42)]:
        s = sl.betainc_reg(a, b, x) + sl.betainc_reg(b, a, 1 - x)
        assert abs(s - 1.0) < 1e-10, (a, b, x, s)


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
    total = len(TESTS)
    print(f'\n{total - failures}/{total} 通过')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
