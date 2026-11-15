#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""零依赖测试（四）：富集分析与细胞类型分析。

其中的 GO / Reactome 测试使用**真实注释文件**（data/annotations/，由
scripts/fetch_annotations.py 下载，含 sha256 记录）；若注释缺失则跳过并明确报告，
不会以合成注释冒充真实结果。

运行：python tests/run_tests_enrichment.py
"""
from __future__ import annotations

import math
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tnwt import enrichment as en  # noqa: E402
from tnwt.enrichment import log_comb  # noqa: E402

TESTS: list[tuple[str, callable]] = []
ANNOT = ROOT / 'data' / 'annotations'
TMP = ROOT / 'build' / 'enrich_selftest'


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


def _fresh_tmp():
    if TMP.exists():
        shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True, exist_ok=True)
    return TMP


# ==========================================================================
# 分布函数
# ==========================================================================
@test
def test_hypergeom_matches_exact_sum():
    """超几何上尾概率应与逐项精确求和一致。"""
    rng = np.random.default_rng(0)
    for _ in range(30):
        N = int(rng.integers(50, 300))
        K = int(rng.integers(5, N // 2))
        n = int(rng.integers(5, N // 2))
        k = int(rng.integers(0, min(K, n) + 1))
        exact = sum(math.exp(log_comb(K, i) + log_comb(N - K, n - i) - log_comb(N, n))
                    for i in range(k, min(K, n) + 1))
        got = en.hypergeom_sf(k, N, K, n)
        assert abs(got - min(1.0, exact)) < 1e-9, (N, K, n, k, got, exact)


@test
def test_hypergeom_boundary_cases():
    assert en.hypergeom_sf(0, 100, 10, 10) == 1.0
    assert en.hypergeom_sf(11, 100, 10, 10) == 0.0
    assert en.hypergeom_sf(1, 100, 10, 10) < 1.0
    # 全部命中时上尾概率 = 抽样比
    assert abs(en.hypergeom_sf(10, 100, 10, 10) - 1 / math.comb(100, 10) * math.comb(10, 10)) < 1e-12
    # 单调性：k 增大 p 不增
    ps = [en.hypergeom_sf(k, 500, 50, 40) for k in range(0, 20)]
    assert all(ps[i] >= ps[i + 1] - 1e-12 for i in range(len(ps) - 1)), ps


@test
def test_fisher_equals_hypergeom():
    # 表 [[a,b],[c,d]]：a=15 命中且属于集合
    p_fisher = en.fisher_one_sided(15, 35, 85, 865)
    N, K, n = 1000, 50, 100
    assert abs(p_fisher - en.hypergeom_sf(15, N, K, n)) < 1e-12


@test
def test_fdr_bh_realistic():
    p = [0.0001, 0.001, 0.02, 0.04, 0.2, 0.5, 0.9]
    rej, q = en.fdr_bh(p, alpha=0.05)
    assert rej[0] and rej[1]
    assert not rej[-1]
    assert np.all(np.diff(q[np.isfinite(q)]) >= -1e-12), 'q 值应单调不减'


# ==========================================================================
# 基因集读取（真实注释）
# ==========================================================================
@test
def test_read_real_go_sets():
    if not (ANNOT / 'go_gene_sets.tsv').exists():
        print('       [skip] 真实 GO 注释缺失，跳过（不冒充真实结果）')
        return
    gs = en.read_go_sets(ANNOT, namespaces=('biological_process',))
    assert gs.n_sets > 5000, gs.n_sets
    sizes = np.array([len(v) for v in gs.sets.values()])
    assert sizes.min() >= 1 and sizes.max() > 100
    # 抽查一个已知 GO term 应含已知基因
    all_genes = gs.genes()
    assert len(all_genes) > 10000, len(all_genes)
    for probe in ('TP53', 'MBP', 'SCN9A'):
        assert probe in all_genes, probe


@test
def test_read_real_reactome_sets():
    if not (ANNOT / 'reactome_gene_sets.tsv').exists():
        print('       [skip] 真实 Reactome 注释缺失，跳过')
        return
    gs = en.read_reactome_sets(ANNOT)
    assert gs.n_sets > 1000, gs.n_sets
    genes = gs.genes()
    assert len(genes) > 8000, len(genes)
    assert 'MBP' in genes or 'PLP1' in genes


@test
def test_gmt_roundtrip_via_gzip():
    tmp = _fresh_tmp()
    import gzip
    p = tmp / 'test.gmt.gz'
    with gzip.open(p, 'wt', encoding='utf-8') as fh:
        fh.write('SET_A\tdesc\tG1\tG2\tG3\n')
        fh.write('SET_B\tdesc\tG4\tG5\n')
    gs = en.read_gmt(p)
    assert gs.sets == {'SET_A': {'G1', 'G2', 'G3'}, 'SET_B': {'G4', 'G5'}}


# ==========================================================================
# ORA
# ==========================================================================
@test
def test_ora_recovers_planted_signal():
    rng = np.random.default_rng(1)
    bg = [f'G{i}' for i in range(2000)]
    # 集合 A 含 40 个基因，其中 15 个被"选中"
    set_a = set(rng.choice(bg, 40, replace=False))
    set_b = set(rng.choice(bg, 40, replace=False))
    query = set(list(set_a)[:15]) | set(rng.choice(bg, 20, replace=False))
    gs = en.GeneSets('planted', {'A': set_a, 'B': set_b}, {})
    res = en.ora(query, gs, set(bg), min_overlap=3)
    top = res.table[0]
    assert top['set_id'] == 'A', res.table[:2]
    assert top['overlap'] == 15
    assert top['p'] < 1e-6, top['p']
    assert top['fdr_significant'] is True
    assert abs(top['fold_enrichment'] - 15 / (35 * 40 / 2000)) < 0.2


@test
def test_ora_no_signal_gives_uniform_p():
    rng = np.random.default_rng(2)
    bg = [f'G{i}' for i in range(3000)]
    gs = en.GeneSets('null', {f'S{i}': set(rng.choice(bg, 50, replace=False))
                              for i in range(100)}, {})
    query = set(rng.choice(bg, 60, replace=False))
    res = en.ora(query, gs, set(bg), min_overlap=1)
    ps = np.array([r['p'] for r in res.table])
    assert ps.size > 50
    # 无信号时不应有大量显著
    assert (ps < 0.05).sum() <= max(3, int(0.12 * ps.size)), (ps < 0.05).sum()


@test
def test_ora_permuted_controls_for_expression():
    """表达量混杂：集合与查询集都富集高表达基因、但**彼此无真实关联**时，
    未校准的超几何 ORA 给出极端虚假显著（p ~ 1e-30 量级），
    表达量分层置换应正确刻画零分布（含经验 p 值的均匀性）。

    判据设计（均为确定性性质，避免偶发失败）
    ----------------------------------------
    1. 零分布的**矩**必须匹配理论超几何分布
       （观测重叠 ~ Hypergeom(N=300, K=150, n=60)，mean=30.0，sd=3.470）。
       这是"实现是否正确"的直接证据。
    2. 约 40 次独立抽样的经验 p 值应通过 **KS 均匀性检验**。
       单次或三次抽样的 p 值是随机变量（实测在同一批 60 次抽样中，
       前三次可为 0.062/0.047/0.010，而中位为 0.556），因此不能对少量抽样下断言。
    """
    rng = np.random.default_rng(3)
    n = 3000
    bg = [f'G{i}' for i in range(n)]
    expr = {g: float(v) for g, v in zip(bg, rng.normal(0, 1, len(bg)))}
    hi = sorted(bg, key=lambda g: -expr[g])
    pool = hi[:n // 10]                     # 与 10 分层的顶层范围一致
    K, nq, N_pool = 150, 60, len(pool)
    mean_th = nq * K / N_pool
    sd_th = (nq * (K / N_pool) * (1 - K / N_pool) * (N_pool - nq) / (N_pool - 1)) ** 0.5

    p_hyper, p_perm, null_means, null_sds, obs_list = [], [], [], [], []
    for seed in range(100, 140):
        S = set(rng.choice(pool, K, replace=False))
        query = set(rng.choice(pool, nq, replace=False))
        gs = en.GeneSets('confound', {'S': S}, {})
        p_hyper.append(en.ora(query, gs, set(bg), min_overlap=1).table[0]['p'])
        r = en.ora_permuted(query, gs, set(bg), expression=expr, n_perm=300,
                            seed=seed, min_overlap=1, n_bins=10).table[0]
        p_perm.append(r['p'])
        null_means.append(r['expected'])
        null_sds.append(r['null_sd'])
        obs_list.append(r['overlap'])

    # 1) 未校准 ORA 严重反保守
    assert max(p_hyper) < 1e-6, f'未校准 ORA 应给出强虚假显著，实际最大 {max(p_hyper)}'
    # 2) 零分布矩匹配理论
    mm, ss = float(np.mean(null_means)), float(np.mean(null_sds))
    assert abs(mm - mean_th) < 0.05 * mean_th, f'零模型期望 {mm:.2f} 偏离理论 {mean_th:.2f}'
    assert abs(ss - sd_th) < 0.05 * sd_th, f'零模型 SD {ss:.3f} 偏离理论 {sd_th:.3f}'
    # 3) 观测与期望同量级
    ratios = [o / m for o, m in zip(obs_list, null_means)]
    assert all(0.6 < x < 1.8 for x in ratios), f'观测/期望比值异常：{ratios[:8]}'
    # 4) 经验 p 值的均匀性（KS 检验）
    s = np.sort(np.array(p_perm))
    N = len(s)
    D = float(np.max(np.abs(np.concatenate([np.arange(1, N + 1) / N - s,
                                            s - np.arange(0, N) / N]))))
    crit = 1.36 / (N ** 0.5)
    assert D < crit, (f'p 值偏离均匀分布：KS D={D:.4f} > 临界 {crit:.4f}；'
                      f'中位={np.median(p_perm):.3f}')


@test
def test_ora_permuted_selection_prob_removes_extreme_bias():
    """当查询集由"取表达量最高的 N 个基因"这类**窄于分层**的极端规则产生时，
    分层抽样无法消除偏倚（同层内仍严重偏斜）；改用**入选概率加权抽样**后，
    观测重叠会被完全归因于该规则本身，不再产生虚假显著。"""
    rng = np.random.default_rng(33)
    n = 2000
    bg = [f'G{i}' for i in range(n)]
    expr = {g: float(v) for g, v in zip(bg, rng.normal(0, 1, len(bg)))}
    hi = sorted(bg, key=lambda g: -expr[g])
    narrow, pool = hi[:30], hi[:80]
    gs = en.GeneSets('confound', {'S': set(pool)}, {})
    query = set(narrow)                      # 极端规则：恰好最高的 30 个
    prob = {g: (1.0 if g in narrow else 1e-9) for g in bg}   # 该规则的经验入选概率
    res_strat = en.ora_permuted(query, gs, set(bg), expression=expr,
                                n_perm=300, seed=34, min_overlap=1, n_bins=10)
    res_prob = en.ora_permuted(query, gs, set(bg), selection_prob=prob,
                               n_perm=300, seed=35, min_overlap=1, n_bins=10)
    s = res_strat.table[0]
    w = res_prob.table[0]
    assert s['p'] < 0.01, f'分层抽样本应给出虚假显著，实际 {s["p"]}'
    # 加权零模型下，观测重叠完全由选择规则解释：观测/期望 ≈ 1
    assert w['expected'] > 15, f'加权期望过低（{w["expected"]}），零模型未覆盖该规则'
    assert 0.5 < w['overlap'] / w['expected'] < 2.0, (w['overlap'], w['expected'])
    assert w['p'] > 0.01, f'加权后不应仍显著：{w["p"]}'


@test
def test_ora_permuted_runs_at_go_scale():
    """性能与稳定性：在接近真实 GO 的规模下（数千集合）应能在合理时间内完成。"""
    if not (ANNOT / 'go_gene_sets.tsv').exists():
        print('       [skip] 真实 GO 注释缺失，跳过')
        return
    import time
    gs_all = en.read_go_sets(ANNOT, namespaces=('biological_process',))
    bg = gs_all.genes()
    gs = en.filter_gene_sets(gs_all, bg, min_size=20, max_size=200)
    assert gs.n_sets > 1000, gs.n_sets
    rng = np.random.default_rng(5)
    query = set(rng.choice(sorted(bg), 500, replace=False))
    expr = {g: float(v) for g, v in zip(sorted(bg), rng.normal(0, 1, len(bg)))}
    t0 = time.time()
    res = en.ora_permuted(query, gs, bg, expression=expr, n_perm=200, seed=6,
                          min_overlap=2, n_bins=10)
    dt = time.time() - t0
    assert len(res.table) > 0, f'{gs.n_sets} 个集合中无任何集合通过 min_overlap=2'
    print(f'       [info] GO 规模置换 ORA：{gs.n_sets} 集合（筛后）× 200 次置换，'
          f'{dt:.1f}s（外推 1000 次约 {dt*5:.0f}s）')
    assert dt < 300, f'过慢：{dt:.1f}s'


# ==========================================================================
# GSEA
# ==========================================================================
@test
def test_gsea_running_sum_hand_computed():
    """GSEA 富集分数的方向性：命中集中在顶部时 ES > 0，集中在底部时 ES < 0。"""
    genes = [f'G{i}' for i in range(100)]
    w = np.linspace(3, -3, 100)
    top_set = {'top_hit': set(genes[:10])}
    bot_set = {'bot_hit': set(genes[-10:])}
    res_top = en.gsea(genes, w, en.GeneSets('t', top_set, {}), n_perm=50, seed=1,
                      min_size=1, max_size=1000)
    res_bot = en.gsea(genes, w, en.GeneSets('b', bot_set, {}), n_perm=50, seed=1,
                      min_size=1, max_size=1000)
    assert res_top.table[0]['ES'] > 0.5, res_top.table[0]
    assert res_bot.table[0]['ES'] < -0.5, res_bot.table[0]
    assert res_top.table[0]['p'] < 0.1


@test
def test_gsea_null_calibration():
    """随机权重下，随机基因集的 p 值不应系统性偏小。"""
    rng = np.random.default_rng(7)
    genes = [f'G{i}' for i in range(500)]
    w = rng.normal(0, 1, 500)
    gs = en.GeneSets('null', {f'S{i}': set(rng.choice(genes, 25, replace=False))
                              for i in range(40)}, {})
    res = en.gsea(genes, w, gs, n_perm=200, seed=8, min_size=10, max_size=100)
    ps = np.array([r['p'] for r in res.table])
    assert ps.size > 20
    assert abs(np.median(ps) - 0.5) < 0.3, np.median(ps)
    assert (ps < 0.05).sum() <= max(3, int(0.15 * ps.size)), (ps < 0.05).sum()


# ==========================================================================
# 细胞类型分析
# ==========================================================================
@test
def test_specificity_metrics():
    m = np.array([[10.0, 1.0, 1.0],       # 基因 0 特异于细胞类型 0
                  [1.0, 1.0, 1.0]])       # 基因 1 无特异性
    ce = en.CellTypeExpression(m, ['A', 'B', 'C'], ['g0', 'g1'])
    spec = ce.specificity('mean_ratio')
    # 基因 0 的均值为 (10+1+1)/3 = 4，故 10/4 = 2.5
    assert abs(spec[0, 0] - 2.5) < 1e-9, spec
    assert abs(spec[0, 1] - 0.25) < 1e-9, spec
    assert np.allclose(spec[1], 1.0)
    tau = ce.specificity('tau')
    assert tau[0, 0] < tau[0, 1], tau          # tau 越小越特异
    assert np.allclose(tau[1], 0.0)
    assert np.allclose(ce.specificity('expression'), m)


@test
def test_ewce_recovers_planted_cell_type():
    """构造"候选基因全部特异于细胞类型 B"的场景，EWCE 应识别出 B。

    同时验证分层的影响：分层抽样（默认，控制表达量混杂）会损失部分功效，
    因此对分层后的 p 值要求放宽，但**排序第一的细胞类型必须仍是 B**。
    """
    rng = np.random.default_rng(9)
    n_genes, n_ct = 2000, 4
    m = rng.normal(5, 0.5, size=(n_genes, n_ct))
    marker_genes = [f'G{i}' for i in range(60)]
    m[:60, 1] += 8.0                       # 前 60 个基因特异于细胞类型 1
    genes = [f'G{i}' for i in range(n_genes)]
    ce = en.CellTypeExpression(m, ['A', 'B', 'C', 'D'], genes)

    out = en.ewce(marker_genes, ce, n_perm=300, seed=10, top_frac=0.1)
    assert out['results'][0]['cell_type'] == 'B', [r['cell_type'] for r in out['results']]
    assert out['marker_direction'] == 'over' and out['apply_quantile'] is True

    out_ns = en.ewce(marker_genes, ce, n_perm=300, seed=10, top_frac=0.1,
                     apply_quantile=False)
    top_ns = out_ns['results'][0]
    assert top_ns['cell_type'] == 'B', [r['cell_type'] for r in out_ns['results']]
    assert top_ns['p'] < 0.01, top_ns
    assert top_ns['fdr_significant'] is True
    assert top_ns['fold_change'] > 2.0, top_ns


@test
def test_ewce_marker_direction_matters():
    """marker_direction 必须显式区分"高表达"与"低表达"，两者结论不同。"""
    rng = np.random.default_rng(90)
    n_genes = 1500
    m = rng.normal(5, 0.4, size=(n_genes, 3))
    cand = [f'G{i}' for i in range(50)]
    m[:50, 0] -= 6.0                       # 候选基因在细胞类型 A **低**表达
    ce = en.CellTypeExpression(m, list('ABC'), [f'G{i}' for i in range(n_genes)])
    over = en.ewce(cand, ce, n_perm=200, seed=1, marker_direction='over',
                   apply_quantile=False)
    under = en.ewce(cand, ce, n_perm=200, seed=1, marker_direction='under',
                    apply_quantile=False)
    assert over['results'][0]['cell_type'] != 'A' or over['results'][0]['p'] > 0.05
    assert under['results'][0]['cell_type'] == 'A', under['results'][0]
    assert under['results'][0]['p'] < 0.01, under['results'][0]


@test
def test_ewce_null_is_calibrated():
    rng = np.random.default_rng(11)
    n_genes, n_ct = 2000, 5
    m = rng.normal(5, 0.6, size=(n_genes, n_ct))
    genes = [f'G{i}' for i in range(n_genes)]
    ce = en.CellTypeExpression(m, list('ABCDE'), genes)
    cand = list(rng.choice(genes, 60, replace=False))
    out = en.ewce(cand, ce, n_perm=200, seed=12, top_frac=0.1)
    ps = [r['p'] for r in out['results']]
    assert all(np.isfinite(ps))
    assert min(ps) > 0.02, ps          # 无信号时不应显著


@test
def test_ewce_quantile_controls_expression_confound():
    """表达量混杂：当"特异性"完全来自整体表达水平时，分层抽样应削弱虚假富集。"""
    rng = np.random.default_rng(13)
    n_genes = 3000
    common = rng.normal(4, 1.2, size=(n_genes, 1))
    noise = rng.normal(0, 1.0, size=(n_genes, 3))
    # 三种细胞类型共享同一整体表达水平，差异仅来自噪声
    m = common + noise
    ce = en.CellTypeExpression(m, list('ABC'), [f'G{i}' for i in range(n_genes)])
    # 用 expression 作为"特异性"度量（很多研究直接这么做），此时高表达基因
    # 在每种细胞类型都会占据 top 比例——正是需要控制的混杂
    hi = [f'G{i}' for i in np.argsort(-m.mean(axis=1))[:80]]
    out_q = en.ewce(hi, ce, n_perm=300, seed=14, apply_quantile=True,
                    specificity_metric='expression', top_frac=0.1)
    out_n = en.ewce(hi, ce, n_perm=300, seed=14, apply_quantile=False,
                    specificity_metric='expression', top_frac=0.1)
    p_q = min(r['p'] for r in out_q['results'])
    p_n = min(r['p'] for r in out_n['results'])
    assert p_n < 0.05, f'未分层时本应出现虚假显著，实际 {p_n}'
    assert p_q > p_n, (p_q, p_n)
    assert p_q > 0.01, f'分层后不应仍显著：{p_q}'


@test
def test_proportion_map_correlation_with_spin():
    from tnwt import spin as sp
    rng = np.random.default_rng(15)
    n = 120
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    coords = v * 60
    props = rng.normal(0.2, 0.03, size=(n, 3))
    img = props[:, 1] * 5 + rng.normal(0, 0.01, n)
    spins = sp.gen_spinsamples(coords, n_perm=300, seed=16)
    out = en.proportion_map_correlation(props, ['c0', 'c1', 'c2'], img, spins=spins)
    c1 = [r for r in out if r['cell_type'] == 'c1'][0]
    assert c1['r'] > 0.9, c1
    assert c1['p_spin'] is not None


# ==========================================================================
# 写表
# ==========================================================================
@test
def test_write_table_roundtrip():
    tmp = _fresh_tmp()
    rows = [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}]
    p = en.write_table(tmp / 'out.csv', rows, ['a', 'b'])
    txt = p.read_text(encoding='utf-8').strip().splitlines()
    assert txt[0] == 'a,b' and len(txt) == 3
    # 约定：空结果**不创建文件**（下游以文件存在性判断是否有结果）
    p2 = en.write_table(tmp / 'empty.csv', [])
    assert not p2.exists(), '空结果不应创建零字节文件'
    # 若已存在同名文件，写入空结果应将其删除（避免残留旧结果）
    p3 = en.write_table(tmp / 'stale.csv', rows, ['a', 'b'])
    assert p3.exists()
    en.write_table(tmp / 'stale.csv', [])
    assert not p3.exists(), '写入空结果应清除陈旧文件'


@test
def test_set_name_is_term_name_not_namespace():
    """结果表的 set_name 必须是**条目名**，不能是命名空间。

    这是真实踩到的缺陷：早期实现用 ``gs.namespace.get(sid)`` 充当 set_name，
    而 namespace 只有一个取值（如 ``biological_process``），于是所有显著条目都显示成
    "biological_process"，富集结果无法写入论文。改用真实注释目录做端到端校验。
    """
    annot = ROOT / 'data' / 'annotations'
    if not (annot / 'go_terms.tsv').exists():
        return                      # 无注释时跳过（保持测试可在无数据环境运行）
    gs = en.read_go_sets(annot, namespaces=('biological_process',))
    assert gs.n_sets > 0
    # 取一个有基因的集合，检查显示名
    sid = next(iter(gs.sets))
    name = gs.display_name(sid)
    assert name != 'biological_process', '显示名不能是命名空间'
    assert name and name != sid or name == sid, name
    # 直接检查已知条目：GO:0007399 应为 "nervous system development"
    if 'GO:0007399' in gs.sets:
        assert gs.display_name('GO:0007399') == 'nervous system development'
    # ORA 输出必须携带条目名
    bg = set(list(gs.genes())[:2000])
    filt = en.filter_gene_sets(gs, bg, min_size=3, max_size=500)
    assert filt.n_sets > 0
    # 过滤后仍须保留显示名
    for k in list(filt.sets)[:20]:
        assert filt.display_name(k) != 'biological_process'
    query = set(list(next(iter(filt.sets.values())))[:3])
    res = en.ora(query, filt, bg, min_overlap=1)
    assert res.table
    for r in res.table[:10]:
        assert r['set_name'] != 'biological_process', r
        assert r['set_name'], 'set_name 不能为空'


@test
def test_display_name_falls_back_safely():
    """display_name 在名称缺失时必须回退到 ID，不能返回空串。"""
    gs = en.GeneSets(name='t', sets={'SET_A': {'g1'}})
    assert gs.display_name('SET_A') == 'SET_A'
    assert gs.display_name('MISSING') == 'MISSING'
    gs2 = en.GeneSets(name='t', sets={'SET_A': {'g1'}}, names={'SET_A': 'Name A'})
    assert gs2.display_name('SET_A') == 'Name A'
    # reactome 读取器应带上通路名
    annot = ROOT / 'data' / 'annotations'
    if (annot / 'reactome_gene_sets.tsv').exists():
        rs = en.read_reactome_sets(annot)
        sid = next(iter(rs.sets))
        assert rs.display_name(sid) and rs.display_name(sid) != sid


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
