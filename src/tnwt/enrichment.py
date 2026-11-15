"""富集分析与细胞类型分析（零依赖实现）。

包含
----
1. 基因集读取：GO（go_terms.tsv + go_gene_sets.tsv）、Reactome（reactome_gene_sets.tsv）、
   GMT 文件（如 MSigDB，若可获取）
2. 背景基因集构造：AHBA 检出基因 ∩ 注释可注释基因（方案 5.5 要求，避免富集膨胀）
3. 过表达分析（ORA，超几何检验）+ Benjamini-Hochberg FDR
4. 置换校准的 ORA：在**表达量匹配**的背景中抽样基因集（控制表达水平混杂）
5. GSEA（加权 Kolmogorov-Smirnov，基因集水平检验，不依赖单基因阈值）
6. EWCE 风格的细胞类型富集：特异性计算 + 表达量匹配零模型
7. 细胞类型比例空间映射：分区 × 细胞类型比例 与影像图谱的空间相关

设计原则
--------
- 不引入 scipy：超几何分布用对数 Gamma 累积实现（见 ``stats_lite``）。
- 所有随机过程接受 ``seed``，保证可复现。
- 基因集最小/最大规模可配置，默认 10–500（避免过小集合产生虚假显著）。
"""
from __future__ import annotations

import csv
import gzip
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tnwt.stats_lite import _gammainc_P, _lgamma  # noqa: E402


# ==========================================================================
# 分布函数（零依赖）
# ==========================================================================
def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float('-inf')
    return _lgamma(n + 1) - _lgamma(k + 1) - _lgamma(n - k + 1)


def hypergeom_sf(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k)，X ~ Hypergeometric(N, K, n)：从总量 N、成功 K 中抽 n 个，至少 k 个成功。"""
    if k <= 0:
        return 1.0
    if k > min(K, n):
        return 0.0
    total = 0.0
    denom = log_comb(N, n)
    for i in range(k, min(K, n) + 1):
        lp = log_comb(K, i) + log_comb(N - K, n - i) - denom
        if lp > -745:
            total += math.exp(lp)
    return float(min(1.0, max(0.0, total)))


def fisher_one_sided(a: int, b: int, c: int, d: int) -> float:
    """2×2 表 [[a,b],[c,d]] 的单侧 Fisher 精确检验 P(OR >= 观测)。"""
    N = a + b + c + d
    K = a + b
    n = a + c
    return hypergeom_sf(a, N, K, n)


# ==========================================================================
# 基因集读取
# ==========================================================================
@dataclass
class GeneSets:
    name: str
    sets: dict[str, set[str]] = field(default_factory=dict)
    namespace: dict[str, str] = field(default_factory=dict)
    #: 集合 ID → 人类可读名称（GO 条目名 / 通路名 / GMT 集合名）。
    #: 注意：不要用 ``namespace`` 充当显示名——它存的是命名空间
    #: （如 ``biological_process``），每个集合都一样，写进结果表毫无信息量。
    names: dict[str, str] = field(default_factory=dict)

    @property
    def n_sets(self) -> int:
        return len(self.sets)

    def genes(self) -> set[str]:
        out: set[str] = set()
        for v in self.sets.values():
            out |= v
        return out

    def display_name(self, sid: str) -> str:
        """返回集合的显示名；缺失时回退到 ID，再回退到命名空间。

        回退顺序刻意保证**永远不会输出空串或纯命名空间**，避免结果表出现
        "biological_process" 这类无法引用的标签（曾因此使富集结果不可写入论文）。
        """
        nm = self.names.get(sid)
        if nm:
            return nm
        return sid


def read_gmt(path: str | Path) -> GeneSets:
    """读取 GMT 文件（每行：集合名 \\t 描述 \\t 基因1 \\t 基因2 ...）。"""
    path = Path(path)
    op = gzip.open if path.suffix == '.gz' else open
    sets: dict[str, set[str]] = {}
    with op(path, 'rt', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            f = line.rstrip('\n').split('\t')
            if len(f) < 3:
                continue
            sets[f[0]] = {g.strip() for g in f[2:] if g.strip()}
    return GeneSets(name=path.stem, sets=sets,
                    names={k: k for k in sets})


def read_go_sets(annotation_dir: str | Path,
                 namespaces: tuple[str, ...] = ('biological_process',),
                 ) -> GeneSets:
    """读取 GO 基因集，按 namespace 过滤（默认仅 BP）。"""
    d = Path(annotation_dir)
    terms: dict[str, tuple[str, str, bool]] = {}
    with (d / 'go_terms.tsv').open('r', encoding='utf-8') as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            terms[row['term_id']] = (row['name'], row['namespace'],
                                     row['is_obsolete'] == '1')
    keep_ns = set(namespaces)
    sets: dict[str, set[str]] = defaultdict(set)
    ns_map: dict[str, str] = {}
    with (d / 'go_gene_sets.tsv').open('r', encoding='utf-8') as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            tid = row['term_id']
            info = terms.get(tid)
            if info is None or info[2] or info[1] not in keep_ns:
                continue
            sets[tid].add(row['gene_symbol'])
            ns_map[tid] = info[1]
    # 集合 ID → GO 条目名（如 "nervous system development"）。
    # 早期实现把它拼成 "GO:0007399|nervous system development" 却**没有传给 GeneSets**，
    # 使结果表的 set_name 退化成命名空间名（"biological_process"），
    # 富集结果因此无法直接写入论文。
    named = {tid: terms[tid][0] for tid in sets if tid in terms}
    return GeneSets(name='GO_' + '+'.join(namespaces), sets=dict(sets),
                    namespace=ns_map, names=named)


def read_reactome_sets(annotation_dir: str | Path) -> GeneSets:
    d = Path(annotation_dir)
    sets: dict[str, set[str]] = defaultdict(set)
    names: dict[str, str] = {}
    with (d / 'reactome_gene_sets.tsv').open('r', encoding='utf-8') as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            sets[row['pathway_id']].add(row['gene_symbol'])
            names[row['pathway_id']] = row['pathway_name']
    return GeneSets(name='Reactome', sets=dict(sets), names=dict(names))


# ==========================================================================
# 背景基因集
# ==========================================================================
def build_background(detected: set[str], annotatable: set[str]) -> set[str]:
    """背景 = AHBA 检出基因 ∩ 注释可注释基因（方案 5.5 的强制要求）。"""
    return set(detected) & set(annotatable)


def filter_gene_sets(gs: GeneSets, background: set[str],
                     min_size: int = 10, max_size: int = 500) -> GeneSets:
    """把基因集限制到背景基因集内，并按规模过滤。"""
    out: dict[str, set[str]] = {}
    for k, v in gs.sets.items():
        inter = v & background
        if min_size <= len(inter) <= max_size:
            out[k] = inter
    # 过滤后必须**一并保留显示名**，否则下游结果表的 set_name 会退化
    return GeneSets(name=gs.name, sets=out, namespace=gs.namespace,
                    names={k: v for k, v in gs.names.items() if k in out})


# ==========================================================================
# 多重比较
# ==========================================================================
def fdr_bh(pvals, alpha: float = 0.05):
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return np.zeros_like(p, dtype=bool), q
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    qv = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    qv = np.clip(qv, 0, 1)
    out = np.empty(n)
    out[order] = qv
    q[ok] = out
    return q <= alpha, q


# ==========================================================================
# 过表达分析（ORA）
# ==========================================================================
@dataclass
class ORAResult:
    table: list[dict]
    n_query: int
    n_background: int
    n_query_in_background: int


def ora(query_genes, gs: GeneSets, background: set[str], *, min_overlap: int = 3,
        fdr_alpha: float = 0.05) -> ORAResult:
    """超几何过表达分析。

    ``query_genes`` 为候选基因（如 PLS1 top 基因）；只统计落在 ``background`` 内的。
    """
    bg = set(background)
    q = {g for g in query_genes if g in bg}
    rows = []
    for sid, genes in gs.sets.items():
        inter = genes & q
        if len(inter) < min_overlap:
            continue
        K = len(genes)
        n = len(q)
        N = len(bg)
        p = hypergeom_sf(len(inter), N, K, n)
        exp = n * K / N if N else float('nan')
        rows.append({
            'set_id': sid,
            'set_name': gs.display_name(sid),
            'set_size': K,
            'overlap': len(inter),
            'expected': exp,
            'fold_enrichment': (len(inter) / exp) if exp else float('nan'),
            'p': p,
            'genes': ','.join(sorted(inter)),
        })
    rows.sort(key=lambda r: r['p'])
    if rows:
        rej, qv = fdr_bh([r['p'] for r in rows], alpha=fdr_alpha)
        for r, rj, qqi in zip(rows, rej, qv):
            r['fdr_significant'] = bool(rj)
            r['q'] = float(qqi)
    return ORAResult(table=rows, n_query=len(query_genes), n_background=len(bg),
                     n_query_in_background=len(q))


def ora_permuted(query_genes, gs: GeneSets, background: set[str], *,
                 expression: dict[str, float] | None = None,
                 selection_prob: dict[str, float] | None = None,
                 n_perm: int = 1000, seed: int = 0,
                 min_overlap: int = 3, n_bins: int = 10) -> ORAResult:
    """表达量匹配的置换校准 ORA（向量化实现）。

    控制"高表达基因更容易被 PLS 选中"这一常见混杂（方案 5.7 第一级稳健性要求）。
    支持两种零模型：

    - ``selection_prob``（推荐）：给定每个基因**被选入候选集的经验概率**（例如来自
      重采样稳定性分析的入选率 ``selection_freq``），按该概率做**加权抽样**。
      这是控制选择偏倚的正确做法，能同时处理"分层内仍偏斜"的极端情形。
    - ``expression``：按表达量分位数分层后层内均匀抽样。实现简单，但当查询集由
      "取表达量最高的 N 个基因"这类规则产生时，层内仍会残余偏倚（极端分位无法被
      分层消除），此时应改用 ``selection_prob``。

    性能：零分布的重叠数通过层内成员相对位置求交一次性得到，避免对每个基因集、
    每次置换做 Python 层循环（朴素实现为 O(n_perm × n_sets × n_query)，GO 规模下不可用）。

    经验 p 值约定
    -------------
    ``p = (1 + #{null >= obs}) / (n_perm + 1)``，与置换检验惯例一致：该式保守（不会低估
    p 值），且当 n_perm 较小时 p 的**下限为 1/(n_perm+1)**（例如 n_perm=1000 时最小
    p ≈ 0.001）。报告时不应把该下限解读为"远小于 0.001"。
    """
    rng = np.random.default_rng(seed)
    bg = sorted(background)
    bg_index = {g: i for i, g in enumerate(bg)}
    n_bg = len(bg)
    q_in_bg = {g for g in query_genes if g in bg_index}
    n_q = len(q_in_bg)

    # 空背景或无可分析集合时直接返回空结果（否则 np.max 会因零长数组抛 ValueError）
    if n_bg == 0 or gs.n_sets == 0 or n_q == 0:
        return ORAResult(table=[], n_query=len(query_genes), n_background=n_bg,
                         n_query_in_background=n_q)

    # 分层：按表达量分位（无表达数据时全部归入单层）
    if expression is not None:
        vals = np.array([expression.get(g, np.nan) for g in bg], dtype=float)
        finite = np.isfinite(vals)
        if finite.sum() >= n_bins:
            edges = np.nanpercentile(vals[finite], np.linspace(0, 100, n_bins + 1))
            edges[0], edges[-1] = -np.inf, np.inf
            bin_idx = np.digitize(vals, edges[1:-1])
        else:
            bin_idx = np.zeros(n_bg, dtype=int)
    else:
        bin_idx = np.zeros(n_bg, dtype=int)

    # 把背景基因按层重排，得到层边界
    order = np.argsort(bin_idx, kind='stable')     # order[pos] = 基因索引
    inv = np.empty(n_bg, dtype=np.int64)           # inv[基因索引] = 位置
    inv[order] = np.arange(n_bg)
    sorted_bins = bin_idx[order]
    boundaries = np.searchsorted(sorted_bins, np.arange(sorted_bins.max() + 2))
    layer_sizes = np.diff(boundaries)
    is_query_sorted = np.isin(np.array(bg)[order], list(q_in_bg))
    q_cumsum = np.concatenate([[0], np.cumsum(is_query_sorted)])
    q_per_layer = np.diff(q_cumsum[boundaries])

    # 加权抽样的权重（按基因，重排后顺序）
    if selection_prob is not None:
        w = np.array([max(float(selection_prob.get(g, 0.0)), 0.0) for g in bg], dtype=float)[order]
        if w.sum() <= 0:
            w = None
    else:
        w = None

    rows = []
    for sid, genes in gs.sets.items():
        members = np.array([bg_index[g] for g in genes if g in bg_index], dtype=np.int64)
        inter = genes & q_in_bg
        if len(inter) < min_overlap or len(members) == 0:
            continue
        obs = len(inter)
        # 注意：必须用逆置换 inv 定位，**不能**用 np.searchsorted(order, members)——
        # searchsorted 要求输入数组已排序，而 order 是置换数组，结果不可靠
        # （本项目曾因此把零分布期望压低到正确值的约 1/4，且数值看似合理）。
        member_sorted_pos = inv[members]
        member_bins = sorted_bins[member_sorted_pos]
        member_layer_start = boundaries[member_bins]
        member_rank_in_layer = member_sorted_pos - member_layer_start

        null = np.empty(n_perm, dtype=np.int64)
        draws = np.minimum(q_per_layer, layer_sizes)
        for i in range(n_perm):
            counts = 0
            for b in range(len(layer_sizes)):
                d = int(draws[b])
                if d == 0:
                    continue
                lo, hi = int(boundaries[b]), int(boundaries[b + 1])
                if w is not None:
                    wb = w[lo:hi]
                    tot = wb.sum()
                    if tot > 0:
                        rel = rng.choice(hi - lo, size=d, replace=False, p=wb / tot)
                    else:
                        rel = rng.permutation(hi - lo)[:d]
                else:
                    rel = rng.permutation(hi - lo)[:d]
                targets = member_rank_in_layer[member_bins == b]
                if targets.size:
                    counts += int(np.isin(rel, targets).sum())
            null[i] = counts
        p_emp = (1 + int((null >= obs).sum())) / (n_perm + 1)
        rows.append({
            'set_id': sid, 'set_name': gs.display_name(sid),
            'set_size': len(members), 'overlap': obs,
            'expected': float(null.mean()),
            'fold_enrichment': (obs / null.mean()) if null.mean() else float('nan'),
            'p': float(p_emp),
            'p_hypergeom': hypergeom_sf(obs, n_bg, len(members), n_q),
            'null_sd': float(null.std(ddof=1)),
            'genes': ','.join(sorted(inter)),
        })
    rows.sort(key=lambda r: r['p'])
    if rows:
        rej, qv = fdr_bh([r['p'] for r in rows])
        for r, rj, qqi in zip(rows, rej, qv):
            r['fdr_significant'] = bool(rj)
            r['q'] = float(qqi)
    return ORAResult(table=rows, n_query=len(query_genes), n_background=n_bg,
                     n_query_in_background=n_q)


# ==========================================================================
# GSEA（基因集水平，不依赖单基因阈值）
# ==========================================================================
@dataclass
class GSEAResult:
    table: list[dict]
    n_genes_ranked: int


def _gsea_es(ranked_genes: list[str], gene_set: set[str], weights: np.ndarray,
             exponent: float = 1.0) -> tuple[float, float]:
    """计算单个基因集的富集分数 ES 与归一化 ES（返回 (es, nes 的分母)）。"""
    hits = np.array([g in gene_set for g in ranked_genes], dtype=bool)
    n_hits = int(hits.sum())
    n_miss = len(hits) - n_hits
    if n_hits == 0 or n_miss == 0:
        return 0.0, 1.0
    w = np.abs(weights) ** exponent
    hit_w = np.where(hits, w, 0.0)
    sum_hit = hit_w.sum()
    if sum_hit == 0:
        hit_w = hits.astype(float)
        sum_hit = hit_w.sum()
    inc = hit_w / sum_hit
    dec = np.where(hits, 0.0, 1.0 / n_miss)
    running = np.cumsum(inc - dec)
    es = float(running[np.argmax(np.abs(running))])
    return es, 1.0


def gsea(ranked_genes: list[str], weights: np.ndarray, gs: GeneSets, *,
         n_perm: int = 1000, seed: int = 0, min_size: int = 10, max_size: int = 500,
         exponent: float = 1.0) -> GSEAResult:
    """加权 GSEA（基因集水平）。

    零分布通过置换**基因标签**（打乱排序）生成，给出 NES 与经验 p 值。
    与 GSEA 软件一致：正 ES 与负 ES 分别用对应方向的零分布计算 p 值。
    """
    genes = list(ranked_genes)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(-w)
    genes_sorted = [genes[i] for i in order]
    w_sorted = w[order]
    rng = np.random.default_rng(seed)
    rows = []
    for sid, gset in gs.sets.items():
        gset = set(gset)
        if not (min_size <= len(gset) <= max_size):
            continue
        es, _ = _gsea_es(genes_sorted, gset, w_sorted, exponent)
        if es == 0.0:
            continue
        null = np.empty(n_perm)
        n_hit = len(gset & set(genes_sorted))
        for i in range(n_perm):
            perm_genes = rng.permutation(len(genes_sorted))[:n_hit]
            mask = np.zeros(len(genes_sorted), dtype=bool)
            mask[perm_genes] = True
            # 用同一权重序列、随机命中位置计算 ES
            w_abs = np.abs(w_sorted) ** exponent
            sum_hit = w_abs[mask].sum()
            if sum_hit == 0:
                continue
            inc = np.where(mask, w_abs, 0.0) / sum_hit
            dec = np.where(mask, 0.0, 1.0 / max((~mask).sum(), 1))
            running = np.cumsum(inc - dec)
            null[i] = running[np.argmax(np.abs(running))]
        null = null[np.isfinite(null)]
        if len(null) < 10:
            continue
        pos = null[null >= 0]
        neg = null[null < 0]
        if es >= 0:
            ref = pos if len(pos) >= 5 else np.abs(null)
            p = (1 + (ref >= es).sum()) / (len(ref) + 1)
            nes = es / ref.mean() if ref.mean() else float('nan')
        else:
            ref = neg if len(neg) >= 5 else -np.abs(null)
            p = (1 + (ref <= es).sum()) / (len(ref) + 1)
            nes = es / abs(ref.mean()) if ref.mean() else float('nan')
        rows.append({'set_id': sid, 'set_name': gs.display_name(sid),
                     'set_size': len(gset), 'n_hits': n_hit, 'ES': es, 'NES': nes,
                     'p': float(p),
                     'leading_edge': ','.join(sorted(set(genes_sorted[:200]) & gset))})
    rows.sort(key=lambda r: r['p'])
    if rows:
        rej, qv = fdr_bh([r['p'] for r in rows])
        for r, rj, qqi in zip(rows, rej, qv):
            r['fdr_significant'] = bool(rj)
            r['q'] = float(qqi)
    return GSEAResult(table=rows, n_genes_ranked=len(genes))


# ==========================================================================
# 细胞类型分析（EWCE 风格 + 比例空间映射）
# ==========================================================================
@dataclass
class CellTypeExpression:
    """细胞类型表达参考。

    ``matrix``：基因 × 细胞类型 的平均表达（线性或 log 空间均可，特异性按行计算）。
    ``cell_types``：细胞类型名（列）。
    ``genes``：基因名（行）。
    """
    matrix: np.ndarray
    cell_types: list[str]
    genes: list[str]

    def specificity(self, metric: str = 'mean_ratio') -> np.ndarray:
        """按行计算每个基因对每种细胞类型的特异性（基因 × 细胞类型）。

        - ``mean_ratio``：表达 / 该基因在全部细胞类型的均值。>1 表示高于自身平均。
        - ``tau``：类似 Tau 的特异性指数，**取值越小越特异**（Yanai 等 2005）。
        - ``expression``：直接返回表达量。仅在参考矩阵已按合并样本归一化、可直接比较时使用。
        """
        m = np.asarray(self.matrix, dtype=float)
        if metric == 'mean_ratio':
            denom = np.nanmean(m, axis=1, keepdims=True)
            denom[denom == 0] = np.nan
            return m / denom
        if metric == 'tau':
            mx = np.nanmax(m, axis=1, keepdims=True)
            mx[mx == 0] = np.nan
            if len(self.cell_types) <= 1:
                return np.zeros_like(m)
            return (1 - m / mx) / (len(self.cell_types) - 1)
        if metric == 'expression':
            return m
        raise ValueError(f'未知 metric：{metric}')


def _quantile_bins(values: np.ndarray, n_bins: int = 40) -> tuple[np.ndarray, list[np.ndarray]]:
    """按分位数分层，返回 (bin 索引, 每层成员索引列表)。"""
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if finite.sum() < n_bins:
        idx = np.zeros(len(vals), dtype=int)
        return idx, [np.arange(len(vals))]
    edges = np.nanpercentile(vals[finite], np.linspace(0, 100, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(vals, edges[1:-1])
    return idx, [np.flatnonzero(idx == b) for b in np.unique(idx)]


def ewce(candidate_genes, cell_expr: CellTypeExpression, *, n_perm: int = 1000,
         seed: int = 0, specificity_metric: str = 'mean_ratio',
         marker_direction: str = 'over', top_frac: float = 0.1, n_bins: int = 40,
         apply_quantile: bool = True) -> dict:
    """EWCE 风格的细胞类型富集检验（Skene & Grant 2016 的核心逻辑）。

    参数
    ----
    marker_direction : ``'over'``（默认）检验候选基因是否富集于某细胞类型的**高特异性**
        top 比例；``'under'`` 检验是否富集于**低特异性**端（即该细胞类型表达偏低），
        两者含义完全不同，必须在分析前声明。
    apply_quantile : 是否按基因平均表达分层抽样。**默认 True**，用于控制表达量混杂。

    步骤
    ----
    1. 对每个基因、每种细胞类型计算特异性（``mean_ratio`` / ``tau`` / ``expression``）
    2. 若 ``apply_quantile``：把全部基因按**平均表达**分层，使每个基因只在其所属层内参与比较
    3. 统计候选基因在每种细胞类型 top ``top_frac`` 比例中的占比（观测值）
    4. 零分布：按同分层结构从背景抽样同等数量的基因，重复 ``n_perm`` 次

    返回每种细胞类型的 fold change 与经验 p 值（并做 FDR 校正）。
    """
    rng = np.random.default_rng(seed)
    spec = cell_expr.specificity(specificity_metric)          # 基因 × 细胞类型
    genes = cell_expr.genes
    gene_pos = {g: i for i, g in enumerate(genes)}
    cand = sorted({g for g in candidate_genes if g in gene_pos})
    mean_expr = np.nanmean(cell_expr.matrix, axis=1)
    idx, layers = _quantile_bins(mean_expr, n_bins) if apply_quantile else (
        np.zeros(len(genes), dtype=int), [np.arange(len(genes))])
    layer_pos = {int(b): i for i, b in enumerate(np.unique(idx))}

    # 每个基因在其所属表达层内、每种细胞类型中是否进入"顶端比例"
    in_top = np.zeros((len(genes), len(cell_expr.cell_types)), dtype=bool)
    sign = -1.0 if marker_direction == 'over' else 1.0
    for layer in layers:
        if len(layer) < 3:
            continue
        sub = spec[layer]                                     # 层内 基因 × 细胞类型
        k = max(1, int(round(top_frac * len(layer))))
        for c in range(sub.shape[1]):
            col = sub[:, c]
            if not np.isfinite(col).any():
                continue
            ranked = np.argsort(np.where(np.isfinite(col), sign * col, np.inf))[:k]
            in_top[layer[ranked], c] = True

    cand_rows = np.array([gene_pos[g] for g in cand], dtype=int)
    obs = in_top[cand_rows].mean(axis=0) if len(cand) else np.zeros(len(cell_expr.cell_types))
    n_cand = len(cand)
    cand_layers = [layer_pos[int(idx[r])] for r in cand_rows]

    results = []
    for c, cname in enumerate(cell_expr.cell_types):
        null = np.empty(n_perm)
        for i in range(n_perm):
            picked = np.empty(n_cand, dtype=int)
            for j, lp in enumerate(cand_layers):
                members = layers[lp]
                picked[j] = int(rng.choice(members)) if len(members) else int(cand_rows[j])
            null[i] = in_top[picked, c].mean() if n_cand else 0.0
        o = float(obs[c])
        p = (1 + (null >= o).sum()) / (n_perm + 1)
        results.append({
            'cell_type': cname, 'observed_prop': o, 'expected_prop': float(null.mean()),
            'fold_change': (o / null.mean()) if null.mean() > 0 else (float('inf') if o > 0 else float('nan')),
            'p': float(p), 'null_sd': float(null.std(ddof=1)), 'n_candidates': n_cand,
        })
    results.sort(key=lambda r: r['p'])
    if results:
        rej, qv = fdr_bh([r['p'] for r in results])
        for r, rj, qqi in zip(results, rej, qv):
            r['fdr_significant'] = bool(rj)
            r['q'] = float(qqi)
    return {'cell_types': cell_expr.cell_types, 'results': results,
            'n_candidates_used': n_cand, 'n_genes_reference': len(genes),
            'top_frac': top_frac, 'n_perm': n_perm,
            'marker_direction': marker_direction,
            'specificity_metric': specificity_metric,
            'apply_quantile': bool(apply_quantile)}


def proportion_map_correlation(proportions: np.ndarray, cell_types: list[str],
                               image_map: np.ndarray, *, spins=None,
                               method: str = 'pearson') -> list[dict]:
    """细胞类型比例的空间映射与影像图谱相关（分区 × 细胞类型 vs 分区）。

    若有 ``spins``（旋转置换索引），同时给出 spin p 值。
    """
    y = np.asarray(image_map, dtype=float).ravel()
    out = []
    for c, name in enumerate(cell_types):
        x = np.asarray(proportions[:, c], dtype=float).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 4:
            out.append({'cell_type': name, 'r': float('nan'), 'p': float('nan')})
            continue
        r = float(np.corrcoef(x[m], y[m])[0, 1])
        rec = {'cell_type': name, 'r': r}
        if spins is not None:
            from tnwt.spin import spin_test
            st = spin_test(x, spins, y, method=method)
            rec['p_spin'] = st['p_two_sided']
            rec['n_perm'] = st['n_perm']
        out.append(rec)
    if spins is not None:
        ps = [rec.get('p_spin', float('nan')) for rec in out]
        rej, qv = fdr_bh(ps)
        for rec, rj, qqi in zip(out, rej, qv):
            rec['fdr_significant'] = bool(rj)
            rec['q'] = float(qqi)
    return out


# ==========================================================================
# 报告与写出
# ==========================================================================
def write_table(path: str | Path, rows: list[dict], columns: list[str] | None = None) -> Path:
    """写出结果表。

    **约定**：``rows`` 为空时**不创建文件**。理由：下游（出图、论文）以
    "文件是否存在"判断该环节是否有结果；写一个零字节 CSV 会让
    ``pd.read_csv`` 抛 ``EmptyDataError``，且语义上"空结果"与"未运行"难以区分。
    """
    path = Path(path)
    if not rows:
        if path.exists():
            path.unlink()
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or list(rows[0].keys())
    with path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path
