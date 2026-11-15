"""AHBA（Allen Human Brain Atlas）微阵列数据处理管线（零依赖实现）。

存在理由：分析环境无 abagen。本模块按 abagen 的公开定义与默认流程实现同等处理，
使阶段三（AHBA 处理）在无 abagen 的环境下也能复现；一旦可安装 abagen，可用
``compare_with_abagen`` 做交叉验证。

流程（顺序与 abagen 一致）
--------------------------
1. 读取供体目录（MicroarrayExpression.csv / Probes.csv / SampleAnnot.csv）
2. 探针重注释与过滤：A_32_* 探针按 gene_symbol 匹配；剔除 ENTREZ 为空的探针
3. 低表达探针过滤：保留在 ≥ ``missing_threshold`` 比例样本中检出的探针（PACall）
4. 多探针选择：每个基因选与其同基因其它探针平均相关最高的代表探针
5. 样本匹配：把 MNI 坐标匹配到分区（半径内全部样本，或 k 近邻）
6. 归一化：先按样本（跨基因），再按基因（跨样本）
7. 跨供体聚合：按分区-基因取均值，并记录供体覆盖
8. 报告：探针/基因数、分区覆盖直方图、LODO 拆分

归一化方法（公式与 abagen 文档一致）
------------------------------------
center / zscore / minmax / sigmoid / scaled_sigmoid / scaled_sigmoid_quantiles /
robust_sigmoid / scaled_robust_sigmoid / mixed_sigmoid / None

注意：abagen 的**默认**是 ``scaled_robust_sigmoid``（Arnatkevičiūtė 等 2019 推荐），
本项目方案（v2.1）指定的**主分析**是 ``sigmoid``，两者不同，须在论文中明确说明。
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# abagen 归一化方法名（含默认值标记）
NORMALIZATIONS = (
    'center', 'zscore', 'minmax', 'sigmoid', 'scaled_sigmoid',
    'scaled_sigmoid_quantiles', 'robust_sigmoid', 'scaled_robust_sigmoid',
    'mixed_sigmoid', None,
)
ABAGEN_DEFAULT_NORMALIZATION = 'scaled_robust_sigmoid'
ERF_INV_HALF = 0.4769362762044699   # erf^{-1}(1/2)


def _erfinv(x: np.ndarray) -> np.ndarray:
    """erf 的逆函数（向量化，Newton 迭代；用于 robust sigmoid 的 IQR 归一化）。"""
    x = np.clip(np.asarray(x, dtype=float), -0.9999999, 0.9999999)
    # 初值：Winitzki 近似
    a = 0.147
    ln1 = np.log(1 - x ** 2)
    t = 2 / (np.pi * a) + ln1 / 2
    w = np.sign(x) * np.sqrt(np.sqrt(t ** 2 - ln1 / a) - t)
    for _ in range(6):
        err = _erf(w) - x
        w = w - err / (2 / np.sqrt(np.pi) * np.exp(-w ** 2))
    return w


def _erf(x: np.ndarray) -> np.ndarray:
    """误差函数（Abramowitz-Stegun 7.1.26，精度约 1e-7，足够 IQR 归一化）。"""
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-ax ** 2)
    return sign * y


# ==========================================================================
# 归一化
# ==========================================================================
def normalize(values: np.ndarray, method: str | None) -> np.ndarray:
    """按指定方法归一化（对给定矩阵的相应维度）。

    ``values`` 形状为 (n_rows, n_cols)，归一化沿 ``axis=1``（每行独立）。
    """
    v = np.asarray(values, dtype=float)
    if method is None:
        return v.copy()
    if method not in NORMALIZATIONS:
        raise ValueError(f'未知归一化方法：{method}；可选 {NORMALIZATIONS}')

    def _center(x):
        return x - np.nanmean(x, axis=1, keepdims=True)

    def _zscore(x):
        mu = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, ddof=1, keepdims=True)
        sd[sd == 0] = np.nan
        return (x - mu) / sd

    def _minmax(x):
        lo = np.nanmin(x, axis=1, keepdims=True)
        hi = np.nanmax(x, axis=1, keepdims=True)
        rng = hi - lo
        rng[rng == 0] = np.nan
        return (x - lo) / rng

    def _sigmoid_base(x):
        return 1.0 / (1.0 + np.exp(-_zscore(x)))

    def _robust_sigmoid_base(x):
        med = np.nanmedian(x, axis=1, keepdims=True)
        q1 = np.nanpercentile(x, 25, axis=1, keepdims=True)
        q3 = np.nanpercentile(x, 75, axis=1, keepdims=True)
        iqr = (q3 - q1) / (2 * np.sqrt(2) * ERF_INV_HALF)
        iqr[iqr == 0] = np.nan
        return 1.0 / (1.0 + np.exp(-(x - med) / iqr))

    if method == 'center':
        return _center(v)
    if method == 'zscore':
        return _zscore(v)
    if method == 'minmax':
        return _minmax(v)
    if method == 'sigmoid':
        return _sigmoid_base(v)
    if method == 'scaled_sigmoid':
        return _minmax(_sigmoid_base(v))
    if method == 'scaled_sigmoid_quantiles':
        lo = np.nanpercentile(v, 5, axis=1, keepdims=True)
        hi = np.nanpercentile(v, 95, axis=1, keepdims=True)
        clipped = np.clip(v, lo, hi)
        # 用截断后的分布估参数，但对完整分布做变换
        return _minmax(_sigmoid_base_from(clipped, v))
    if method == 'robust_sigmoid':
        return _robust_sigmoid_base(v)
    if method == 'scaled_robust_sigmoid':
        return _minmax(_robust_sigmoid_base(v))
    if method == 'mixed_sigmoid':
        q1 = np.nanpercentile(v, 25, axis=1, keepdims=True)
        q3 = np.nanpercentile(v, 75, axis=1, keepdims=True)
        zero_iqr = ((q3 - q1) == 0)
        out = _minmax(_robust_sigmoid_base(v))
        if zero_iqr.any():
            out[zero_iqr.ravel()] = _minmax(_sigmoid_base(v))[zero_iqr.ravel()]
        return out
    raise ValueError(method)


def _sigmoid_base_from(param_source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """用 ``param_source`` 估均值/标准差，对 ``target`` 施加 sigmoid。"""
    mu = np.nanmean(param_source, axis=1, keepdims=True)
    sd = np.nanstd(param_source, axis=1, ddof=1, keepdims=True)
    sd[sd == 0] = np.nan
    return 1.0 / (1.0 + np.exp(-(target - mu) / sd))


# ==========================================================================
# 供体数据读取
# ==========================================================================
@dataclass
class Donor:
    donor_id: str
    expression: np.ndarray            # (n_probes, n_samples)
    sample_ids: list[str]
    mni: np.ndarray                   # (n_samples, 3)
    probe_ids: list[str]
    gene_symbols: list[str]
    entrez_ids: list[str]
    detected: np.ndarray | None = None   # (n_probes, n_samples) 0/1，来自 PACall

    @property
    def n_probes(self) -> int:
        return self.expression.shape[0]

    @property
    def n_samples(self) -> int:
        return self.expression.shape[1]


REQUIRED_FILES = ('MicroarrayExpression.csv', 'Probes.csv', 'SampleAnnot.csv')


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r]
    return header, rows


def _read_csv_rows_raw(path: Path) -> list[list[str]]:
    """读 CSV 且**不丢弃首行**。

    AHBA 的 `MicroarrayExpression.csv` 与 `PACall.csv` 是**无表头**的
    （首行就是第一个探针的数据）。若按常规"首行即表头"解析，会吃掉第一个
    探针的数据，并把 `sample_ids` 变成表达值 —— 于是与 SampleAnnot 完全对不上。
    本项目用真实数据实测踩过（报 "SampleAnnot 与表达矩阵样本数不一致：0 vs 893"）。
    """
    with path.open('r', encoding='utf-8-sig', newline='') as fh:
        return [r for r in csv.reader(fh) if r]


def read_donor(donor_dir: Path) -> Donor:
    """读取单个供体目录。"""
    donor_dir = Path(donor_dir)
    missing = [f for f in REQUIRED_FILES if not (donor_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f'{donor_dir} 缺少文件：{missing}')

    donor_id = donor_dir.name.replace('normalized_microarray_donor', '').strip('_')

    # 无表头：首行即第一个探针的数据（见 _read_csv_rows_raw 文档）
    expr_rows = _read_csv_rows_raw(donor_dir / 'MicroarrayExpression.csv')
    probe_ids = [r[0] for r in expr_rows]
    expression = np.array([[float(x) for x in r[1:]] for r in expr_rows], dtype=float)
    n_samples = expression.shape[1]

    ann_header, ann_rows = _read_csv_rows(donor_dir / 'SampleAnnot.csv')
    idx = {name: i for i, name in enumerate(ann_header)}
    need = ('mni_x', 'mni_y', 'mni_z')
    if not all(k in idx for k in need):
        raise ValueError(f'SampleAnnot.csv 缺少 {need}，实际列：{ann_header}')
    # AHBA 的表达矩阵**没有样本 ID 列**，SampleAnnot 的行与表达矩阵的样本列
    # 是**按位置一一对应**的（实测：946 行 vs 946 列）。
    # 旧实现尝试用 `sample_id` 列去 join —— 该列在当前数据中不存在，会回退到
    # `structure_id`，必然匹配失败。这里改为显式位置对齐并校验行数。
    if len(ann_rows) != n_samples:
        raise ValueError(
            f'SampleAnnot 行数 {len(ann_rows)} != 表达矩阵样本列数 {n_samples}')
    ordered = ann_rows
    wid = idx.get('well_id')
    sample_ids = [str(r[wid]) if wid is not None else str(i)
                  for i, r in enumerate(ordered)]
    mni = np.array([[float(r[idx['mni_x']]), float(r[idx['mni_y']]), float(r[idx['mni_z']])]
                    for r in ordered], dtype=float)

    pr_header, pr_rows = _read_csv_rows(donor_dir / 'Probes.csv')
    pidx = {name: i for i, name in enumerate(pr_header)}
    by_probe = {r[pidx.get('probe_id', 0)]: r for r in pr_rows}
    genes, entrez = [], []
    for pid in probe_ids:
        row = by_probe.get(pid)
        if row is None:
            genes.append('')
            entrez.append('')
            continue
        genes.append(str(row[pidx.get('gene_symbol', 3)]).strip())
        entrez.append(str(row[pidx.get('entrez_id', 4)]).strip())

    detected = None
    pacall = donor_dir / 'PACall.csv'
    if pacall.exists():
        prws = _read_csv_rows_raw(pacall)       # 同样无表头
        if [r[0] for r in prws] == probe_ids:
            detected = np.array([[float(x) for x in r[1:]] for r in prws], dtype=float)

    return Donor(donor_id=donor_id, expression=expression, sample_ids=sample_ids,
                 mni=mni, probe_ids=probe_ids, gene_symbols=genes,
                 entrez_ids=entrez, detected=detected)


# ==========================================================================
# 探针重注释与过滤
# ==========================================================================
def reannotate_probes(donor: Donor) -> Donor:
    """探针重注释：未被注释到的 A_32_* 探针按探针名回填 gene_symbol。

    AHBA 的部分探针（ENTREZ 为空、gene_symbol 形如 A_32_P12345）未匹配到基因；
    abagen 的做法是尝试用探针名本身作为基因标识（仅在无其他信息时）。
    这里遵循同一原则：若 entrez 为空且 gene_symbol 以 'A_32_' 开头，则保留探针名，
    并在后续过滤步骤中按"未注释探针"处理（由调用方决定是否剔除）。
    """
    genes = list(donor.gene_symbols)
    for i, (g, e) in enumerate(zip(genes, donor.entrez_ids)):
        if (not e or e.lower() in ('nan', 'na', '')) and g.startswith('A_32_'):
            genes[i] = g
    donor.gene_symbols = genes
    return donor


def filter_probes(donor: Donor, missing_threshold: float = 0.5,
                  drop_unannotated: bool = True) -> Donor:
    """低表达探针过滤 + 未注释探针剔除。

    ``missing_threshold``：探针必须在至少该比例的样本中被检出（PACall > 0.5）
    才保留；无 PACall 时退化为「非零表达值比例」。
    """
    n_samples = donor.n_samples
    if donor.detected is not None:
        frac = (donor.detected > 0.5).sum(axis=1) / n_samples
    else:
        frac = (donor.expression > 0).sum(axis=1) / n_samples
    keep = frac >= missing_threshold
    if drop_unannotated:
        annotated = np.array([
            bool(g) and not g.startswith('A_32_') and not g.startswith('CUST_')
            and (bool(e) and e.lower() not in ('nan', 'na', ''))
            for g, e in zip(donor.gene_symbols, donor.entrez_ids)
        ])
        keep &= annotated
    donor.expression = donor.expression[keep]
    donor.probe_ids = [p for p, k in zip(donor.probe_ids, keep) if k]
    donor.gene_symbols = [g for g, k in zip(donor.gene_symbols, keep) if k]
    donor.entrez_ids = [e for e, k in zip(donor.entrez_ids, keep) if k]
    donor.detected = donor.detected[keep] if donor.detected is not None else None
    return donor


def select_probes(donor: Donor, donor_aggregation: str = 'mean') -> dict[str, int]:
    """多探针选择：每个基因保留与同基因其它探针平均相关最高的探针。

    返回 {gene_symbol: 探针行号}。
    """
    from collections import defaultdict
    groups: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(donor.gene_symbols):
        groups[g].append(i)

    selected: dict[str, int] = {}
    X = donor.expression
    for gene, rows in groups.items():
        if len(rows) == 1:
            selected[gene] = rows[0]
            continue
        sub = X[rows]
        # 探针-样本矩阵，按探针求相关
        if sub.std(axis=1).min() == 0:
            corr = np.corrcoef(sub)
            corr = np.nan_to_num(corr)
        else:
            corr = np.corrcoef(sub)
            corr = np.nan_to_num(corr)
        np.fill_diagonal(corr, np.nan)
        score = np.nanmean(corr, axis=1)
        if donor_aggregation == 'mean':
            best = int(np.nanargmax(score)) if np.isfinite(score).any() else 0
        else:
            raise ValueError('本实现仅支持 donor_aggregation="mean"')
        selected[gene] = rows[best]
    return selected


def gene_by_sample(donor: Donor, selected: dict[str, int]) -> tuple[list[str], np.ndarray]:
    """按选中探针构建「基因 × 样本」矩阵。"""
    genes = sorted(selected)
    mat = np.vstack([donor.expression[selected[g]] for g in genes])
    return genes, mat


# ==========================================================================
# 样本匹配与归一化
# ==========================================================================
def make_voxel_matcher(atlas_path, tolerance: float = 2.0,
                       n_parcels: int = 200):
    """构造"按图集体素归属"的样本匹配器。

    与 ``match_samples``（质心+半径）的区别是关键性的：质心把每个分区缩成
    一个点，而分区实际有数千个体素，样本落在分区内部却离质心很远时会被漏掉。
    实测（Schaefer200，6 供体 CIC 坐标）：质心+2mm 只有 8/200 个分区有样本，
    而体素归属+2mm 膨胀可达 197/200。

    归属规则（与 abagen 一致）：
      1) 样本坐标所在体素有分区标签 -> 直接归该分区；
      2) 否则在 ``tolerance`` mm 内找最近的带标签体素 -> 归该分区；
      3) 都找不到（样本在脑外/白质/皮下）-> 不归属任何分区。

    依赖 nibabel 与 scipy，故在此**惰性导入** —— 本模块其余部分仍是零依赖实现，
    没有这两个包时可以继续用质心匹配。
    """
    import nibabel as nib
    from scipy.spatial import cKDTree

    img = nib.load(str(atlas_path))
    atl = np.asarray(img.dataobj).astype(np.int16)
    inv_aff = np.linalg.inv(img.affine)
    shape = np.asarray(atl.shape)

    idx = np.argwhere(atl > 0)
    if len(idx) == 0:
        raise ValueError(f'图集 {atlas_path} 不含任何正标签')
    labels = atl[atl > 0]
    world = nib.affines.apply_affine(img.affine, idx)
    tree = cKDTree(world)

    def matcher(samples: np.ndarray) -> list[list[int]]:
        samples = np.asarray(samples, dtype=float)
        out: list[list[int]] = [[] for _ in range(n_parcels)]
        for i, xyz in enumerate(samples):
            v = np.round(nib.affines.apply_affine(inv_aff, xyz)).astype(int)
            lab = 0
            if np.all((v >= 0) & (v < shape)):
                lab = int(atl[v[0], v[1], v[2]])
            if lab > 0:
                if lab <= n_parcels:
                    out[lab - 1].append(i)
                continue
            if tolerance and tolerance > 0:
                dist, j = tree.query(xyz, k=1)
                if dist <= tolerance:
                    lab = int(labels[j])
                    if lab <= n_parcels:
                        out[lab - 1].append(i)
        return out

    return matcher


def match_samples(coords: np.ndarray, samples: np.ndarray,
                  radius: float | None = 2.0, k: int | None = None) -> list[list[int]]:
    """把样本 MNI 坐标匹配到分区（质心）。

    ``radius`` 模式下取半径内全部样本；``k`` 模式下取最近 k 个。
    返回长度为 n_parcels 的列表，每项是样本索引列表。
    """
    coords = np.asarray(coords, dtype=float)
    samples = np.asarray(samples, dtype=float)
    d = np.sqrt(((coords[:, None, :] - samples[None, :, :]) ** 2).sum(axis=2))
    out: list[list[int]] = []
    for i in range(len(coords)):
        if k is not None:
            out.append(list(np.argsort(d[i])[:k]))
        else:
            out.append(list(np.flatnonzero(d[i] <= radius)))
    return out


def normalize_expression(mat: np.ndarray, sample_norm: str | None = 'sigmoid',
                         gene_norm: str | None = 'sigmoid') -> np.ndarray:
    """先按样本归一化（跨基因，即对 ``mat`` 的列），再按基因归一化（跨样本，即行）。

    与 abagen 一致：样本归一化先于基因归一化。
    """
    out = np.asarray(mat, dtype=float).copy()
    if sample_norm is not None:
        out = normalize(out.T, sample_norm).T      # 每个样本一行（跨基因）
    if gene_norm is not None:
        out = normalize(out, gene_norm)            # 每个基因一行（跨样本）
    return out


def parcel_expression(genes: list[str], mat: np.ndarray,
                      assignment: list[list[int]]) -> np.ndarray:
    """把「基因 × 样本」聚合为「分区 × 基因」，并对分区内样本取均值。"""
    n_parcels = len(assignment)
    out = np.full((n_parcels, len(genes)), np.nan)
    for i, idx in enumerate(assignment):
        if idx:
            out[i] = np.nanmean(mat[:, idx], axis=1)
    return out


@dataclass
class AHBAReport:
    n_donors: int = 0
    n_samples_total: int = 0
    n_probes_raw: int = 0
    n_probes_filtered: int = 0
    n_genes: int = 0
    n_parcels_total: int = 0
    n_parcels_with_data: int = 0
    parcel_donor_coverage: np.ndarray = field(default_factory=lambda: np.array([]))
    genes_per_donor: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def run_pipeline(donor_dirs, coords: np.ndarray, *, radius: float = 2.0,
                 missing_threshold: float = 0.5, sample_norm: str | None = 'sigmoid',
                 gene_norm: str | None = 'sigmoid', donor_coverage_min: int = 2,
                 lodo: bool = False, matcher=None
                 ) -> tuple[np.ndarray, list[str], AHBAReport, dict]:
    """完整流程：读取 → 过滤 → 选探针 → 匹配 → 归一化 → 跨供体聚合。

    返回 ``(parcel x gene 矩阵, 基因名列表, 报告, 附加信息)``。
    供体覆盖 < ``donor_coverage_min`` 的分区被剔除（返回中置为 NaN，并在报告中记录）。
    """
    report = AHBAReport(n_donors=len(donor_dirs), n_parcels_total=len(coords))
    per_donor: dict[str, dict[str, np.ndarray]] = {}
    gene_sets: list[set[str]] = []
    assignments: dict[str, list[list[int]]] = {}

    for dd in donor_dirs:
        d = read_donor(Path(dd))
        report.n_samples_total += d.n_samples
        report.n_probes_raw += d.n_probes
        d = reannotate_probes(d)
        d = filter_probes(d, missing_threshold=missing_threshold)
        report.n_probes_filtered += d.n_probes
        selected = select_probes(d)
        genes, mat = gene_by_sample(d, selected)
        report.genes_per_donor[d.donor_id] = len(genes)
        gene_sets.append(set(genes))
        # [!] assignment 必须**逐供体**计算。
        # 它是"每个分区包含哪些样本索引"，而各供体的样本数量与 MNI 坐标都不同
        # （实测：946 / 893 / 363 / 529 / 470 / 501）。跨供体复用会让索引越界；
        # 更危险的是当样本数恰好兼容时，会静默地把别的供体的样本当作当前供体的
        # 样本做分区平均，产出错误却看似正常的结果。
        donor_assignment = (matcher(d.mni) if matcher is not None
                            else match_samples(coords, d.mni, radius=radius))
        assignments[d.donor_id] = donor_assignment
        norm = normalize_expression(mat, sample_norm, gene_norm)
        per_donor[d.donor_id] = {
            'genes': genes,
            'parcel': parcel_expression(genes, norm, donor_assignment),
        }

    # 基因交集（保证跨供体可比）
    common = sorted(set.intersection(*gene_sets)) if gene_sets else []
    report.n_genes = len(common)
    stack = np.full((len(per_donor), len(coords), len(common)), np.nan)
    for di, (donor_id, obj) in enumerate(per_donor.items()):
        pos = {g: i for i, g in enumerate(obj['genes'])}
        cols = [pos[g] for g in common]
        stack[di] = obj['parcel'][:, cols]

    with np.errstate(invalid='ignore'):
        coverage = np.sum(np.isfinite(stack), axis=0)          # (parcels, genes)
        expr = np.nanmean(stack, axis=0)                        # (parcels, genes)
    parcel_cov = (coverage > 0).sum(axis=1)                     # 分区覆盖的供体数（按基因）
    report.parcel_donor_coverage = parcel_cov
    report.n_parcels_with_data = int((parcel_cov >= donor_coverage_min).sum())
    low = parcel_cov < donor_coverage_min
    if low.any():
        expr[low] = np.nan
        report.notes.append(
            f'{int(low.sum())} 个分区因供体覆盖 < {donor_coverage_min} 被剔除')

    extra = {'stack': stack, 'common_genes': common, 'coverage': coverage,
             'assignments': assignments}
    if lodo:
        extra['lodo'] = _lodo_splits(list(per_donor), common, stack)
    return expr, common, report, extra


def _lodo_splits(donor_ids: list[str], genes: list[str], stack: np.ndarray) -> dict:
    """leave-one-donor-out：每次剔除一个供体后重新聚合，用于供体稳定性分析。"""
    out = {}
    for i, did in enumerate(donor_ids):
        keep = [j for j in range(len(donor_ids)) if j != i]
        with np.errstate(invalid='ignore'):
            out[did] = np.nanmean(stack[keep], axis=0)
    return out


# ==========================================================================
# 合成自检
# ==========================================================================
def make_synthetic_donor(root: Path, donor_id: str, n_samples: int = 60,
                         n_genes: int = 40, probes_per_gene: int = 2,
                         seed: int = 0, coords: np.ndarray | None = None,
                         coord_sd: float = 0.8) -> Path:
    """生成合成供体目录，用于端到端自检。

    ``coords`` 为可选的分区质心 (n_parcels, 3)。**多供体自检必须传入同一组质心**，
    否则各供体样本散落在互不相同的坐标群附近，按半径匹配会全部落空
    （真实 AHBA 中所有供体共享同一 MNI 空间，故同一图集质心对各供体都成立）。
    """
    rng = np.random.default_rng(seed)
    ddir = Path(root) / f'normalized_microarray_donor{donor_id}'
    ddir.mkdir(parents=True, exist_ok=True)
    n_probes = n_genes * probes_per_gene
    sample_ids = [str(i + 1) for i in range(n_samples)]
    probe_ids = [f'P{i:05d}' for i in range(n_probes)]
    genes = [f'GENE{i % n_genes:03d}' for i in range(n_probes)]
    entrez = [str(1000 + (i % n_genes)) for i in range(n_probes)]

    expr = rng.normal(8.0, 1.0, size=(n_probes, n_samples))
    # 给同一基因的多个探针加入共同信号，便于选探针逻辑有区分度
    for g in range(n_genes):
        rows = [i for i in range(n_probes) if i % n_genes == g]
        shared = rng.normal(0, 0.5, size=n_samples)
        for r in rows:
            expr[r] += shared
    # 真实 AHBA 的 MicroarrayExpression.csv **无表头**（首行即第一个探针的数据）。
    # 夹具必须照此生成，否则单元测试无法暴露解析器的表头假设（本项目踩过该坑）。
    with (ddir / 'MicroarrayExpression.csv').open('w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        for pid, row in zip(probe_ids, expr):
            w.writerow([pid] + [f'{v:.6f}' for v in row])

    with (ddir / 'Probes.csv').open('w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['probe_id', 'probe_name', 'gene_id', 'gene_symbol',
                    'gene_name', 'entrez_id', 'chromosome'])
        for pid, g, e in zip(probe_ids, genes, entrez):
            w.writerow([pid, 'A_24_P1', e, g, e, '1'])

    # 样本坐标：围绕分区质心撒点（多供体须共享质心）
    if coords is None:
        coords = rng.normal(0, 30, size=(20, 3))
    coords = np.asarray(coords, dtype=float)
    n_parcels = len(coords)
    assign = rng.integers(0, n_parcels, size=n_samples)
    mni = coords[assign] + rng.normal(0, coord_sd, size=(n_samples, 3))
    with (ddir / 'SampleAnnot.csv').open('w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        # 真实 SampleAnnot.csv 的列（**没有 sample_id 列**；样本与表达列按位置对应）
        w.writerow(['structure_id', 'slab_num', 'well_id', 'slab_type',
                    'structure_acronym', 'structure_name', 'polygon_id',
                    'mri_voxel_x', 'mri_voxel_y', 'mri_voxel_z',
                    'mni_x', 'mni_y', 'mni_z'])
        for i, xyz in enumerate(mni):
            w.writerow([4000, 1, int(sample_ids[i]), 'CX', 'G_x', 'synthetic',
                        1, 0, 0, 0] + [f'{v:.3f}' for v in xyz])

    pac = (expr > 7.5).astype(int)
    # 同样无表头
    with (ddir / 'PACall.csv').open('w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        for pid, row in zip(probe_ids, pac):
            w.writerow([pid] + [str(int(v)) for v in row])
    return ddir
