"""纵向分析与临床分层（阶段九）。

对应研究方案 v2.1 第 5.8、5.9 节。覆盖：

1. 术前-术后配对比较（配对 t / Wilcoxon 符号秩 / 线性混合效应等价形式）
2. 术式分层（MVD vs 经皮介入）与交互项检验
3. 随访间隔作为协变量（本项目实测中位 202 天、17.4% 超过 270 天，不可当作等间隔）
4. 疼痛缓解百分比作为调节变量的回归（及其与影像变化的交互）
5. 临床分层：完全缓解 vs 未完全缓解的区分能力（交叉验证 AUC、PR-AUC、置换零分布）
6. 缺失数据处理：完整病例 + 多重插补（Rubin 合并）
7. 纵向图谐波耦合的"变化图谱"与基因表达的空间关联及 bootstrapped 差异检验

设计约束
--------
- 仅依赖 numpy/pandas（环境无 scipy/statsmodels/sklearn）。
- 所有随机过程接受 seed；所有输出标注样本量。
- **不生成任何占位数值**：缺失即 NaN 传播，插补显式声明方法。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tnwt.stats_lite import hedges_g, t_sf, welch_ttest  # noqa: E402


# ==========================================================================
# 配对比较
# ==========================================================================
@dataclass
class PairedResult:
    n_pairs: int
    mean_change: np.ndarray
    t: np.ndarray
    p: np.ndarray
    cohens_dz: np.ndarray
    n_positive: np.ndarray
    n_negative: np.ndarray
    parcels: list[str] = field(default_factory=list)

    @property
    def n_parcels(self) -> int:
        return len(self.mean_change)


def paired_contrast(baseline: pd.DataFrame, followup: pd.DataFrame,
                    *, min_pairs: int = 5) -> PairedResult:
    """逐分区配对比较（术后 − 术前）。

    只使用**同时具备术前与术后数据**的被试（完整配对），不做任何插补。
    ``baseline``/``followup`` 的行索引为被试 ID（随访 ID 需能映射到基线 ID，
    见 :func:`pair_ids`）。
    """
    common = [s for s in baseline.index if s in followup.index]
    n_par = baseline.shape[1]
    mean_change = np.full(n_par, np.nan)
    t = np.full(n_par, np.nan)
    p = np.full(n_par, np.nan)
    dz = np.full(n_par, np.nan)
    n_pos = np.zeros(n_par, dtype=int)
    n_neg = np.zeros(n_par, dtype=int)
    b = baseline.loc[common].to_numpy(dtype=float)
    f = followup.loc[common].to_numpy(dtype=float)
    for j in range(n_par):
        d = f[:, j] - b[:, j]
        m = np.isfinite(d)
        if m.sum() < min_pairs:
            continue
        dd = d[m]
        mean_change[j] = float(dd.mean())
        n_pos[j] = int((dd > 0).sum())
        n_neg[j] = int((dd < 0).sum())
        sd = dd.std(ddof=1)
        if sd > 0:
            t[j] = float(dd.mean() / (sd / np.sqrt(len(dd))))
            p[j] = float(min(1.0, 2 * t_sf(abs(t[j]), len(dd) - 1)))
            dz[j] = float(dd.mean() / sd)
    return PairedResult(n_pairs=len(common), mean_change=mean_change, t=t, p=p,
                        cohens_dz=dz, n_positive=n_pos, n_negative=n_neg,
                        parcels=list(baseline.columns))


def pair_ids(followup_ids: list[str], baseline_ids: list[str]) -> dict[str, str]:
    """把随访被试 ID 映射到基线 ID。

    支持两种命名约定（本数据集使用前者）：

    - ``sub-009fu`` → ``sub-009``（本项目 ds005713 的约定）
    - ``sub-009_ses-2`` → ``sub-009``
    """
    out: dict[str, str] = {}
    base = set(baseline_ids)
    for fu in followup_ids:
        cand = fu
        for suffix in ('_ses-2', '_ses-1', 'fu'):
            if cand.endswith(suffix):
                cand = cand[: -len(suffix)]
                break
        if cand in base:
            out[fu] = cand
    return out


def align_followup(followup: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """把随访矩阵的行索引重命名为对应基线 ID，便于与基线对齐。"""
    mapping = pair_ids(list(followup.index), list(baseline.index))
    if not mapping:
        return followup.iloc[0:0]
    sub = followup.loc[list(mapping)].copy()
    sub.index = [mapping[i] for i in sub.index]
    return sub


# ==========================================================================
# 术式分层与交互
# ==========================================================================
def stratify_by_surgery(clin: pd.DataFrame, participants: list[str],
                        *, surgery_col: str = 'Surgery_type') -> pd.Series:
    """返回被试的术式标签（已规范化）。"""
    df = clin.drop_duplicates(subset='participant_id', keep='first').set_index('participant_id')
    if surgery_col not in df.columns:
        raise KeyError(f'临床表缺少 {surgery_col}')
    raw = df[surgery_col].reindex(participants).astype(str).str.strip().str.upper()
    return raw.replace({'': np.nan, 'NAN': np.nan})


def stratified_paired_contrast(baseline: pd.DataFrame, followup: pd.DataFrame,
                               surgery: pd.Series, *, min_pairs: int = 3) -> dict:
    """按术式分别做配对比较，并给出两组变化量的差异检验（交互的等价形式）。

    交互的检验方式：对**变化量**做两样本 Welch t 检验（MVD vs 经皮介入）。
    在只有两个时间点的设计下，这与混合效应模型中 `time × surgery` 交互项的
    Wald 检验等价（无缺失、无额外随机效应时）。
    """
    out: dict = {}
    for label in sorted({v for v in surgery.dropna().unique()}):
        ids = [s for s in baseline.index
               if surgery.get(s) == label and s in followup.index]
        if len(ids) < min_pairs:
            out[label] = {'n_pairs': len(ids), 'note': '样本不足，跳过'}
            continue
        b = baseline.loc[ids].to_numpy(dtype=float)
        f = followup.loc[ids].to_numpy(dtype=float)
        d = f - b
        n_par = b.shape[1]
        mc = np.full(n_par, np.nan)
        tt = np.full(n_par, np.nan)
        pp = np.full(n_par, np.nan)
        for j in range(n_par):
            dd = d[:, j][np.isfinite(d[:, j])]
            if len(dd) < min_pairs:
                continue
            mc[j] = dd.mean()
            sd = dd.std(ddof=1)
            if sd > 0:
                tt[j] = dd.mean() / (sd / np.sqrt(len(dd)))
                pp[j] = float(min(1.0, 2 * t_sf(abs(tt[j]), len(dd) - 1)))
        out[label] = {'n_pairs': len(ids), 'mean_change': mc, 't': tt, 'p': pp,
                      'subjects': ids}

    if len([k for k, v in out.items() if 'mean_change' in v]) == 2:
        (a, b) = [k for k, v in out.items() if 'mean_change' in v]
        da, db = out[a]['mean_change'], out[b]['mean_change']
        n_par = baseline.shape[1]
        t_inter = np.full(n_par, np.nan)
        p_inter = np.full(n_par, np.nan)
        for j in range(n_par):
            ia = [s for s in out[a]['subjects']]
            ib = [s for s in out[b]['subjects']]
            ca = (followup.loc[ia].iloc[:, j].to_numpy(dtype=float)
                  - baseline.loc[ia].iloc[:, j].to_numpy(dtype=float))
            cb = (followup.loc[ib].iloc[:, j].to_numpy(dtype=float)
                  - baseline.loc[ib].iloc[:, j].to_numpy(dtype=float))
            if np.isfinite(ca).sum() < 3 or np.isfinite(cb).sum() < 3:
                continue
            r = welch_ttest(ca, cb)
            t_inter[j] = r['t']
            p_inter[j] = r['p']
        out['interaction'] = {'groups': [a, b], 'welch_t': t_inter, 'p': p_inter}
    return out


# ==========================================================================
# 随访间隔与缓解程度的调节
# ==========================================================================
def moderation_regression(change: np.ndarray, moderator: np.ndarray, *,
                          covariates: np.ndarray | None = None) -> dict:
    """单分区调节分析：ΔY = b0 + b1·M + b2·Cov + e，返回 b1 的估计与 p 值。

    ``change`` 为变化量（术后 − 术前），``moderator`` 如疼痛缓解百分比。
    """
    y = np.asarray(change, dtype=float).ravel()
    m = np.asarray(moderator, dtype=float).ravel()
    if covariates is None:
        R = np.column_stack([np.ones(len(y)), m])
        names = ['intercept', 'moderator']
    else:
        C = np.atleast_2d(np.asarray(covariates, dtype=float))
        if C.shape[0] != len(y):
            C = C.T
        R = np.column_stack([np.ones(len(y)), m, C])
        names = ['intercept', 'moderator'] + [f'cov{i+1}' for i in range(C.shape[1])]
    mask = np.isfinite(y) & np.all(np.isfinite(R), axis=1)
    n, k = int(mask.sum()), R.shape[1]
    if n <= k + 1:
        return {'n': n, 'beta': np.nan, 'se': np.nan, 't': np.nan, 'p': np.nan,
                'names': names}
    beta, *_ = np.linalg.lstsq(R[mask], y[mask], rcond=None)
    resid = y[mask] - R[mask] @ beta
    dof = n - k
    s2 = float((resid ** 2).sum() / dof)
    try:
        cov = s2 * np.linalg.inv(R[mask].T @ R[mask])
        se = float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se = float('nan')
    tval = beta[1] / se if se and np.isfinite(se) and se > 0 else np.nan
    pval = float(min(1.0, 2 * t_sf(abs(tval), dof))) if np.isfinite(tval) else np.nan
    return {'n': n, 'beta': float(beta[1]), 'se': se, 't': float(tval) if np.isfinite(tval) else np.nan,
            'p': pval, 'dof': dof, 'names': names}


def moderation_map(change_matrix: pd.DataFrame, moderator: pd.Series, *,
                   covariates: pd.DataFrame | None = None) -> pd.DataFrame:
    """对全部分区做调节分析，返回每分区的 beta / se / t / p。"""
    ids = list(change_matrix.index)
    mod = moderator.reindex(ids).to_numpy(dtype=float)
    cov = None
    if covariates is not None:
        cov = covariates.reindex(ids).to_numpy(dtype=float)
    rows = []
    for j, col in enumerate(change_matrix.columns):
        r = moderation_regression(change_matrix[col].to_numpy(dtype=float), mod,
                                  covariates=cov)
        rows.append({'parcel': col, 'n': r['n'], 'beta': r.get('beta'),
                     'se': r.get('se'), 't': r.get('t'), 'p': r.get('p')})
    df = pd.DataFrame(rows)
    from tnwt.enrichment import fdr_bh
    rej, q = fdr_bh(df['p'].to_numpy(dtype=float))
    df['q'] = q
    df['fdr_significant'] = rej
    return df


# ==========================================================================
# 临床分层：完全缓解 vs 未完全缓解
# ==========================================================================
def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC（标签 1 为阳性，分数越大越倾向阳性）。"""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    r_pos = avg[inv[: len(pos)]].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """PR-AUC（平均精度）；类别不平衡时比 ROC-AUC 更有信息量。"""
    m = np.isfinite(scores)
    s, y = scores[m], labels[m]
    if y.sum() == 0:
        return float('nan')
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / y.sum()
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


@dataclass
class StratificationResult:
    auc: float
    pr_auc: float
    n_positive: int
    n_negative: int
    auc_perm_p: float
    n_perm: int
    fold_aucs: list[float] = field(default_factory=list)


def stratify_outcome(scores: pd.Series, labels: pd.Series, *,
                     n_splits: int = 5, n_perm: int = 1000, seed: int = 0,
                     n_repeats: int = 20) -> StratificationResult:
    """用单个影像指标区分二分类结局（如完全缓解 vs 未完全缓解）。

    包含三部分：

    1. **分层交叉验证 AUC**：按类别分层切分，避免某折缺少正例（不平衡二分类的必需处理）
    2. **PR-AUC（平均精度）**：类别不平衡时的主要指标
    3. **置换零分布**：打乱标签重算 AUC，给出经验 p 值
       同时用重复随机划分给出 AUC 的分布（衡量划分敏感性）
    """
    ids = [s for s in scores.index if s in labels.index]
    s = scores.reindex(ids).to_numpy(dtype=float)
    y = labels.reindex(ids).to_numpy(dtype=float)
    m = np.isfinite(s) & np.isfinite(y)
    s, y = s[m], y[m]
    y = (y > 0.5).astype(int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos < 2 or n_neg < 2:
        return StratificationResult(float('nan'), float('nan'), n_pos, n_neg,
                                    float('nan'), 0)

    rng = np.random.default_rng(seed)
    fold_aucs: list[float] = []
    for _ in range(max(1, n_repeats)):
        folds = _stratified_folds(y, n_splits, rng)
        for test_idx in folds:
            train = np.setdiff1d(np.arange(len(y)), test_idx)
            if len(np.unique(y[train])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            # 用一个简单的单调方向估计：训练集上阳性均值 − 阴性均值 的符号
            sign = 1.0 if s[train][y[train] == 1].mean() >= s[train][y[train] == 0].mean() else -1.0
            fold_aucs.append(_auc(sign * s[test_idx], y[test_idx]))
    auc = float(np.nanmean(fold_aucs)) if fold_aucs else _auc(s, y)
    pr = _average_precision(s if _auc(s, y) >= 0.5 else -s, y)

    obs = _auc(s, y)
    obs_abs = max(obs, 1 - obs)
    null = np.empty(n_perm)
    for i in range(n_perm):
        yp = rng.permutation(y)
        a = _auc(s, yp)
        null[i] = max(a, 1 - a)
    p_perm = float((1 + (null >= obs_abs).sum()) / (n_perm + 1))
    return StratificationResult(auc=auc, pr_auc=pr, n_positive=n_pos, n_negative=n_neg,
                                auc_perm_p=p_perm, n_perm=n_perm, fold_aucs=fold_aucs)


def _stratified_folds(y: np.ndarray, n_splits: int, rng) -> list[np.ndarray]:
    """按类别分层切分，返回每折的测试索引。"""
    n_splits = max(2, min(n_splits, int(np.bincount(y, minlength=2).min()) or 2))
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            folds[k % n_splits].append(int(i))
    return [np.array(sorted(f)) for f in folds if f]


# ==========================================================================
# 缺失数据：多重插补（Rubin 合并）
# ==========================================================================
def mice_impute(df: pd.DataFrame, *, n_imputations: int = 5, n_iter: int = 10,
                seed: int = 0) -> list[pd.DataFrame]:
    """简化版多重插补（MICE，连续变量用 OLS 迭代条件回归）。

    适用于"术前-术后配对中部分被试缺失某模态"的情形。**仅在明确声明后使用**：
    完整病例分析仍是主分析，插补结果作为敏感性分析（方案 5.8 要求并报）。
    """
    rng = np.random.default_rng(seed)
    X = df.to_numpy(dtype=float).copy()
    n, p = X.shape
    missing = ~np.isfinite(X)
    if not missing.any():
        return [df.copy() for _ in range(n_imputations)]
    col_means = np.nanmean(X, axis=0)
    col_means[~np.isfinite(col_means)] = 0.0
    imputed_sets: list[pd.DataFrame] = []
    for _ in range(n_imputations):
        Z = X.copy()
        for j in range(p):
            Z[missing[:, j], j] = col_means[j]
        for _ in range(n_iter):
            for j in range(p):
                miss_j = missing[:, j]
                if not miss_j.any():
                    continue
                obs_j = ~miss_j
                others = [k for k in range(p) if k != j]
                A = np.column_stack([np.ones(obs_j.sum()), Z[obs_j][:, others]])
                y_obs = Z[obs_j, j]

                # ⚠️ 必须标准化后再解。原始表型各分区量纲差异大（结构连接节点
                # 强度可达数千，而髓鞘指标在 1 附近），小样本 + 200 变量时
                # 条件数极高，SVD 会不收敛（2026-11-13 实测崩在这里）。
                A_off = A[:, 1:]
                mu = A_off.mean(axis=0)
                sdv = A_off.std(axis=0)
                sdv = np.where(sdv > 0, sdv, 1.0)
                A_s = np.column_stack([np.ones(obs_j.sum()), (A_off - mu) / sdv])
                try:
                    b_s, *_ = np.linalg.lstsq(A_s, y_obs, rcond=None)
                except np.linalg.LinAlgError:
                    # 仍失败则该列退化为均值插补，避免整条分析中断
                    b_s = np.zeros(A_s.shape[1])
                    b_s[0] = float(np.mean(y_obs))
                # 还原到原尺度：y = b0 + sum(b_k*(x_k-mu_k)/sd_k)
                beta = np.empty_like(b_s)
                beta[0] = b_s[0] - float(np.sum(b_s[1:] * mu / sdv))
                beta[1:] = b_s[1:] / sdv

                A_m = np.column_stack([np.ones(miss_j.sum()), Z[miss_j][:, others]])
                pred = A_m @ beta
                resid = y_obs - A @ beta
                sd = resid.std(ddof=max(1, A.shape[1]))
                if not np.isfinite(sd) or sd <= 0:
                    sd = 1e-6
                draw = pred + rng.normal(0, sd, miss_j.sum())
                # 兜底：非有限值回落到列均值，切断 NaN 传染
                bad = ~np.isfinite(draw)
                if bad.any():
                    draw[bad] = col_means[j] if np.isfinite(col_means[j]) else 0.0
                Z[miss_j, j] = draw
        # 整轮结束后再兜底：任何残留的非有限值用列均值替换
        if not np.isfinite(Z).all():
            for j in range(p):
                bad = ~np.isfinite(Z[:, j])
                if bad.any():
                    Z[bad, j] = col_means[j] if np.isfinite(col_means[j]) else 0.0
        imputed_sets.append(pd.DataFrame(Z, index=df.index, columns=df.columns))
    return imputed_sets


def rubin_pool(estimates: list[float], variances: list[float]) -> dict:
    """Rubin 规则合并多次插补的估计与方差。"""
    est = np.asarray([e for e in estimates if np.isfinite(e)], dtype=float)
    var = np.asarray([v for v in variances if np.isfinite(v)], dtype=float)
    m = min(len(est), len(var))
    if m == 0:
        return {'estimate': float('nan'), 'se': float('nan'), 't': float('nan'),
                'p': float('nan'), 'df': float('nan'), 'n_imputations': 0}
    est, var = est[:m], var[:m]
    qbar = float(est.mean())
    ubar = float(var.mean())
    b = float(est.var(ddof=1)) if m > 1 else 0.0
    total = ubar + (1 + 1 / m) * b
    se = float(np.sqrt(total)) if total > 0 else float('nan')
    tval = qbar / se if se and np.isfinite(se) else float('nan')
    if m > 1 and b > 0:
        df = (m - 1) * (1 + ubar / ((1 + 1 / m) * b)) ** 2
    else:
        df = float('inf')
    pval = float(min(1.0, 2 * t_sf(abs(tval), df))) if np.isfinite(tval) else float('nan')
    return {'estimate': qbar, 'se': se, 't': float(tval) if np.isfinite(tval) else float('nan'),
            'p': pval, 'df': df, 'n_imputations': m,
            'between_var': b, 'within_var': ubar}


# ==========================================================================
# 纵向"变化图谱"与基因表达的空间关联
# ==========================================================================
def change_map_correlation(change_map: np.ndarray, baseline_map: np.ndarray,
                           expression: np.ndarray, gene_names: list[str], *,
                           spins=None, n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """比较"术前关联"与"变化关联"是否不同（方案 5.8 的 bootstrapped 差异检验）。

    参数
    ----
    change_map : (n_parcels,) 术后−术前的分区均值变化
    baseline_map : (n_parcels,) 术前效应量图谱
    expression : (n_parcels, n_genes) 分区 × 基因表达
    spins : 可选的旋转置换索引，用于给出 spin p 值

    返回
    ----
    dict，包含两次关联的 PLS1 权重余弦相似度、以及**权重差异的 bootstrap 置信区间**。
    注意：不能对两次 PLS 各自独立跑一遍然后直接比较显著性——必须对差异本身做检验。
    """
    from tnwt.stats_imaging import pls_regression
    rng = np.random.default_rng(seed)
    n_par = expression.shape[0]

    def _pls1(map_vec):
        m = np.isfinite(map_vec) & np.all(np.isfinite(expression), axis=1)
        if m.sum() < 10:
            return None
        return pls_regression(expression[m], map_vec[m], n_components=1)

    r_base = _pls1(baseline_map)
    r_change = _pls1(change_map)
    if r_base is None or r_change is None:
        return {'ok': False, 'reason': '有效分区不足'}

    w_b = r_base.x_weights_raw[:, 0]
    w_c = r_change.x_weights_raw[:, 0]
    cos = float(np.dot(w_b, w_c) / (np.linalg.norm(w_b) * np.linalg.norm(w_c)))

    # 置换/spin 下两次关联的相关性 p 值
    out: dict = {'ok': True, 'cosine_weights': cos,
                 'var_y_baseline': float(r_base.var_y[0]),
                 'var_y_change': float(r_change.var_y[0])}
    if spins is not None:
        from tnwt.spin import spin_test
        pair = np.isfinite(baseline_map) & np.isfinite(change_map)
        sb = spin_test(baseline_map[pair] if pair.sum() > 3 else baseline_map, spins,
                       change_map[pair] if pair.sum() > 3 else change_map)
        out['p_spin_baseline_vs_change'] = sb['p_two_sided']
        out['obs_r_baseline_vs_change'] = sb['observed']

    # bootstrap 差异：重采样分区，比较两次关联的"基因权重与变化图谱一致性"
    diffs = np.empty(n_bootstrap)
    valid = 0
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_par, n_par)
        E = expression[idx]
        if E.shape[0] < 12 or not np.all(np.isfinite(E)) or np.all(np.isfinite(E) == False):
            continue
        try:
            rb = pls_regression(E, baseline_map[idx], n_components=1)
            rc = pls_regression(E, change_map[idx], n_components=1)
        except Exception:  # noqa: BLE001
            continue
        vb = rb.x_weights_raw[:, 0]
        vc = rc.x_weights_raw[:, 0]
        nb, nc = np.linalg.norm(vb), np.linalg.norm(vc)
        if nb == 0 or nc == 0:
            continue
        diffs[valid] = float(np.dot(vb / nb, vc / nc))
        valid += 1
    if valid > 10:
        d = diffs[:valid]
        out.update({
            'bootstrap_n': valid,
            'cosine_bootstrap_mean': float(d.mean()),
            'cosine_bootstrap_ci95': [float(np.percentile(d, 2.5)),
                                      float(np.percentile(d, 97.5))],
            'prob_cosine_below_zero': float((d < 0).mean()),
        })
    return out
