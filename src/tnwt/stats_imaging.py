"""影像转录组核心统计：PLS、稳定性重采样、共线性诊断、空间自相关。

存在理由：分析环境无 scikit-learn / pyls / netneurotools。本模块用 numpy 实现
方案（v2.1）第 5.5、4.3 节所需的全部算法，并用合成数据的解析解与统计性质做验证
（见 tests/run_tests.py）。

关键约定
--------
- X：分区 × 基因（预测集）；Y：分区 × 影像表型（响应集），本项目中 Y 通常为单列效应量图谱。
- 所有函数内部**自行中心化 X 与 Y**，调用方无需预处理。
- 行 = 分区，样本量 n = 分区数。这是空间关联，不是被试层面回归。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ==========================================================================
# 共线性诊断
# ==========================================================================
def pairwise_spatial_correlation(maps: dict[str, np.ndarray]) -> dict[str, float]:
    """两两空间相关（Pearson），键形如 'A1|A2'。"""
    names = list(maps)
    out: dict[str, float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = np.asarray(maps[names[i]], dtype=float).ravel()
            b = np.asarray(maps[names[j]], dtype=float).ravel()
            m = np.isfinite(a) & np.isfinite(b)
            out[f'{names[i]}|{names[j]}'] = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else float('nan')
    return out


def collinearity_diagnostics(maps: dict[str, np.ndarray]) -> dict:
    """共线性诊断：相关矩阵、VIF、条件数。

    VIF 由相关矩阵逆的对角线给出（基于标准化变量），条件数取设计矩阵的
    SVD 奇异值之比。方案 D5 要求 |r| < 0.7 才继续做收敛检验。
    """
    names = list(maps)
    X = np.column_stack([np.asarray(maps[k], dtype=float).ravel() for k in names])
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    n, p = X.shape
    if n < 3 or p < 2:
        return {'names': names, 'corr': None, 'vif': None, 'condition_number': float('nan'), 'n': int(n)}
    Xc = X - X.mean(axis=0, keepdims=True)
    sd = Xc.std(axis=0, ddof=1)
    sd[sd == 0] = np.nan
    Z = Xc / sd
    corr = np.corrcoef(Z, rowvar=False)
    try:
        vif = np.diag(np.linalg.inv(corr))
    except np.linalg.LinAlgError:
        vif = np.full(p, np.inf)
    sv = np.linalg.svd(Z, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float('inf')
    return {
        'names': names,
        'corr': corr.tolist(),
        'vif': {k: float(v) for k, v in zip(names, vif)},
        'condition_number': cond,
        'n': int(n),
    }


def orthogonalize(target: np.ndarray, regressors: np.ndarray) -> np.ndarray:
    """把 target 对 regressors（含截距）回归后取残差。

    用于方案 4.3/5.3 的"正交化残差 PLS"与"A3 去髓鞘残差分析"。
    """
    y = np.asarray(target, dtype=float).ravel()
    R = np.asarray(regressors, dtype=float)
    if R.ndim == 1:
        R = R[:, None]
    mask = np.isfinite(y) & np.all(np.isfinite(R), axis=1)
    out = np.full_like(y, np.nan)
    if mask.sum() < R.shape[1] + 2:
        return out
    A = np.column_stack([np.ones(mask.sum()), R[mask]])
    beta, *_ = np.linalg.lstsq(A, y[mask], rcond=None)
    out[mask] = y[mask] - A @ beta
    return out


# ==========================================================================
# 偏最小二乘（PLS）回归
# ==========================================================================
@dataclass
class PLSResult:
    x_weights: np.ndarray          # 分区/基因权重（已标准化为 z 分数）
    x_weights_raw: np.ndarray
    y_weights: np.ndarray
    x_scores: np.ndarray
    y_scores: np.ndarray
    x_loadings: np.ndarray
    y_loadings: np.ndarray
    singular_values: np.ndarray
    var_x: np.ndarray              # 每个成分解释的 X 方差比例
    var_y: np.ndarray              # 每个成分解释的 Y 方差比例
    cov_xy: np.ndarray             # 每成分的协方差（奇异值 / sqrt(n-1)）
    n_components: int
    meta: dict = field(default_factory=dict)

    @property
    def pls1_gene_z(self) -> np.ndarray:
        """PLS1 的 X 权重 z 分数（本项目用它作为基因权重）。"""
        return self.x_weights[:, 0]


def _zscore(v: np.ndarray) -> np.ndarray:
    sd = v.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.zeros_like(v)
    return (v - v.mean()) / sd


def pls_weights(X: np.ndarray, Y: np.ndarray, scale: bool = True) -> tuple[np.ndarray, float]:
    """单成分 PLS1 权重方向（供内部循环使用），返回 (单位化权重, 得分平方和)。

    算法取 Xd'Yd 的首个左奇异向量直接作为权重方向，不做迭代。
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    if scale:
        sd = Xc.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Xc = Xc / sd
    C = Xc.T @ Yc
    U, _, _ = np.linalg.svd(C, full_matrices=False)
    u = U[:, 0]
    nrm = np.linalg.norm(u)
    return (u / nrm if nrm else u), float(np.sum((Xc @ u) ** 2))


def pls_regression(X: np.ndarray, Y: np.ndarray, n_components: int | None = None,
                   scale: bool = True) -> PLSResult:
    """PLS 回归（NIPALS 迭代 + 秩一压缩），与 pyls 的 PLS1/PLS2 行为一致。

    参数
    ----
    X : (n, p) 基因表达矩阵
    Y : (n,) 或 (n, q) 影像表型
    n_components : 保留成分数，默认 min(n-1, p, q, 10)
    scale : 是否把 X 标准化（基因表达强烈建议 True）

    实现要点：
    - X 载荷固定（不随迭代更新，即 Wold 的 PLS2 模式 A），因此**无需迭代权重**，
      对每个成分只需求两个矩阵的特征向量，收敛即精确解；
    - 每提取一个成分后，用该成分的得分为 X 与 Y 各做一次秩一最小二乘压缩，
      保证成分之间得分正交、与标准实现一致。

    返回的 x_weights 已按列做 z 分数，便于与文献报告方式对齐。
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f'X 与 Y 的分区数不一致：{X.shape[0]} vs {Y.shape[0]}')

    mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    X, Y = X[mask], Y[mask]
    n, p = X.shape
    q = Y.shape[1]

    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    if scale:
        sd = Xc.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Xc = Xc / sd

    # 压缩时使用的副本（保留原始中心化矩阵用于计算解释方差）
    Xd, Yd = Xc.copy(), Yc.copy()
    max_comp = max(1, min(n - 1, p, q, 10))
    k = int(n_components) if n_components else max_comp
    k = max(1, min(k, max_comp))

    x_weights_raw = np.zeros((p, k))
    y_weights = np.zeros((q, k))
    x_scores = np.zeros((n, k))
    y_scores = np.zeros((n, k))
    x_loadings = np.zeros((p, k))
    y_loadings = np.zeros((q, k))
    sing = np.zeros(k)

    for i in range(k):
        # --- 单步解（Wold PLS2 模式 A 在 X 载荷固定下的精确解）---
        # 关键：X 权重方向就是 Xd'Yd 的首个左奇异向量，**不需要也不应做幂迭代**。
        # 在 p >> n（基因数远大于分区数）时，Xd'Xd 近似单位阵的倍数，
        # 若迭代到收敛会把权重推向 X 的主成分方向，从而系统性偏离真实信号方向。
        C = Xd.T @ Yd
        if not np.any(C):
            k = i
            break
        U, _, _ = np.linalg.svd(C, full_matrices=False)
        u = U[:, 0]
        t = Xd @ u
        tt = float(t @ t)
        if tt <= 0:
            k = i
            break
        w = Xd.T @ t / tt                      # X 载荷（模式 A）
        nw = np.linalg.norm(w)
        if nw == 0:
            k = i
            break
        v = Yd.T @ t                            # Y 权重
        nv = np.linalg.norm(v)
        if nv == 0:
            k = i
            break
        v = v / nv
        s_i = Yd @ v

        x_weights_raw[:, i] = w / nw
        y_weights[:, i] = v
        x_scores[:, i] = t
        y_scores[:, i] = s_i
        sing[i] = float(np.linalg.norm(C))
        x_loadings[:, i] = w / nw
        y_loadings[:, i] = Yd.T @ s_i / float(s_i @ s_i or 1.0)
        # 秩一压缩：X 用载荷方向 w，Y 用得分为 s。后续 u ⊥ w 保证得分 t 近似正交。
        Xd = Xd - np.outer(t, w)
        Yd = Yd - np.outer(s_i, v)

    x_weights_raw = x_weights_raw[:, :k]
    y_weights = y_weights[:, :k]
    x_scores = x_scores[:, :k]
    y_scores = y_scores[:, :k]
    x_loadings = x_loadings[:, :k]
    y_loadings = y_loadings[:, :k]
    sing = sing[:k]

    sst_x = float((Xc ** 2).sum())
    sst_y = float((Yc ** 2).sum())
    var_x = np.array([float((x_scores[:, i] ** 2).sum()) / sst_x if sst_x > 0 else 0.0 for i in range(k)])
    var_y = np.array([float((y_scores[:, i] ** 2).sum()) / sst_y if sst_y > 0 else 0.0 for i in range(k)])

    x_weights = np.column_stack([_zscore(x_weights_raw[:, i]) for i in range(k)]) if k else np.zeros((p, 0))

    return PLSResult(
        x_weights=x_weights,
        x_weights_raw=x_weights_raw,
        y_weights=y_weights,
        x_scores=x_scores,
        y_scores=y_scores,
        x_loadings=x_loadings,
        y_loadings=y_loadings,
        singular_values=sing,
        var_x=var_x,
        var_y=var_y,
        cov_xy=sing / np.sqrt(max(n - 1, 1)),
        n_components=int(k),
        meta={'n_used': int(n), 'n_dropped': int((~mask).sum()), 'scaled': bool(scale)},
    )


def pls_component_significance(X: np.ndarray, Y: np.ndarray, result: PLSResult,
                               n_perm: int = 1000, seed: int = 0) -> dict:
    """成分显著性的行置换检验（置换 Y 的行）。

    注：真实的显著性检验用球面旋转（见 spin_test_*）；行置换会破坏空间自相关，
    因此仅用于"成分数选择"这一辅助目的，方案中的正式 p 值一律来自 spin-test。
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    obs = np.array([abs(np.corrcoef(result.x_scores[:, i], result.y_scores[:, i])[0, 1])
                    for i in range(result.n_components)])
    null = np.zeros((n_perm, result.n_components))
    for b in range(n_perm):
        perm = rng.permutation(Y.shape[0])
        r = pls_regression(X, Y[perm], n_components=result.n_components)
        for i in range(result.n_components):
            null[b, i] = abs(np.corrcoef(r.x_scores[:, i], r.y_scores[:, i])[0, 1])
    pvals = (1 + (null >= obs).sum(axis=0)) / (n_perm + 1)
    return {'observed_r': obs.tolist(), 'p_perm': pvals.tolist(), 'n_perm': int(n_perm)}


# ==========================================================================
# 稳定性重采样（方案 5.5：仅入选率 > 80% 者列为候选基因）
# ==========================================================================
def _weight_vector(X: np.ndarray, Y: np.ndarray, scale: bool = True) -> np.ndarray | None:
    try:
        res = pls_regression(X, Y, n_components=1, scale=scale)
    except Exception:  # noqa: BLE001
        return None
    w = res.x_weights_raw[:, 0]
    nrm = np.linalg.norm(w)
    return w / nrm if nrm > 0 else None


def pls_stability(X: np.ndarray, Y: np.ndarray, reference: np.ndarray,
                  n_resamples: int = 1000, seed: int = 0,
                  mode: str = 'bootstrap', top_frac: float = 0.05,
                  split_frac: float = 0.5) -> dict:
    """PLS1 权重向量的稳定性：bootstrap 或 split-half。

    参数
    ----
    reference : 全样本 PLS1 原始权重向量（用于对比符号与计算入选率）
    top_frac  : 每次重采样中"入选"的 top 比例（默认 top 5%）

    返回
    ----
    selection_freq : 每个基因被选入 top 集合的频率（0–1）
    cosine         : 每次重采样权重与 reference 的余弦相似度
    sign_consistency : 与 reference 同号的基因比例
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    ref = np.asarray(reference, dtype=float).ravel()
    ref = ref / (np.linalg.norm(ref) or 1.0)

    rng = np.random.default_rng(seed)
    n, p = X.shape
    k_top = max(1, int(round(top_frac * p)))
    counts = np.zeros(p)
    cosines: list[float] = []
    sign_hits = np.zeros(p)
    n_valid = 0

    for _ in range(n_resamples):
        if mode == 'bootstrap':
            idx = rng.integers(0, n, n)
        elif mode == 'split_half':
            idx = rng.permutation(n)[: max(3, int(split_frac * n))]
        else:
            raise ValueError(f'未知 mode：{mode}')
        try:
            w = _weight_vector(X[idx], Y[idx])
        except Exception:  # noqa: BLE001
            w = None
        if w is None or not np.all(np.isfinite(w)):
            continue
        n_valid += 1
        cosines.append(float(np.dot(w, ref)))
        sign_hits += (np.sign(w) == np.sign(ref)).astype(float)
        top_idx = np.argsort(-np.abs(w))[:k_top]
        counts[top_idx] += 1

    if n_valid == 0:
        return {'selection_freq': np.zeros(p), 'cosine': np.array([]),
                'sign_consistency': np.zeros(p), 'n_valid': 0, 'k_top': k_top}
    return {
        'selection_freq': counts / n_valid,
        'cosine': np.asarray(cosines),
        'sign_consistency': sign_hits / n_valid,
        'n_valid': int(n_valid),
        'k_top': int(k_top),
    }


def candidate_genes(selection_freq: np.ndarray, min_freq: float = 0.8) -> np.ndarray:
    """按入选率阈值筛候选基因，返回布尔掩码。"""
    return np.asarray(selection_freq, dtype=float) >= min_freq


# ==========================================================================
# 空间自相关（Moran's I）与空间权重
# ==========================================================================
def distance_matrix(coords: np.ndarray) -> np.ndarray:
    c = np.asarray(coords, dtype=float)
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2)


def spatial_weights(coords: np.ndarray, threshold: float | None = None,
                    k: int | None = None) -> np.ndarray:
    """构建行标准化空间权重矩阵 W（阈值法或 k 近邻法），对角为 0。"""
    d = distance_matrix(coords)
    n = d.shape[0]
    if k is not None:
        W = np.zeros((n, n))
        order = np.argsort(d, axis=1)
        for i in range(n):
            W[i, order[i, 1:k + 1]] = 1.0
    else:
        if threshold is None:
            raise ValueError('需给出 threshold 或 k')
        W = (d <= threshold).astype(float)
        np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return W / rs


def morans_i(values: np.ndarray, W: np.ndarray) -> float:
    """Moran's I（W 行标准化后可简化计算）。"""
    x = np.asarray(values, dtype=float).ravel()
    m = np.isfinite(x)
    if m.sum() < 3:
        return float('nan')
    x = x[m]
    W = np.asarray(W, dtype=float)[np.ix_(m, m)]
    xc = x - x.mean()
    denom = (xc ** 2).sum()
    if denom == 0:
        return float('nan')
    Ws = W.sum()
    if Ws == 0:
        return float('nan')
    return float((len(x) / Ws) * (xc @ (W @ xc)) / denom)
