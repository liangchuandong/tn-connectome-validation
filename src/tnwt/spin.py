"""球面旋转置换检验（spin test）与多重比较校正。

存在理由：分析环境无 netneurotools。方案（v2.1）5.5 节要求用球形旋转置换
保留空间自相关结构，并明确旧接口 `spin_test` 已不推荐、应使用
`gen_spinsamples` / `rotate_spatial_data` 的等价实现。本模块提供该等价实现：

  gen_rotation_matrices -> 均匀采样 SO(3)
  gen_spinsamples       -> 旋转坐标并取最近邻，得到置换索引
  spin_test             -> 用置换索引做空间相关检验
  fdr_bh / fwer_perm    -> 多重比较校正

数学要点：对归一化到单位球的坐标施加均匀随机旋转，再按最近邻重新分配分区标签，
即可在**保留空间自相关**的前提下生成零分布。行置换（shuffle）不具备该性质，
因此不能替代 spin test。
"""
from __future__ import annotations

import numpy as np


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """在 SO(3) 上均匀采样旋转矩阵（QR 分解 + 反射修正）。"""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def gen_rotation_matrices(n_perm: int, seed: int = 0) -> np.ndarray:
    """生成 n_perm 个 3x3 旋转矩阵。"""
    rng = np.random.default_rng(seed)
    return np.stack([_random_rotation(rng) for _ in range(int(n_perm))])


def to_unit_sphere(coords: np.ndarray, center_by_mean: bool = True) -> np.ndarray:
    """把坐标投影到单位球（spin test 的标准前置步骤）。"""
    c = np.asarray(coords, dtype=float)
    if c.shape[1] != 3:
        raise ValueError(f'坐标需为 (n, 3)，实际 {c.shape}')
    if center_by_mean:
        c = c - c.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return c / norms


def gen_spinsamples(coords: np.ndarray, n_perm: int = 5000, seed: int = 0,
                    norm_sphere: bool = True, bijective: bool = True) -> np.ndarray:
    """生成旋转置换索引。

    返回 (n_perm, n) 的整数数组：``idx[p, i]`` 表示第 p 次旋转后，位置 i 上
    应放置的**原始分区编号**。

    实现：旋转单位球坐标 → 对每个旋转后的位置找最近的原始点。
    ``bijective=True`` 时使用**贪心唯一分配**（按距离全局升序配对，每个源点只用一次），
    保证得到一个真正的置换（无重复、无遗漏），与 netneurotools 的做法一致；
    这在零模型中是必要的，否则重复索引会在子集分析中造成越界与偏差。
    """
    c = np.asarray(coords, dtype=float)
    if norm_sphere:
        c = to_unit_sphere(c)
    rots = gen_rotation_matrices(n_perm, seed=seed)
    n = c.shape[0]
    idx = np.empty((len(rots), n), dtype=np.int64)
    for p, R in enumerate(rots):
        rotated = c @ R.T
        # d2[i, j] = 位置 i（旋转后）到原始点 j 的平方距离
        d2 = ((rotated[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
        if bijective:
            # 全局升序贪心：优先满足最近的点对，每个原始点只被分配一次
            flat = np.argsort(d2, axis=None, kind='stable')
            ti, sj = np.divmod(flat, n)
            taken_t = np.zeros(n, dtype=bool)
            taken_s = np.zeros(n, dtype=bool)
            assign = np.empty(n, dtype=np.int64)
            filled = 0
            for t, s in zip(ti, sj):
                if taken_t[t] or taken_s[s]:
                    continue
                assign[t] = s
                taken_t[t] = True
                taken_s[s] = True
                filled += 1
                if filled == n:
                    break
            if filled < n:   # 理论上不会发生，保留兜底
                for t in np.flatnonzero(~taken_t):
                    assign[t] = int(np.argmin(np.where(taken_s, np.inf, d2[t])))
                    taken_s[assign[t]] = True
            idx[p] = assign
        else:
            idx[p] = np.argmin(d2, axis=1)
    return idx


def spin_test(x: np.ndarray, spins: np.ndarray, y: np.ndarray | None = None,
              method: str = 'pearson'):
    """用旋转置换索引检验空间关联。

    参数
    ----
    x : (n,) 待旋转的空间图谱（通常是影像效应量图谱）
    spins : (n_perm, n) ``gen_spinsamples`` 的输出
    y : (n,) 固定图谱；若为 None 则检验 x 自身的空间自相关式一致性（不常用）
    method : 'pearson' 或 'spearman'

    返回
    ----
    dict(observed, null, p, p_two_sided, n_perm, z)
    """
    x = np.asarray(x, dtype=float).ravel()
    if y is None:
        y = x.copy()
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return {'observed': float('nan'), 'null': np.array([]), 'p': float('nan'),
                'p_two_sided': float('nan'), 'n_perm': 0, 'z': float('nan')}

    def corr(a, b):
        """Pearson 相关。

        归一化约定（已实测确认）：numpy 2.x 的 np.corrcoef 与 pandas 3.x 的 Series.corr
        均采用**总体标准差（ddof=0）**归一化，即
            r = mean((a-ā)(b-b̄)) / (std0(a) * std0(b))
        注意这与"用无偏标准误 ddof=1"的写法**不等价**（后者会小 (n-1)/n 倍）。
        本实现跟随 numpy/pandas 的约定，并用回归测试
        （test_spin_observed_equals_numpy_corrcoef）锁死该等价性，
        避免此类静默偏差系统性低估所有空间关联。
        """
        if method == 'spearman':
            a = _rank(a)
            b = _rank(b)
        sa, sb = a.std(ddof=0), b.std(ddof=0)
        if sa == 0 or sb == 0:
            return float('nan')
        return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))

    # 只在完整数据上做置换：把 x 的缺失位置剔除后重新索引
    xi = x[mask]
    yi = y[mask]
    obs = corr(xi, yi)

    # 缺失位置的处理：把子集索引压缩到 [0, m)，再对每次旋转的映射取子集。
    # 注意：compact 必须显式初始化为 -1，不能用 np.empty（被掩码漏下的槽位会保留垃圾值，
    # 而 s[keep] 恰好可能指向这些槽位，导致越界）。
    keep = np.flatnonzero(mask)
    m = len(keep)
    compact = np.full(len(x), -1, dtype=np.int64)
    compact[keep] = np.arange(m)
    null = np.empty(len(spins))
    for p, s in enumerate(spins):
        sub = s[keep]
        if np.any(compact[sub] < 0):     # 不应发生；兜底用秩映射
            order = np.argsort(sub, kind='stable')
            remap = np.empty(len(sub), dtype=np.int64)
            remap[order] = np.arange(len(sub))
            null[p] = corr(xi[remap], yi)
        else:
            null[p] = corr(xi[compact[sub]], yi)

    null = null[np.isfinite(null)]
    n_perm = len(null)
    if n_perm == 0 or not np.isfinite(obs):
        return {'observed': obs, 'null': null, 'p': float('nan'),
                'p_two_sided': float('nan'), 'n_perm': 0, 'z': float('nan')}

    # 双尾：以 |统计量| 比较（旋转零分布近似对称，取绝对值更稳健）
    p_two = (1 + (np.abs(null) >= abs(obs)).sum()) / (n_perm + 1)
    p_one = (1 + (null >= obs).sum()) / (n_perm + 1) if obs >= 0 else (1 + (null <= obs).sum()) / (n_perm + 1)
    z = (obs - null.mean()) / null.std(ddof=1) if null.std(ddof=1) > 0 else float('nan')
    return {'observed': float(obs), 'null': null, 'p': float(p_one),
            'p_two_sided': float(p_two), 'n_perm': int(n_perm), 'z': float(z)}


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind='stable')
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    return ranks


def spin_test_refit(stat_fn, y: np.ndarray, spins: np.ndarray) -> dict:
    """**每次置换都重新拟合模型**的空间置换检验（推荐用于投稿）。

    为什么需要它
    ------------
    :func:`spin_test` 采用"固定权重"方案：先在真实图谱上拟合模型，置换时只旋转图谱。
    当特征数远大于样本数时（本项目为 15,131 个基因 vs 193 个分区），模型可以拟合几乎
    任何平滑图谱；权重是专门为对齐真实图谱而挑出来的，因此

    - 观测统计量被系统性抬高；
    - 零分布却仍以 0 为中心，无法反映"挑选本身"带来的膨胀。

    结果是 **p 值偏小、检验过于宽松**。实测：A3_gwc 在固定权重下 p = 0.125，
    改为置换内重新拟合后 p = 0.527，而观测值 0.545 与零分布均值 0.550 几乎相等，
    属完全阴性。

    设计
    ----
    ``stat_fn`` 接收**换位后的全长图谱**（含 NaN），自行处理缺失值并返回一个标量统计量。
    把模型拟合完全交给调用方，本函数不依赖任何具体模型，因而不引入模块耦合。

    **旋转约定**：本函数旋转的是**影像图谱 ``y``**（即对 ``y`` 应用置换索引）。
    等价写法是 ``spin_test(y, spins, x)``——旋转 y、固定模型输出 x。
    注意 ``spin_test(x, spins, y)`` 旋转的是 x，索引族不同，零分布逐项并不相同，
    两方案并列报告时必须统一为同一约定，否则数字不可比。

    ``spins`` 的索引基于全长图谱，故 ``stat_fn`` 必须先在内部做掩码再取子集，
    不能直接拿全长去索引已经取过子集的矩阵（会越界）。

    返回
    ----
    dict(observed, null, p_two_sided, n_perm, null_mean, null_sd, null_max_abs)
    """
    y = np.asarray(y, dtype=float).ravel()
    obs = float(stat_fn(y))
    null = np.empty(len(spins), dtype=float)
    for i, s in enumerate(spins):
        null[i] = stat_fn(y[s])
    null = null[np.isfinite(null)]
    n_perm = len(null)
    if n_perm == 0 or not np.isfinite(obs):
        return {'observed': obs, 'null': null, 'p_two_sided': float('nan'),
                'n_perm': 0, 'null_mean': float('nan'), 'null_sd': float('nan'),
                'null_max_abs': float('nan')}
    p_two = (1 + int((np.abs(null) >= abs(obs)).sum())) / (n_perm + 1)
    return {'observed': obs, 'null': null, 'p_two_sided': float(p_two),
            'n_perm': int(n_perm), 'null_mean': float(null.mean()),
            'null_sd': float(null.std(ddof=1)),
            'null_max_abs': float(np.abs(null).max())}


def fdr_bh(pvals, alpha: float = 0.05):
    """Benjamini-Hochberg FDR。返回 (拒绝掩码, 校正后 q 值)。"""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return np.zeros_like(p, dtype=bool), q
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    qv = ranked * n / (np.arange(1, n + 1))
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qv = np.clip(qv, 0, 1)
    out = np.empty(n)
    out[order] = qv
    q[ok] = out
    return q <= alpha, q


def fwer_perm(observed: float, null: np.ndarray) -> float:
    """置换检验的 FWER 校正 p 值（对单个统计量即其置换 p 值）。"""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(observed):
        return float('nan')
    return float((1 + (np.abs(null) >= abs(observed)).sum()) / (len(null) + 1))
