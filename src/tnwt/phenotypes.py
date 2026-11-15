"""影像表型提取与混淆控制（阶段四–五）。

对应研究方案 v2.1 第 5.2、5.3 节，以及派生数据契约 ``docs/derived_data_contract.md``。

职责
----
1. 读取契约规定的派生数据（分区向量与连接矩阵），做格式与维度校验
2. 计算三个影像表型：
   - A1 白质连接强度：直接取用户的流线数 / FA 加权分区向量
   - A2 结构-功能耦合：由结构连接矩阵（SC）与功能连接矩阵（FC）/时间序列计算
     图谐波意义上的"结构对功能的解释度"。**由本模块计算，不从用户处接收**
   - A3 灰白质对比度：按分区向量取用，并以 T1w/T2w 作为平行髓鞘指标
3. 侧别处理：因 AHBA 仅有部分供体右半球采样，默认取"对侧"半球生成分析图谱，
   同时保留镜像对称图谱作敏感性分析
4. 混淆控制：以年龄（含平方项）、性别、平均 FD、Fazekas 分级为协变量取残差
5. 组间比较：Welch t 与 Hedges' g 效应量图谱，并支持样本量匹配 / FD 匹配 / 年龄匹配
   三类敏感性分析
6. 输出：分区 × 被试的表型矩阵 + 效应量图谱 + 报告

设计约束
--------
- 只使用 numpy/pandas（环境无 scipy/sklearn）。
- 所有输出为"真实数据计算所得"；本模块**不生成任何占位数值**。
- 缺失值一律以 NaN 传播，不做插补（插补策略属纵向分析，见阶段九）。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tnwt.stats_lite import hedges_g, welch_ttest  # noqa: E402

#: 逐被试分区向量目录（契约布局，见 docs/derived_data_contract.md）。
#:
#: ``a3_t1t2_std`` 是 ``a3_t1t2`` 的**白质内参标定版**。原始 T1w/T2w 存在严重的
#: 逐被试缩放异质性（逐被试中位数相差 9.3 倍、呈双峰，CV 0.571），
#: 标定后收敛至 1.6 倍、无双峰（CV 0.083）。
#:
#: **两者都保留**是刻意的：标定前后的分布对比本身就是论文必须报告的质量控制证据
#: （见 A3 技术备忘表 5-1）。但**下游统计与论文结论只能使用 ``a3_t1t2_std``**，
#: 原始版仅用于对照展示，不得参与组间比较解释或进入收敛判定。
PER_SUBJECT_DIRS = ('a1_streamlines', 'a1_fa', 'a3_gwc', 'a3_t1t2', 'a3_t1t2_std')
MATRIX_DIRS = ('a2/sc', 'a2/fc')
REQUIRED_QC = ('participant_id', 'session', 'bold_mean_fd')


# ==========================================================================
# 读取
# ==========================================================================
@dataclass
class Derivatives:
    vectors: dict[str, pd.DataFrame]      # 目录名 → (被试 × 分区)
    matrices: dict[str, dict[str, np.ndarray]]   # 目录名 → {被试: (n,n)}
    qc: pd.DataFrame
    atlas_labels: list[str] = field(default_factory=list)
    n_parcels: int = 0

    def subjects(self, kind: str) -> list[str]:
        df = self.vectors.get(kind)
        return list(df.index) if df is not None else []


def _read_vector_file(path: Path, n_parcels: int) -> np.ndarray:
    """读取单个被试的分区向量（支持长表 parcel_index/value 与宽表）。"""
    df = pd.read_csv(path)
    cols = [str(c) for c in df.columns]
    if 'parcel_index' in cols and 'value' in cols:
        idx = pd.to_numeric(df['parcel_index'], errors='coerce').to_numpy()
        vals = pd.to_numeric(df['value'], errors='coerce').to_numpy()
        order = np.argsort(idx)
        vals = vals[order]
    else:
        vals = pd.to_numeric(df[cols[0]], errors='coerce').to_numpy()
    if len(vals) != n_parcels:
        raise ValueError(f'{path.name}: 分区数 {len(vals)} != {n_parcels}')
    return vals.astype(float)


def _read_matrix_file(path: Path) -> np.ndarray:
    if path.suffix == '.npy':
        m = np.load(path)
    else:
        m = pd.read_csv(path, header=None).to_numpy(dtype=float)
    return np.asarray(m, dtype=float)


def load_derivatives(root: str | Path, atlas_labels: list[str] | None = None,
                     n_parcels: int = 200) -> Derivatives:
    """读取派生数据目录。

    注意：分区向量目录中可能同时包含**基线**与**随访**被试（后者 ID 常带 ``fu`` 后缀），
    两者会被一并读入同一个 DataFrame。调用方需自行按 ID 区分——见
    :func:`split_followup_rows` 与 :func:`load_derivatives_split`。
    """
    root = Path(root)
    vectors: dict[str, pd.DataFrame] = {}
    matrices: dict[str, dict[str, np.ndarray]] = {}

    for sub in PER_SUBJECT_DIRS:
        d = root / sub
        if not d.is_dir():
            continue
        rows, names = [], []
        for p in sorted(d.glob('*.csv')):
            rows.append(_read_vector_file(p, n_parcels))
            names.append(p.stem)
        if rows:
            vectors[sub] = pd.DataFrame(np.vstack(rows), index=names,
                                        columns=[f'P{i+1}' for i in range(n_parcels)])

    for sub in MATRIX_DIRS:
        d = root / sub
        if not d.is_dir():
            continue
        mats: dict[str, np.ndarray] = {}
        for p in sorted(list(d.glob('*.npy')) + list(d.glob('*.csv'))):
            stem = p.stem
            parts = stem.split('_')
            key = parts[0]
            kind = '_'.join(parts[1:]) if len(parts) > 1 else ''
            # ⚠️ `a2/sc/` 下每个被试有**两个**矩阵（`_streamlines` 与 `_fa`），
            # 它们 split('_')[0] 都得到同一个被试名。首版靠字典覆盖，
            # 恰好字母序靠后的 `_streamlines` 胜出 —— 结果对，但纯属偶然，
            # 命名一变就会静默用错矩阵。
            # 这里改为**显式**指定：a2/sc 只取流线数矩阵（`_streamlines`），
            # 其余目录接受无后缀或任意后缀。
            if sub == 'a2/sc' and kind and kind != 'streamlines':
                continue
            m = _read_matrix_file(p)
            if m.ndim == 2 and m.shape == (n_parcels, n_parcels):
                mats[key] = m
        if mats:
            matrices[sub] = mats

    qc_path = root / 'qc' / 'subjects_qc.csv'
    qc = pd.read_csv(qc_path) if qc_path.exists() else pd.DataFrame()

    return Derivatives(vectors=vectors, matrices=matrices, qc=qc,
                       atlas_labels=atlas_labels or [], n_parcels=n_parcels)


FOLLOWUP_SUFFIXES = ('fu', '_ses-2')


def is_followup_id(sid: str) -> bool:
    """判断被试 ID 是否属于随访（本项目 ds005713 用 ``sub-009fu``）。"""
    return str(sid).endswith(FOLLOWUP_SUFFIXES)


def split_followup_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把「基线 + 随访」混合矩阵拆成两个 DataFrame（保持原始顺序）。"""
    flags = np.array([is_followup_id(i) for i in df.index], dtype=bool)
    return df[~flags], df[flags]


def load_derivatives_split(root: str | Path, n_parcels: int = 200
                           ) -> tuple[Derivatives, Derivatives]:
    """读取派生数据并按基线/随访拆分（向量与连接矩阵都拆）。

    返回 ``(baseline, followup)`` 两个 :class:`Derivatives`。这是纵向分析的必需入口：
    若只用 :func:`load_derivatives` 而不拆分，随访被试会与基线混在同一矩阵中，
    导致组间比较把随访当作独立被试（本项目早期即因此丢失全部随访数据）。
    """
    full = load_derivatives(root, n_parcels=n_parcels)

    def _subset(d: Derivatives, fu: bool) -> Derivatives:
        v: dict[str, pd.DataFrame] = {}
        for k, df in d.vectors.items():
            _, f = split_followup_rows(df)
            sub = f if fu else split_followup_rows(df)[0]
            if not sub.empty:
                v[k] = sub
        m: dict[str, dict[str, np.ndarray]] = {}
        for k, mats in d.matrices.items():
            sel = {s: a for s, a in mats.items() if is_followup_id(s) == fu}
            if sel:
                m[k] = sel
        qc = d.qc
        if not qc.empty and 'participant_id' in qc.columns:
            flags = qc['participant_id'].map(is_followup_id)
            qc = qc[flags] if fu else qc[~flags]
        return Derivatives(vectors=v, matrices=m, qc=qc,
                           atlas_labels=d.atlas_labels, n_parcels=d.n_parcels)

    return _subset(full, False), _subset(full, True)


# ==========================================================================
# A2：结构-功能耦合（图谐波）
# ==========================================================================
def structure_laplacian(sc: np.ndarray, normalize: bool = True) -> np.ndarray:
    """由结构连接矩阵构造拉普拉斯矩阵。

    ``sc`` 应非负且对称（流线数或 FA 加权）。返回对称拉普拉斯
    ``L = D - A``（``normalize=True`` 时返回对称归一化 ``I - D^{-1/2} A D^{-1/2}``）。
    """
    A = np.asarray(sc, dtype=float).copy()
    A = np.nan_to_num(A, nan=0.0)
    A = np.maximum(A, 0.0)
    A = (A + A.T) / 2.0                       # 强制对称
    np.fill_diagonal(A, 0.0)
    deg = A.sum(axis=1)
    deg[deg == 0] = 1.0
    if not normalize:
        return np.diag(deg) - A
    dinv = 1.0 / np.sqrt(deg)
    return np.eye(len(A)) - (A * dinv[:, None]) * dinv[None, :]


def harmonic_coupling(fc: np.ndarray, sc: np.ndarray, *, n_modes: int | None = None,
                      metric: str = 'r2') -> np.ndarray:
    """计算每个分区的"结构对功能的解释度"（A2 表型）。

    做法（图谐波 / 图信号处理）：

    1. 由结构连接 ``sc`` 构造拉普拉斯 ``L`` 并做特征分解，得到图谐波基 ``U``
       （低频模态 = 结构上平滑的模式）
    2. 把功能连接矩阵每一行视为定义在分区上的图信号 ``s_i``
    3. 用前 ``n_modes`` 个低频谐波基对 ``s_i`` 做最小二乘重建，得到 ``R²``
       ——即"结构约束能在多大程度上解释该分区的功能组织"

    参数
    ----
    n_modes : 使用的低频模态数，默认取 ``max(2, n_parcels // 10)``
    metric  : ``'r2'`` 返回 R²；``'r'`` 返回与重建信号的相关系数

    返回
    ----
    (n_parcels,) 每个分区的耦合值；对角元素被排除在拟合之外。
    """
    FC = np.asarray(fc, dtype=float)
    n = FC.shape[0]
    if n_modes is None:
        n_modes = max(2, n // 10)
    n_modes = int(min(max(2, n_modes), n - 2))

    L = structure_laplacian(sc)
    eigval, U = np.linalg.eigh(L)
    order = np.argsort(eigval)
    B = U[:, order[:n_modes]]                 # 低频谐波基 (n, k)

    out = np.full(n, np.nan)
    for i in range(n):
        s = FC[i].copy()
        s[i] = np.nan                        # 排除自身
        mask = np.isfinite(s)
        if mask.sum() < n_modes + 2:
            continue
        y = s[mask]
        A = B[mask]
        # 最小二乘重建
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ coef
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        if ss_tot <= 0:
            continue
        r2 = 1.0 - ss_res / ss_tot
        if metric == 'r2':
            out[i] = r2
        elif metric == 'r':
            denom = np.sqrt(ss_tot * float(((yhat - yhat.mean()) ** 2).sum()))
            out[i] = float(((y - y.mean()) * (yhat - yhat.mean())).sum() / denom) if denom > 0 else np.nan
        else:
            raise ValueError(f'未知 metric：{metric}')
    return out


def coupling_from_timeseries(ts: np.ndarray, sc: np.ndarray, **kwargs) -> np.ndarray:
    """若交付了分区时间序列，先算 FC 再算耦合（等价路径，便于交叉验证）。"""
    X = np.asarray(ts, dtype=float)
    if X.ndim != 2:
        raise ValueError(f'时间序列需为二维，实际 {X.shape}')
    # 统一为 (n_parcels, n_timepoints)：行数应与结构矩阵维度一致
    n = np.asarray(sc).shape[0]
    if X.shape[0] != n and X.shape[1] == n:
        X = X.T
    if X.shape[0] != n:
        raise ValueError(f'时间序列维度 {X.shape} 与分区数 {n} 不匹配')
    fc = np.corrcoef(X)                       # 与 harmonic_coupling 路径完全一致
    return harmonic_coupling(fc, sc, **kwargs)


def compute_a2(deriv: Derivatives, *, sc_kind: str = 'a2/sc', fc_kind: str = 'a2/fc',
               n_modes: int | None = None) -> pd.DataFrame:
    """对全部被试计算 A2，返回 (被试 × 分区) 矩阵。"""
    sc_all = deriv.matrices.get(sc_kind, {})
    fc_all = deriv.matrices.get(fc_kind, {})
    subjects = sorted(set(sc_all) & set(fc_all))
    rows, names = [], []
    for sid in subjects:
        rows.append(harmonic_coupling(fc_all[sid], sc_all[sid], n_modes=n_modes))
        names.append(sid)
    cols = [f'P{i+1}' for i in range(deriv.n_parcels)]
    return pd.DataFrame(np.vstack(rows) if rows else np.zeros((0, deriv.n_parcels)),
                        index=names, columns=cols)


# ==========================================================================
# 側别处理
# ==========================================================================
def split_hemispheres(n_parcels: int) -> tuple[np.ndarray, np.ndarray]:
    """Schaefer 图集按前一半左半球、后一半右半球排列。返回 (左索引, 右索引)。"""
    half = n_parcels // 2
    return np.arange(half), np.arange(half, n_parcels)


def contralateral_map(values: np.ndarray, pain_side: str) -> np.ndarray:
    """按疼痛侧别取对侧半球，并折叠为长度 n/2 的向量（索引 i 为左半球顺序）。

    ``pain_side='R'`` → 取左半球；``'L'`` → 取右半球；``'Bilateral'`` → 取两侧均值。
    """
    v = np.asarray(values, dtype=float)
    left, right = split_hemispheres(len(v))
    side = (pain_side or '').strip().upper()
    if side in ('R', 'RIGHT'):
        return v[left]
    if side in ('L', 'LEFT'):
        return v[right]
    return (v[left] + v[right]) / 2.0          # 双侧：取平均


def mirror_symmetric(values: np.ndarray) -> np.ndarray:
    """镜像对称图谱：左右半球取均值后对称化，保持原始长度。"""
    v = np.asarray(values, dtype=float)
    left, right = split_hemispheres(len(v))
    return np.concatenate([(v[left] + v[right]) / 2.0, (v[left] + v[right]) / 2.0])


# ==========================================================================
# 混淆控制
# ==========================================================================
CONFOUND_COLS = {
    'age': 'age',
    'sex': 'sex_label',
    'mean_fd': 'bold_mean_fd',
    'fazekas': 'fazekas_grade',
}


def _clinical_indexed(clinical: pd.DataFrame) -> pd.DataFrame:
    """把临床表按 participant_id 建索引，并处理重复行。

    真实数据中同一 ID 可能出现多行（例如基线行与随访行合并、或多来源拼接）。
    这里保留**首行**并去重，避免 ``reindex`` 因重复标签报错；
    调用方如需不同策略（如取末行、或按 session 过滤）应显式预处理临床表。
    """
    df = clinical.copy()
    if 'participant_id' not in df.columns:
        raise KeyError('临床表缺少 participant_id 列')
    dup = df['participant_id'].duplicated().sum()
    if dup:
        df = df.drop_duplicates(subset='participant_id', keep='first')
    return df.set_index('participant_id')


def _col_as_float(df: pd.DataFrame, name: str, n: int) -> np.ndarray | None:
    """安全地把临床表某一列转成 float 数组。

    真实数据的临床表可能缺少该列（``df.get`` 返回 ``None``）、或该列名与预期不符
    导致取到标量；此外列名可能带单位后缀（如 ``bold_mean_fd (mm)``）。
    本函数统一处理这些情形，避免在 ``residualize`` 这类热路径上崩在
    ``'numpy.float64' has no attribute 'to_numpy'``。
    """
    if name in df.columns:
        col = df[name]
    else:
        # 模糊匹配：忽略大小写与 " (xx)" 后缀
        target = name.lower()
        match = None
        for c in df.columns:
            base = str(c).split('(')[0].strip().lower()
            if base == target or str(c).lower().startswith(target):
                match = c
                break
        if match is None:
            return None
        col = df[match]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    s = pd.to_numeric(pd.Series(col), errors='coerce')
    arr = s.to_numpy(dtype=float)
    if arr.shape[0] != n:
        return None
    return arr


def build_design(clinical: pd.DataFrame, ids: list[str], *,
                 with_age_squared: bool = True) -> tuple[np.ndarray, list[str]]:
    """构造混淆协变量设计矩阵（已含截距）。返回 (X, 列名)。"""
    df = _clinical_indexed(clinical).reindex(ids)
    cols: list[np.ndarray] = []
    names: list[str] = []
    n = len(ids)
    age = _col_as_float(df, 'age', n)
    if age is not None and np.isfinite(age).any():
        a = np.where(np.isfinite(age), age, np.nanmean(age))
        cols.append(a)
        names.append('age')
        if with_age_squared:
            cols.append(a ** 2)
            names.append('age_sq')
    sex_raw = df['sex_label'] if 'sex_label' in df.columns else None
    if sex_raw is not None:
        s = pd.Series(sex_raw).astype(str).str.upper().str.strip()
        cols.append(s.isin(['M', 'MALE', '1']).astype(float).to_numpy())
        names.append('sex_male')
    fd = _col_as_float(df, 'bold_mean_fd', n)
    if fd is not None and np.isfinite(fd).any():
        cols.append(np.where(np.isfinite(fd), fd, np.nanmean(fd)))
        names.append('mean_fd')
    fz = _col_as_float(df, 'fazekas_grade', n)
    if fz is not None and np.isfinite(fz).any():
        cols.append(np.where(np.isfinite(fz), fz, 0.0))
        names.append('fazekas')
    if not cols:
        return np.ones((n, 1)), ['intercept']
    X = np.column_stack([np.ones(n)] + cols)
    return X, ['intercept'] + names


def residualize(matrix: pd.DataFrame, clinical: pd.DataFrame, *,
                with_age_squared: bool = True) -> pd.DataFrame:
    """对每个分区，以混淆协变量回归后取残差（逐列独立）。"""
    ids = list(matrix.index)
    X, _ = build_design(clinical, ids, with_age_squared=with_age_squared)
    out = np.full(matrix.shape, np.nan)
    for j in range(matrix.shape[1]):
        y = matrix.iloc[:, j].to_numpy(dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < X.shape[1] + 2:
            continue
        beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        out[mask, j] = y[mask] - X[mask] @ beta
    return pd.DataFrame(out, index=matrix.index, columns=matrix.columns)


# ==========================================================================
# 组间比较
# ==========================================================================
@dataclass
class ContrastResult:
    g: np.ndarray                 # Hedges' g（患者 - 对照）
    t: np.ndarray
    p: np.ndarray
    n_patient: int
    n_control: int
    parcels: list[str] = field(default_factory=list)

    def significant(self, alpha: float = 0.05, fdr: bool = True) -> np.ndarray:
        from tnwt.enrichment import fdr_bh
        if fdr:
            rej, _ = fdr_bh(self.p, alpha=alpha)
            return rej
        return self.p < alpha


def _dedup_series(s: pd.Series, name: str = 'group') -> pd.Series:
    """去掉重复索引，保留首次出现（与 ``_clinical_indexed`` 的策略一致）。"""
    if s.index.has_duplicates:
        s = s[~s.index.duplicated(keep='first')]
    return s.rename(name)


def group_contrast(matrix: pd.DataFrame, group: pd.Series, *,
                   patient_label: str = 'patient') -> ContrastResult:
    """逐分区 Welch t 检验与 Hedges' g（患者 − 对照）。"""
    g_vals = _dedup_series(group).reindex(matrix.index).astype(str)
    is_pat = (g_vals == patient_label).to_numpy()
    n_parcels = matrix.shape[1]
    g = np.full(n_parcels, np.nan)
    t = np.full(n_parcels, np.nan)
    p = np.full(n_parcels, np.nan)
    for j in range(n_parcels):
        col = matrix.iloc[:, j].to_numpy(dtype=float)
        a, b = col[is_pat], col[~is_pat]
        if np.isfinite(a).sum() < 3 or np.isfinite(b).sum() < 3:
            continue
        g[j] = hedges_g(a, b)
        r = welch_ttest(a, b)
        t[j] = r['t']
        p[j] = r['p']
    return ContrastResult(g=g, t=t, p=p,
                          n_patient=int(is_pat.sum()), n_control=int((~is_pat).sum()),
                          parcels=list(matrix.columns))


def n_matched_resampling(matrix: pd.DataFrame, group: pd.Series, *,
                         n_rep: int = 1000, seed: int = 0,
                         patient_label: str = 'patient') -> dict:
    """样本量匹配敏感性：从较大组随机抽取与较小组相同数量，重复 n_rep 次。"""
    g_vals = _dedup_series(group).reindex(matrix.index).astype(str)
    pat = list(g_vals.index[g_vals == patient_label])
    ctl = list(g_vals.index[g_vals != patient_label])
    n_small = min(len(pat), len(ctl))
    if n_small < 3:
        return {'n_rep': 0, 'g_mean': np.array([]), 'g_sd': np.array([]),
                'sign_consistency': np.array([]), 'n_small': n_small}
    rng = np.random.default_rng(seed)
    ref = group_contrast(matrix, group, patient_label=patient_label).g
    gs = np.empty((n_rep, matrix.shape[1]))
    for i in range(n_rep):
        sub_pat = rng.choice(pat, n_small, replace=False)
        sub_ctl = rng.choice(ctl, n_small, replace=False)
        sub = matrix.loc[list(sub_pat) + list(sub_ctl)]
        sub_group = pd.Series([patient_label] * n_small + ['control'] * n_small,
                              index=sub.index)
        gs[i] = group_contrast(sub, sub_group, patient_label=patient_label).g
    sign = np.nanmean(np.sign(gs) == np.sign(ref), axis=0)
    return {'n_rep': n_rep, 'g_mean': np.nanmean(gs, axis=0), 'g_sd': np.nanstd(gs, axis=0),
            'sign_consistency': sign, 'n_small': n_small}


def matched_subsample(matrix: pd.DataFrame, clinical: pd.DataFrame, *,
                      match_on: str = 'age', caliper: float = 0.25,
                      group: pd.Series | None = None,
                      patient_label: str = 'patient') -> list[str]:
    """按指定协变量做最邻近匹配，返回匹配后的被试 ID 列表（患者+对照）。

    用于方案 5.3 要求的**年龄匹配**与 **FD 匹配**敏感性分析。
    """
    df = _clinical_indexed(clinical)
    col = match_on if match_on in df.columns else CONFOUND_COLS.get(match_on, match_on)
    if col not in df.columns:
        raise KeyError(f'临床表中找不到匹配变量 {match_on}（尝试列 {col}）')
    vals = pd.to_numeric(df[col], errors='coerce')
    g = (group if group is not None else df.get('group_norm'))
    g = _dedup_series(pd.Series(g)).reindex(matrix.index).astype(str)
    pat = [s for s in matrix.index if g.get(s) == patient_label and np.isfinite(vals.get(s, np.nan))]
    ctl = [s for s in matrix.index if g.get(s) != patient_label and np.isfinite(vals.get(s, np.nan))]
    used: set[str] = set()
    out: list[str] = []
    sd = float(vals.loc[pat + ctl].std(ddof=1))
    tol = caliper * sd
    for p in sorted(pat, key=lambda s: float(vals[s])):
        avail = [c for c in ctl if c not in used]
        if not avail:
            break
        best = min(avail, key=lambda c: abs(float(vals[c]) - float(vals[p])))
        if abs(float(vals[best]) - float(vals[p])) <= tol:
            used.add(best)
            out += [p, best]
    return out


# ==========================================================================
# 汇总报告
# ==========================================================================
def phenotype_summary(matrix: pd.DataFrame, name: str) -> dict:
    finite = np.isfinite(matrix.to_numpy(dtype=float))
    return {
        'phenotype': name,
        'n_subjects': int(matrix.shape[0]),
        'n_parcels': int(matrix.shape[1]),
        'frac_finite': float(finite.mean()) if matrix.size else 0.0,
        'parcels_all_nan': int((~finite.any(axis=0)).sum()),
        'subjects_all_nan': int((~finite.any(axis=1)).sum()),
        'value_mean': float(np.nanmean(matrix.to_numpy(dtype=float))) if finite.any() else None,
        'value_sd': float(np.nanstd(matrix.to_numpy(dtype=float))) if finite.any() else None,
    }
