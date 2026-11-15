#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
零依赖统计工具（仅 numpy/pandas）。

存在理由：DSH 捆绑 Python 运行时只含 numpy/pandas/openpyxl/python-docx 等，
不含 scipy/statsmodels。为了让"数据清点与描述统计"这一步在任何环境下都能跑通，
这里用 numpy 实现所需的分布函数与检验，不引入编译型依赖。

已实现：
- _betainc_reg  : 正则化不完全 Beta 函数（连分式，Numerical Recipes 风格）
- t_sf          : t 分布上尾概率
- welch_ttest   : Welch 两样本 t 检验
- chi2_sf       : 卡方分布上尾概率（级数）
- chi2_contingency : 列联表卡方检验（含 Yates 校正选项）
- cohen_d / hedges_g

若后续安装了 scipy，可直接用 scipy 替换（接口保持一致的返回值结构）。
"""
from __future__ import annotations

import numpy as np

_FPMIN = 1e-300
_EPS = 3e-16


def _betacf(a: float, b: float, x: float, maxit: int = 300) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc_reg(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = _lgamma(a + b) - _lgamma(a) - _lgamma(b)
    front = np.exp(ln_beta + a * np.log(x) + b * np.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _lgamma(z: float) -> float:
    # Lanczos 近似（g=7，9 个系数），精度约 1e-13，足够描述统计使用
    g = 7
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    if z < 0.5:
        return float(np.log(np.pi / np.sin(np.pi * z)) - _lgamma(1.0 - z))
    z -= 1.0
    x = c[0]
    for i in range(1, g + 2):
        x += c[i] / (z + i)
    t = z + g + 0.5
    return float(0.5 * np.log(2 * np.pi) + (z + 0.5) * np.log(t) - t + np.log(x))


def _finite_df(df: float) -> float:
    """把 +inf/NaN 自由度映射为极大有限值。

    t 分布在 df→∞ 时趋于标准正态。实现中 x = df/(df+t²) 在 df=inf 时会得到
    inf/inf = NaN，因此必须在入口处替换为极大有限值（否则 p 值静默变为 NaN，
    进而被上层 min(1.0, ...) 变成 1.0，表现为"极显著却 p=1"的错误结论）。
    实测：Rubin 合并在插补间方差为 0 时给出 df=inf，正会触发该路径。
    """
    if not np.isfinite(df) or df <= 0:
        return 1e12
    return float(df)


def t_sf(t: float, df: float) -> float:
    """P(T > t)。双尾概率需乘 2。df 为 inf/NaN 时按标准正态处理。"""
    d = _finite_df(df)
    x = d / (d + t * t)
    p = 0.5 * betainc_reg(d / 2.0, 0.5, x)
    return p if t > 0 else 1.0 - p


def welch_ttest(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return dict(t=np.nan, df=np.nan, p=np.nan, mean_a=np.nan, mean_b=np.nan, n_a=na, n_b=nb)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va / na + vb / nb
    t = (a.mean() - b.mean()) / np.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return dict(t=float(t), df=float(df), p=float(min(1.0, 2 * t_sf(abs(t), df))),
                mean_a=float(a.mean()), mean_b=float(b.mean()), n_a=na, n_b=nb)


def _gammainc_P(a: float, x: float) -> float:
    """正则化下不完全 Gamma 函数 P(a, x)（Numerical Recipes: gammp）。"""
    if x <= 0.0:
        return 0.0
    gln = _lgamma(a)
    if x < a + 1.0:
        # 级数展开
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(500):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-15:
                break
        return float(total * np.exp(-x + a * np.log(x) - gln))
    # 连分式展开（Lentz 法）
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return float(1.0 - np.exp(-x + a * np.log(x) - gln) * h)


def chi2_sf(x: float, k: int) -> float:
    """卡方上尾概率 P(X > x)，自由度 k。"""
    if x <= 0:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - _gammainc_P(k / 2.0, x / 2.0))))


def chi2_contingency(table, correction: bool = False):
    table = np.asarray(table, dtype=float)
    total = table.sum()
    row = table.sum(axis=1, keepdims=True)
    col = table.sum(axis=0, keepdims=True)
    expected = row @ col / total
    with np.errstate(divide='ignore', invalid='ignore'):
        stat = np.nansum((np.abs(table - expected) - (0.5 if correction else 0.0)) ** 2 / expected)
    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    return dict(chi2=float(stat), dof=int(dof), p=chi2_sf(stat, dof), expected=expected)


def cohen_d(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan')
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return float('nan')
    return float((a.mean() - b.mean()) / sp)


def hedges_g(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    d = cohen_d(a, b)
    if not np.isfinite(d) or na + nb < 4:
        return float('nan')
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float(d * j)


if __name__ == '__main__':
    # 自检：与 scipy 参考值比对（scipy 可用时启用）
    a = [1.2, 1.5, 1.9, 2.2, 2.6, 3.0, 3.4]
    b = [0.8, 1.1, 1.3, 1.6, 1.9, 2.1]
    print('welch:', welch_ttest(a, b))
    print('chi2 p (df=1, x=3.84):', chi2_sf(3.84, 1), '(期望≈0.05)')
    print('t_sf(2.0, df=10)*2:', 2 * t_sf(2.0, 10), '(期望≈0.0734)')
    try:
        from scipy import stats as sps  # type: ignore
        print('scipy welch:', sps.ttest_ind(a, b, equal_var=False))
        print('scipy chi2:', sps.chi2.sf(3.84, 1))
    except Exception as exc:  # noqa: BLE001
        print('scipy 不可用（预期）:', exc)
