"""跨模态收敛检验与稳定性分析（阶段七）。

对应研究方案 v2.1 第 4.3、5.5、5.7 节。本模块补齐三块此前只有文字描述、尚无实现的分析：

1. **共线性与收敛判据**（方案 4.3）：两两空间 |r| < 0.7 且 PLS1 权重余弦 > 0.5
   且 spin-p < 0.05（FDR）方判定收敛；若 |r| ≥ 0.7 则转正交化残差分析
2. **稳健性指标**（方案 5.7 第一级）：不同坐标集 / 归一化 / 阈值 / LODO 条件下
   top 基因集合的重叠率（Jaccard 与 overlap coefficient）
3. **独立一致性**（方案 5.7 第二级）：与外部报告的基因权重向量做空间相关

设计要点
--------
- 判据**事前固化**在配置中（``config/analysis.yaml`` 的 ``convergence`` 段），
  本模块只负责计算与判定，不临时调整阈值。
- 所有重叠率同时给出 Jaccard 与 overlap coefficient，因为前者会被集合大小差异
  惩罚、后者不会；两者的差异本身是"稳定性结论是否稳健"的信息。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tnwt.stats_imaging import collinearity_diagnostics, orthogonalize, pls_regression  # noqa: E402
from tnwt.enrichment import fdr_bh  # noqa: E402


# ==========================================================================
# 判据
# ==========================================================================
@dataclass
class ConvergenceCriteria:
    """方案 4.3 事前固化的收敛判据。"""
    max_pairwise_abs_r: float = 0.7
    min_weight_cosine: float = 0.5
    max_spin_p_fdr: float = 0.05

    @classmethod
    def from_config(cls, cfg: dict) -> 'ConvergenceCriteria':
        c = (cfg or {}).get('convergence') or {}
        return cls(
            max_pairwise_abs_r=float(c.get('max_pairwise_abs_r', 0.7)),
            min_weight_cosine=float(c.get('min_weight_cosine', 0.5)),
            max_spin_p_fdr=float(c.get('max_spin_p_fdr', 0.05)),
        )


def spatial_independence(maps: dict[str, np.ndarray]) -> dict:
    """计算两两空间相关并给出是否满足"可分离"前置条件。"""
    diag = collinearity_diagnostics(maps)
    corr = np.array(diag['corr'])
    names = list(maps)
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs[f'{names[i]}|{names[j]}'] = float(corr[i, j])
    return {'names': names, 'corr_matrix': corr.tolist(), 'pairwise_abs_r': pairs,
            'max_abs_r': float(np.nanmax(np.abs(list(pairs.values())))) if pairs else float('nan'),
            'vif': diag['vif'], 'condition_number': diag['condition_number'],
            'n': diag['n']}


# ==========================================================================
# PLS 权重一致性
# ==========================================================================
def weight_cosine(w1: np.ndarray, w2: np.ndarray) -> float:
    """两个权重向量的余弦相似度（带符号；负值表示方向相反）。"""
    a = np.asarray(w1, dtype=float).ravel()
    b = np.asarray(w2, dtype=float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float('nan')
    na, nb = np.linalg.norm(a[m]), np.linalg.norm(b[m])
    if na == 0 or nb == 0:
        return float('nan')
    return float(np.dot(a[m], b[m]) / (na * nb))


def convergence_test(effect_maps: dict[str, np.ndarray], expression: np.ndarray,
                     gene_names: list[str], *, criteria: ConvergenceCriteria,
                     spins_per_modality: dict[str, np.ndarray] | None = None,
                     n_components: int = 1,
                     excluded_modalities: dict[str, str] | None = None,
                     spin_p_precomputed: dict[str, float] | None = None,
                     spin_mode: str | None = None) -> dict:
    """完整收敛检验：可分离性 → 各模态 PLS → 权重一致性 → spin-p → FDR 判定。

    参数
    ----
    effect_maps : {模态名: 效应量图谱 (n_parcels,)}
    expression : (n_parcels, n_genes) 基因表达矩阵
    spins_per_modality : 可选，{模态名: 旋转置换索引 (n_perm, n_parcels)}
    excluded_modalities : 可选，{模态名: 剔除原因}

        对应决策日志 **D11**：组间效应图谱未通过 FDR 的模态**不参与收敛判定**
        （其实像层面无可靠信号，若仍纳入"收敛"会给出无法解释的结论）。
        被剔除的模态**仍会计算 PLS**（其转录组关联作为探索性结果报告），
        但不计入 ``converged`` 判定。

    返回
    ----
    dict，含各模态 PLS1 权重、两两余弦、spin p 值、FDR 校正后的判定，
    以及最终 ``converged`` 布尔值（参与判定的模态须**同时**满足全部判据）。
    """
    names = list(effect_maps)
    excluded = dict(excluded_modalities or {})
    participating = [n for n in names if n not in excluded]
    out: dict = {'modalities': names,
                 'participating_modalities': participating,
                 'excluded_modalities': excluded,
                 'criteria': {
                     'max_pairwise_abs_r': criteria.max_pairwise_abs_r,
                     'min_weight_cosine': criteria.min_weight_cosine,
                     'max_spin_p_fdr': criteria.max_spin_p_fdr,
                 }}
    if excluded:
        out['exclusion_note'] = (
            '按 D11：以下模态的组间图谱未通过 FDR，不参与收敛判定'
            f'（其转录组关联仅作探索性报告）：{excluded}')
    if len(participating) < 2:
        out['converged'] = False
        out['insufficient_participants'] = True
        out['note'] = (f'参与收敛判定的模态仅 {len(participating)} 个，'
                       '无法构成跨模态收敛检验')

    # 可分离性只在**参与判定**的模态之间评估
    indep = spatial_independence({k: effect_maps[k] for k in participating}
                                 if participating else effect_maps)
    out['spatial_independence'] = indep
    separable = bool(indep['max_abs_r'] < criteria.max_pairwise_abs_r) if (
        participating and len(participating) >= 2) else False
    out['separable'] = separable

    # 各模态独立 PLS
    weights, var_y, spins_p = {}, {}, {}
    for name in names:
        y = np.asarray(effect_maps[name], dtype=float).ravel()
        m = np.isfinite(y) & np.all(np.isfinite(expression), axis=1)
        if m.sum() < 10:
            out.setdefault('warnings', []).append(f'{name}: 有效分区不足，跳过')
            continue
        r = pls_regression(expression[m], y[m], n_components=n_components)
        weights[name] = r.x_weights_raw[:, 0]
        var_y[name] = float(r.var_y[0])
    out['n_modalities_analysed'] = len(weights)
    out['var_y'] = var_y

    # spin p 值
    # 优先使用调用方预先算好的 p（run_transcriptome 会按 --spin-mode 计算
    # fixed / refit / both，其中 refit 是"每次置换重新拟合 PLS"的严格零模型）。
    if spin_p_precomputed:
        for name in weights:
            if name in spin_p_precomputed:
                spins_p[name] = float(spin_p_precomputed[name])
        out['spin_mode'] = spin_mode or 'precomputed'
    elif spins_per_modality:
        from tnwt.spin import spin_test
        for name in weights:
            sp = spins_per_modality.get(name)
            if sp is None:
                continue
            y = np.asarray(effect_maps[name], dtype=float).ravel()
            x = expression @ weights[name]
            # 约定：旋转**影像图谱** y（与 spin_test_refit 一致），
            # 而不是旋转模型输出 x——两种索引族不同，混用会使数字不可比。
            st = spin_test(y, sp, x)
            spins_p[name] = st['p_two_sided']
        out['spin_mode'] = spin_mode or 'fixed_fallback'
    if spins_p:
        out['spin_p'] = spins_p
        keys = sorted(spins_p)
        rej, q = fdr_bh([spins_p[k] for k in keys])
        out['spin_q'] = {k: float(v) for k, v in zip(keys, q)}
        out['spin_fdr_significant'] = {k: bool(v) for k, v in zip(keys, rej)}

    # 两两权重余弦（只在参与判定的模态之间，避免被剔除模态污染结论）
    cos = {}
    for i, a in enumerate(participating):
        for b in participating[i + 1:]:
            if a in weights and b in weights:
                cos[f'{a}|{b}'] = weight_cosine(weights[a], weights[b])
    out['weight_cosine'] = cos
    min_cos = min(cos.values()) if cos else float('nan')
    out['min_pairwise_cosine'] = float(min_cos)

    # 判定（D11：只用参与判定的模态）
    cos_ok = bool(np.isfinite(min_cos) and min_cos >= criteria.min_weight_cosine)
    if 'spin_fdr_significant' in out and participating:
        p_ok = bool(all(out['spin_fdr_significant'].get(k, False)
                        for k in participating))
    else:
        p_ok = False
    enough = len(participating) >= 2
    out['criteria_met'] = {'separable': bool(separable), 'weight_cosine': cos_ok,
                           'spin_p_fdr': p_ok,
                           'enough_participants': enough}
    out['converged'] = bool(enough and separable and cos_ok and p_ok)

    # 不可分离时的正交化残差分析（方案 4.3 的回退路径；仅对参与判定的模态）
    if (not separable) and len(participating) >= 2:
        out['fallback'] = orthogonalized_residual_analysis(
            {k: effect_maps[k] for k in participating}, expression)
    return out


def orthogonalized_residual_analysis(effect_maps: dict[str, np.ndarray],
                                     expression: np.ndarray) -> dict:
    """正交化残差分析：把每个模态对其它模态回归后取残差，再各自做 PLS。

    用途：当两两 |r| ≥ 0.7（指标不可分离）时，检验某模态是否仍提供
    **独立于其他模态**的分子信息。方案 4.3 要求的回退路径。

    ⚠️ 重要局限（必须与结果一起报告）
    ---------------------------------
    当模态之间**高度共线**时（正是触发本回退路径的情形），回归设计矩阵条件数极大，
    正交化本身是**病态**的：所得的"残差"之间仍会存在非零相关，且其数值依赖于
    数值精度与变量顺序，不具可解释性。

    因此本函数额外返回 ``orthogonalization_condition_number``：
    **若条件数过大（本模块以 1e6 为阈值），残差分析的结论不可采信**，
    应在论文中明确说明"指标不可分离，模态特异性分子解释无法确立"，
    而不是把病态残差当作独立证据。
    """
    names = list(effect_maps)
    residuals: dict[str, np.ndarray] = {}
    out: dict = {'modalities': names, 'residual_variance_explained': {}}
    cond_numbers: dict[str, float] = {}
    for name in names:
        others = [effect_maps[k] for k in names if k != name]
        R = np.column_stack([np.asarray(o, dtype=float).ravel() for o in others])
        y = np.asarray(effect_maps[name], dtype=float).ravel()
        # 记录回归设计的条件数（标准化后）
        m0 = np.isfinite(y) & np.all(np.isfinite(R), axis=1)
        if m0.sum() > R.shape[1] + 1:
            A = np.column_stack([np.ones(m0.sum()), R[m0]])
            sd = A.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0
            sv = np.linalg.svd((A - A.mean(axis=0)) / sd, compute_uv=False)
            cond_numbers[name] = float(sv[0] / sv[-1]) if sv[-1] > 0 else float('inf')
        else:
            cond_numbers[name] = float('nan')
        res = orthogonalize(y, R)
        residuals[name] = res
        m = np.isfinite(y) & np.isfinite(res)
        sst = float(((y[m] - y[m].mean()) ** 2).sum())
        ssr = float((res[m] ** 2).sum())
        out['residual_variance_explained'][name] = (
            float(1 - ssr / sst) if sst > 0 else float('nan'))
    out['orthogonalization_condition_number'] = cond_numbers
    # ⚠️ 不能用 np.nanmax：条件数为 inf 时（截距列标准差为 0，必然如此）
    # np.nanmax 会返回 NaN，于是 "NaN > 阈值" 为 False，**病态保护会静默失效**。
    # 正确做法：只要存在 inf 就视为 inf。
    finite_conds = [v for v in cond_numbers.values() if np.isfinite(v)]
    if any(np.isinf(v) for v in cond_numbers.values()):
        max_cond = float('inf')
    elif finite_conds:
        max_cond = float(max(finite_conds))
    else:
        max_cond = float('nan')
    out['max_condition_number'] = max_cond
    ill_conditioned = bool((not np.isfinite(max_cond)) or max_cond > 1e6)
    out['ill_conditioned'] = ill_conditioned
    if ill_conditioned:
        out['interpretation_warning'] = (
            '正交化设计矩阵条件数过大（> 1e6），残差分析病态，'
            '其残差相关与 PLS 结果不可作为模态独立性的证据；'
            '应报告"指标不可分离"，不做模态特异性解释。')
    # 残差之间的相关
    resid_corr = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y2 = residuals[a], residuals[b]
            m = np.isfinite(x) & np.isfinite(y2)
            resid_corr[f'{a}|{b}'] = (float(np.corrcoef(x[m], y2[m])[0, 1])
                                      if m.sum() > 3 else float('nan'))
    out['residual_pairwise_r'] = resid_corr
    # 各残差的 PLS 解释方差（仅在条件数可接受时才有解释意义）
    var_y = {}
    for name in names:
        res = residuals[name]
        m = np.isfinite(res) & np.all(np.isfinite(expression), axis=1)
        if m.sum() < 10:
            continue
        r = pls_regression(expression[m], res[m], n_components=1)
        var_y[name] = float(r.var_y[0])
    out['residual_var_y'] = var_y
    return out


# ==========================================================================
# 稳健性：集合重叠率
# ==========================================================================
def overlap_coefficients(set_a, set_b) -> dict:
    """两个基因集合的重叠指标。

    - **Jaccard** = |A∩B| / |A∪B|：对集合大小差异敏感
    - **overlap coefficient** = |A∩B| / min(|A|,|B|)：对大小差异不敏感

    同时报告两者：当二者差异大时，说明"稳定性"结论部分由集合大小驱动，
    应在论文中显式说明。
    """
    A, B = set(set_a), set(set_b)
    inter = len(A & B)
    union = len(A | B)
    denom = min(len(A), len(B)) if A and B else 0
    return {'n_a': len(A), 'n_b': len(B), 'n_intersection': inter,
            'jaccard': (inter / union) if union else float('nan'),
            'overlap_coefficient': (inter / denom) if denom else float('nan')}


def top_gene_set(z: np.ndarray, gene_names: list[str], *, threshold: float | None = None,
                 top_frac: float | None = None, top_n: int | None = None) -> set[str]:
    """按阈值/比例/数量取 top 基因集合（三者择一）。"""
    z = np.asarray(z, dtype=float).ravel()
    names = list(gene_names)
    if len(z) != len(names):
        raise ValueError(f'权重与基因名长度不一致：{len(z)} vs {len(names)}')
    m = np.isfinite(z)
    if top_n is not None:
        idx = np.argsort(-np.abs(np.where(m, z, -np.inf)))[:top_n]
        return {names[i] for i in idx if m[i]}
    if top_frac is not None:
        k = max(1, int(round(top_frac * m.sum())))
        idx = np.argsort(-np.abs(np.where(m, z, -np.inf)))[:k]
        return {names[i] for i in idx if m[i]}
    if threshold is None:
        raise ValueError('需给出 threshold、top_frac 或 top_n')
    return {names[i] for i in range(len(z)) if m[i] and abs(z[i]) >= threshold}


def robustness_matrix(reference_z: np.ndarray, gene_names: list[str],
                      variants: dict[str, np.ndarray], *, threshold: float = 2.0,
                      top_frac: float | None = None) -> dict:
    """对多组扰动（坐标集/归一化/阈值/LODO）计算与参考结果的重叠率。

    ``variants`` : {扰动名: 该条件下的 PLS1 权重向量}
    """
    ref = top_gene_set(reference_z, gene_names, threshold=threshold, top_frac=top_frac)
    rows = {}
    for name, z in variants.items():
        if z is None or len(z) != len(reference_z):
            rows[name] = {'note': '权重长度不一致或缺失', **overlap_coefficients(ref, set())}
            continue
        s = top_gene_set(z, gene_names, threshold=threshold, top_frac=top_frac)
        rows[name] = overlap_coefficients(ref, s)
    vals = [r['overlap_coefficient'] for r in rows.values()
            if np.isfinite(r.get('overlap_coefficient', np.nan))]
    return {'reference_set_size': len(ref), 'variants': rows,
            'min_overlap_coefficient': float(min(vals)) if vals else float('nan'),
            'median_overlap_coefficient': float(np.median(vals)) if vals else float('nan')}


def external_consistency(own_z: np.ndarray, external_z: np.ndarray,
                         gene_names: list[str]) -> dict:
    """与外部报告的基因权重向量做一致性检验（方案 5.7 第二级验证）。

    按基因名对齐（取交集），返回交集基因数、权重相关与余弦。
    """
    own = {g: float(z) for g, z in zip(gene_names, np.asarray(own_z, dtype=float))}
    ext_arr = np.asarray(external_z, dtype=float)
    if len(ext_arr) != len(gene_names):
        raise ValueError('外部权重长度需与基因名一致（调用方负责对齐基因顺序）')
    ext = {g: float(z) for g, z in zip(gene_names, ext_arr)}
    common = sorted(g for g in own if g in ext
                    and np.isfinite(own[g]) and np.isfinite(ext[g]))
    if len(common) < 5:
        return {'n_common_genes': len(common),
                'note': f'共同基因仅 {len(common)} 个（少于 5），无法检验'}
    a = np.array([own[g] for g in common])
    b = np.array([ext[g] for g in common])
    return {'n_common_genes': len(common),
            'pearson_r': float(np.corrcoef(a, b)[0, 1]),
            'cosine': weight_cosine(a, b),
            'n_positive_agreement': int(np.sum(np.sign(a) == np.sign(b))),
            'sign_agreement_rate': float(np.mean(np.sign(a) == np.sign(b)))}


# ==========================================================================
# 报告渲染
# ==========================================================================
def render_convergence_report(conv: dict) -> str:
    """把收敛检验结果渲染为人类可读的 Markdown 片段。"""
    L = ['## 跨模态收敛检验', '']
    crit = conv.get('criteria') or {}
    L += [f"- 事前判据：两两空间 |r| < {crit.get('max_pairwise_abs_r')}；"
          f"PLS1 权重余弦 > {crit.get('min_weight_cosine')}；"
          f"spin-p（FDR）< {crit.get('max_spin_p_fdr')}", '']
    part = conv.get('participating_modalities')
    if part is not None:
        L += [f"- 参与判定的模态：{', '.join(part) if part else '（无）'}", '']
    excl = conv.get('excluded_modalities') or {}
    if excl:
        L += ['- **按 D11 剔除的模态**（组间图谱未过 FDR，不参与收敛判定）：', '']
        for k, v in excl.items():
            L.append(f'  - {k}：{v}')
        L += ['', '  这些模态的转录组关联仅作**探索性**结果报告，'
                  '不得用于"共享分子基础"结论。', '']
    if conv.get('insufficient_participants'):
        L += [f"- ⚠️ {conv.get('note')}", '']
    ind = conv.get('spatial_independence') or {}
    if ind:
        mx = ind.get('max_abs_r')
        mx_s = 'NA' if mx is None or not np.isfinite(mx) else f'{mx:.3f}'
        L += [f"- 可分离性：最大两两 |r| = {mx_s}"
              f"（条件数 {ind.get('condition_number'):.1f}）"
              f" → {'满足' if conv.get('separable') else '不满足'}", '']
        for k, v in (ind.get('pairwise_abs_r') or {}).items():
            L.append(f"  - |r({k})| = {abs(v):.3f}")
        L.append('')
    wc = conv.get('weight_cosine') or {}
    if wc:
        L += ['- PLS1 权重余弦：', '']
        for k, v in wc.items():
            L.append(f"  - {k}: {v:+.3f}")
        mc = conv.get('min_pairwise_cosine')
        L += ['', f"- 最小余弦 = {'NA' if mc is None or not np.isfinite(mc) else f'{mc:+.3f}'}", '']
    if conv.get('spin_p'):
        L += ['- spin 检验：', '', '  | 模态 | p | q（FDR） | 显著 | 参与判定 |',
              '  |---|---|---|---|---|']
        for k in sorted(conv['spin_p']):
            L.append(f"  | {k} | {conv['spin_p'][k]:.4f} | "
                     f"{conv.get('spin_q', {}).get(k, float('nan')):.4f} | "
                     f"{conv.get('spin_fdr_significant', {}).get(k)} | "
                     f"{'是' if (part is None or k in part) else '否（D11）'} |")
        L.append('')
    met = conv.get('criteria_met') or {}
    L += ['- **判据逐条核对**：', '',
          f"  - 参与模态数 ≥ 2：{met.get('enough_participants')}",
          f"  - 可分离：{met.get('separable')}",
          f"  - 权重一致性：{met.get('weight_cosine')}",
          f"  - spin 显著性：{met.get('spin_p_fdr')}", '',
          f"- **结论**：{'满足全部判据 → 支持共享分子基础' if conv.get('converged') else '未满足全部判据 → 报告模态特异性、指标不可分离或无法判定收敛'}", '']
    if conv.get('fallback'):
        fb = conv['fallback']
        mc = fb.get('max_condition_number')
        mc_s = 'inf' if (mc is not None and np.isinf(mc)) else f'{mc:.3g}' if mc is not None else 'NA'
        L += ['### 回退路径：正交化残差分析（因指标不可分离）', '',
              f"- 正交化设计矩阵最大条件数：{mc_s}", '']
        if fb.get('ill_conditioned'):
            L += [f"- ⚠️ **该方法在此情形下病态**：{fb.get('interpretation_warning')}", '']
        L += ['- 各模态被其他模态解释的方差比例：', '']
        for k, v in (fb.get('residual_variance_explained') or {}).items():
            L.append(f'  - {k}: {v:.3f}')
        L += ['', '- 残差间的两两相关（病态时非零，不可作为独立性证据）：', '']
        for k, v in (fb.get('residual_pairwise_r') or {}).items():
            L.append(f'  - {k}: {v:+.4f}')
        L += ['', '- 残差的 PLS1 解释方差（仅在条件数可接受时具解释意义）：', '']
        for k, v in (fb.get('residual_var_y') or {}).items():
            L.append(f'  - {k}: {v:.3f}')
        L.append('')
    return '\n'.join(L)
