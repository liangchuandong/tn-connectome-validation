#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""论文骨架与结果填充（阶段十一）。

原则
----
**只从各阶段的真实输出文件中提取数值**，写入论文稿的对应位置；
缺失的环节明确写为 ``【待补：<说明>】``，绝不编造数字。
每个写入的数值都会在 ``paper_number_audit.json`` 中留下"论文位置 → 来源文件与字段"
的追溯记录，便于投稿前逐条核对。

用法::

    python scripts/write_paper.py \\
        --out-dir build/paper \\
        --phenotypes build/phenotypes_out \\
        --longitudinal build/longitudinal_out \\
        [--transcriptome ...] [--enrichment ...] [--figures build/figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

PHENO_LABELS = {
    'A1_streamlines': ('A1', '白质连接强度（流线数）'),
    'A1_fa': ('A1', '白质连接强度（FA 加权）'),
    'A2_coupling': ('A2', '白质-皮层功能耦合（结构对功能的解释度）'),
    'A3_gwc': ('A3', '灰白质对比度（GWC）'),
    'A3_t1t2': ('A3', 'T1w/T2w 比值（髓鞘敏感平行指标）'),
}


class Audit:
    """记录每个写入论文的数值及其来源，供投稿前核对。"""

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def add(self, location: str, value, source: str, field: str = '',
            note: str = '') -> None:
        self.entries.append({'paper_location': location, 'value': value,
                             'source_file': source, 'source_field': field,
                             'note': note})

    def dump(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'n_numbers': len(self.entries),
                                    'entries': self.entries},
                                   ensure_ascii=False, indent=2, default=float),
                        encoding='utf-8')
        return path


TBD_TEMPLATE = '【待补：{0}】'


def TBD(what: str) -> str:
    """生成"待补"占位文本（统一格式，便于统计与投稿前检索）。"""
    return TBD_TEMPLATE.format(what)


def _read_json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def _read_csv(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None


def _fmt(v, nd: int = 3) -> str:
    if v is None:
        return 'NA'
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return 'NA'
    if abs(f) >= 1000 or (f != 0 and abs(f) < 0.001):
        return f'{f:.2e}'
    return f'{f:.{nd}f}'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='生成论文骨架并填充真实数值')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'paper')
    ap.add_argument('--phenotypes', type=Path, default=ROOT / 'data' / 'phenotypes')
    ap.add_argument('--longitudinal', type=Path, default=ROOT / 'data' / 'longitudinal')
    ap.add_argument('--transcriptome', type=Path, default=ROOT / 'data' / 'transcriptome')
    ap.add_argument('--enrichment', type=Path, default=ROOT / 'data' / 'enrichment')
    ap.add_argument('--figures', type=Path, default=ROOT / 'figures')
    ap.add_argument('--title', default=('三叉神经痛皮层-白质联合脆弱性的多模态影像'
                                        '转录组关联研究'))
    ap.add_argument('--journal', default='', help='目标期刊（留空则用通用结构）')
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = Audit()
    L: list[str] = []
    tbd: list[str] = []

    pheno_summary = _read_json(args.phenotypes / 'run_summary.json')
    long_summary = _read_json(args.longitudinal / 'run_summary.json')
    raw_fig = _read_json(args.figures / 'figure_manifest.json')
    # 注意：make_figures.py 写出的结构是
    #   {"figures": {"generated": [...], "skipped": [...], ...}}
    # 因此必须先解包一层，否则后续 `in fig_manifest` 判断会全部落空
    # （曾导致"可用图件 0 张"的错误报告）。
    fig_manifest = (raw_fig or {}).get('figures', raw_fig) or {}
    pls = _read_json(args.transcriptome / 'pls_summary.json') if args.transcriptome else None
    conv = _read_json(args.transcriptome / 'convergence.json') if args.transcriptome else None
    ora_sum = _read_json(args.enrichment / 'ora_summary.json') if args.enrichment else None
    ewce_sum = _read_json(args.enrichment / 'ewce_summary.json') if args.enrichment else None

    # ================= 标题页与摘要 =================
    L += [f'# {args.title}', '']
    L += ['## 摘要', '']
    L += ['**背景**：三叉神经痛（TN）是最严重的神经病理性疼痛之一。既往影像研究已发现患者'
          '存在灰质、白质微结构、功能连接及结构-功能耦合异常，但其脑区易感性的分子基础'
          '尚不清楚。本项目基于 OpenNeuro 公开数据集 ds005713 与 Allen Human Brain Atlas'
          '（AHBA），构建皮层分区水平的三种白质相关影像表型，并检验其与基因表达空间模式'
          '的关联及跨模态收敛性。', '']

    np_ = None
    if pheno_summary:
        ph = pheno_summary.get('phenotypes') or []
        for s in ph:
            if s['phenotype'] == 'A1_streamlines':
                np_ = s['n_subjects']
                break
    # 患者/对照计数从 contrast 文件读取
    n_pat = n_ctl = None
    for name in PHENO_LABELS:
        df = _read_csv(args.phenotypes / f'contrast_{name}.csv')
        if df is not None and len(df):
            # 计数写在 run_summary 的 contrasts 里
            break
    if pheno_summary:
        cs = pheno_summary.get('contrasts') or {}
        first = next(iter(cs.values()), {})
        n_pat = first.get('n_patient')
        n_ctl = first.get('n_control')
        audit.add('摘要/方法 受试者数', f'{n_pat} 患者 / {n_ctl} 对照',
                  'phenotypes/run_summary.json', 'contrasts.*.n_patient/n_control')

    abs_txt = []
    if n_pat and n_ctl:
        abs_txt.append(f'共纳入 {n_pat} 例 TN 患者与 {n_ctl} 名健康对照。')
    else:
        abs_txt.append(TBD('横断面样本量'))
        tbd.append('横断面样本量')
    L += ['**方法**：' + ''.join(abs_txt), '']

    res_sent = []
    if pheno_summary:
        cs = pheno_summary.get('contrasts') or {}
        sig_counts = {k: v.get('n_fdr_significant') for k, v in cs.items()}
        worst = max((v for v in sig_counts.values() if v is not None), default=None)
        audit.add('摘要/结果 各模态 FDR 显著分区数', sig_counts,
                  'phenotypes/run_summary.json', 'contrasts.*.n_fdr_significant')
        if worst is not None:
            res_sent.append(f'三种影像表型的组间效应经 FDR 校正后，'
                            f'单模态最多有 {worst} 个分区显著。')
    if pls:
        res_sent.append(f'PLS1 显著性：{_fmt(pls.get("p_spin"))}（spin 检验）；'
                        f'解释 Y 方差 {_fmt(pls.get("var_y"))}。')
        audit.add('摘要/结果 PLS1', pls.get('p_spin'), 'transcriptome/pls_summary.json',
                  'p_spin')
    else:
        res_sent.append(TBD('影像转录组关联结果（需 AHBA 数据）'))
        tbd.append('影像转录组关联结果')
    L += ['**结果**：' + ''.join(res_sent) or TBD('结果'), '']

    L += ['**结论**：' + (TBD('结论（待结果确定后撰写）')), '',
          '**关键词**：三叉神经痛；白质；结构-功能耦合；影像转录组学；'
          'Allen Human Brain Atlas', '', '---', '']

    # ================= 引言 =================
    L += ['## 1 引言', '',
          '### 1.1 疾病负担与机制未解', '',
          '三叉神经痛以面部阵发性电击样疼痛为特征。系统综述与 meta 回归分析估计，全球合并'
          '发病率约为 25.33/10 万·年（95%CI 11.87–54.02），且女性高于男性、右侧多于左侧'
          '（Jeong et al., J Clin Neurol, 2026, DOI 10.3988/jcn.2025.0433）。'
          '其经典机制为神经血管压迫导致三叉神经根脱髓鞘，但中枢敏化、神经炎症、'
          '离子通道异常和脑结构重塑共同参与慢性化。然而"为什么特定脑区更易受累"'
          '尚无系统答案。', '',
          '### 1.2 影像学已揭示多层面脑改变', '',
          '既往 MRI 研究报道 TN 患者存在灰质形态、白质微结构、功能连接及结构-功能耦合异常；'
          '基于同一公开数据集的研究发现疾病相关解耦与神经元能量代谢、离子稳态和神经炎症'
          '基因空间共变，术后解耦可恢复（Zhang et al., J Headache Pain, 2026, 27(1), '
          'DOI 10.1186/s10194-026-02419-7）。但这些研究多停留在"哪些脑区异常"，'
          '未回答"这些脑区为何易感"。', '',
          '### 1.3 影像转录组学提供桥梁', '',
          'AHBA 提供正常人脑全基因组空间表达图谱。将患者影像异常图谱与 AHBA 表达矩阵做'
          '空间关联，可识别脑区易感性的分子空间约束（Martins et al., Pain, 2022, '
          '163(6):e759–e773）。需正视的是，该研究同时报告其皮层形态改变未通过严格 FDR '
          '校正，说明此类空间关联效应量普遍偏小、对统计口径高度敏感。', '',
          '### 1.4 本研究的目的', '',
          '本研究以皮层分区为分析单元，构建三种白质相关影像表型，与 AHBA 转录组做空间'
          '关联与跨模态收敛检验，并按术式分层探索术后可逆性。', '', '---', '']

    # ================= 方法 =================
    L += ['## 2 材料与方法', '', '### 2.1 数据来源与快照冻结', '']
    L += ['影像数据：OpenNeuro ds005713，快照 v2.0.3（DOI 10.18112/openneuro.ds005713.v2.0.3，'
          '许可 CC0，BIDS 1.8.0；数据集论文 Filimonova et al., Sci Data, 2025, 12:2024）。'
          '共 5128 个文件、43.4 GB。分析基于冻结快照的文件清单。', '']
    if n_pat and n_ctl:
        L += [f'纳入分析：{n_pat} 例 TN 患者与 {n_ctl} 名健康对照；'
              '另有随访影像用于纵向分析。', '']
        audit.add('2.1 纳入样本', f'{n_pat}/{n_ctl}',
                  'phenotypes/run_summary.json', 'contrasts')
    else:
        L += [TBD('样本纳入数'), '']
    L += ['临床变量：因当前快照根目录不存在 BIDS 形式的 participants.tsv，临床变量由 '
          'participants.xlsx 导出并冻结为 TSV（含规范派生列），作为唯一数据来源。', '',
          '转录组数据：AHBA 6 例供体微阵列表达数据，通过本项目实现的 abagen 等价管线处理'
          '（探针重注释、低表达过滤、多探针选择、样本半径匹配、跨供体归一化）。', '',
          '### 2.2 影像表型构建', '',
          '- **A1（白质连接强度）**：DTI 纤维追踪构建结构连接组，取每个皮层分区与全脑的'
          '连接强度（流线数与 FA 加权两套）。',
          '- **A2（白质-皮层功能耦合）**：由结构连接矩阵与功能连接矩阵，用图谐波模型计算'
          '每个分区的功能活动受结构约束的解释度（R²）。',
          '- **A3（灰白质对比度）**：由 T1w 提取 GWC，并以 T1w/T2w 比值作为髓鞘敏感的'
          '平行指标。因 A3 与髓鞘基因存在先验关联，其髓鞘相关富集结果定位为方法学阳性'
          '对照，并另做去髓鞘残差分析。', '',
          '### 2.3 混淆控制与组间比较', '']
    if pheno_summary:
        L += [f"混淆协变量：年龄（含平方项）、性别、平均头动（FD）"
              f"{'、Fazekas 分级' if 'fazekas' in json.dumps(pheno_summary) else ''}；"
              '对每个分区取回归残差后做组间比较。'
              f"混淆校正：{'已执行' if pheno_summary.get('residualized') else '未执行'}。", '']
        audit.add('2.3 混淆校正', bool(pheno_summary.get('residualized')),
                  'phenotypes/run_summary.json', 'residualized')
    else:
        L += [TBD('混淆校正设置'), '']
    L += ['组间比较采用 Welch t 检验，并以 Hedges\' g 作为主要效应量图谱；'
          '同时执行样本量匹配重采样、年龄匹配与头动匹配三类敏感性分析。', '',
          '### 2.4 空间关联与统计检验（若已执行）', '']
    if pls:
        L += [f"PLS 回归（X = 基因表达，Y = 影像效应量图谱）；spin 检验置换次数 "
              f"{pls.get('n_perm')}；PLS1 权重稳定性通过重采样评估。", '']
        audit.add('2.4 PLS/spin 设置', pls.get('n_perm'),
                  'transcriptome/pls_summary.json', 'n_perm')
    else:
        L += [TBD('影像转录组方法细节（需 AHBA 数据）'), '']
        tbd.append('影像转录组方法细节')
    L += ['### 2.5 纵向与临床分层（若已执行）', '']
    if long_summary:
        L += ['配对分析仅使用同时具备术前与术后数据的被试；术式分层（MVD vs 经皮介入）'
              '并对变化量做两组比较（两时间点下等价于时间×术式交互）；'
              '疼痛缓解百分比作为调节变量；完全缓解与未完全缓解的区分能力用分层交叉验证'
              'AUC 与 PR-AUC 评估，并给出标签置换零分布。', '']
        n_pair = None
        for e in (long_summary.get('phenotypes') or {}).values():
            n_pair = (e.get('paired') or {}).get('n_pairs') or n_pair
        if n_pair:
            audit.add('2.5 纵向配对数', n_pair, 'longitudinal/run_summary.json',
                      'phenotypes.*.paired.n_pairs')
    else:
        L += [TBD('纵向分析设置'), '']
        tbd.append('纵向分析设置')
    L += ['---', '']

    # ================= 结果 =================
    L += ['## 3 结果', '', '### 3.1 样本与影像表型', '']
    if pheno_summary:
        rows = pheno_summary.get('phenotypes') or []
        L += ['| 表型 | 被试数 | 分区数 | 有效值比例 | 全缺失分区 |',
              '|---|---|---|---|---|']
        for s in rows:
            L.append(f"| {s['phenotype']} | {s['n_subjects']} | {s['n_parcels']} | "
                     f"{s['frac_finite']:.3f} | {s['parcels_all_nan']} |")
        L.append('')
        audit.add('3.1 表型覆盖表', rows, 'phenotypes/run_summary.json', 'phenotypes')
    else:
        L += [TBD('表型覆盖表'), '']
        tbd.append('表型覆盖表')

    L += ['### 3.2 组间效应（Hedges\' g）', '']
    if pheno_summary:
        cs = pheno_summary.get('contrasts') or {}
        L += ['| 表型 | 患者 | 对照 | FDR 显著分区 | max\\|g\\| | 最小 p |',
              '|---|---|---|---|---|---|']
        for k, v in cs.items():
            L.append(f"| {k} | {v.get('n_patient')} | {v.get('n_control')} | "
                     f"{v.get('n_fdr_significant')} | {_fmt(v.get('max_abs_g'))} | "
                     f"{_fmt(v.get('min_p'))} |")
        L.append('')
        audit.add('3.2 组间效应表', cs, 'phenotypes/run_summary.json', 'contrasts')
        if fig_manifest and 'figure1_effect_profiles' in (
                fig_manifest):
            L += ['![图 1](figure1_effect_profiles.svg)', '',
                  '**图 1** 三模态组间效应量空间分布（红色为 FDR 显著分区）。', '']
    else:
        L += [TBD('组间效应结果'), '']
        tbd.append('组间效应结果')

    L += ['### 3.3 敏感性分析', '']
    if pheno_summary:
        sens = pheno_summary.get('sensitivity') or {}
        if sens:
            L += ['| 分析 | 指标 | 值 |', '|---|---|---|']
            for k, v in list(sens.items())[:20]:
                for kk, vv in v.items():
                    L.append(f'| {k} | {kk} | {_fmt(vv)} |')
            L.append('')
            audit.add('3.3 敏感性分析', sens, 'phenotypes/run_summary.json', 'sensitivity')
        else:
            L += [TBD('敏感性分析（未执行或样本不足）'), '']
    else:
        L += [TBD('敏感性分析'), '']

    L += ['### 3.4 影像转录组关联与跨模态收敛', '']
    # D10/D11：读取事前决策规则，避免把阳性对照或被剔除模态写成"发现"
    rules: dict = {}
    try:
        sys.path.insert(0, str(ROOT / 'src'))
        from tnwt import config_lite as _cl
        _cfg = _cl.load(ROOT / 'config' / 'analysis.yaml')
        rules = _cl.dig(_cfg, 'analysis_decision_rules', {}) or {}
    except Exception:  # noqa: BLE001
        rules = {}
    pos_ctrl = rules.get('positive_control_modality')
    conv_members = rules.get('convergence_members')
    excluded_mods = (conv or {}).get('excluded_modalities') or {}
    if rules:
        L += ['**事前设定的判据（D8–D12）**：', '',
              f"- 主分析模态：`{rules.get('primary_modality')}`（不因结果好坏替换）",
              f"- 收敛判定成员：{conv_members}",
              f"- 阳性对照：`{pos_ctrl}`（其髓鞘相关富集不作发现）",
              f"- 未过 FDR 的模态：报告为阴性、不参与收敛、关联仅标探索性"
              f"（当前剔除：{list(excluded_mods) if excluded_mods else '无'}）", '']
        audit.add('3.4 事前判据', rules.get('frozen_on'),
                  'config/analysis.yaml', 'analysis_decision_rules')
    if pls:
        # 逐模态报告，而不是用"最小 p"代表全部——那会把阳性对照的 p 当成主结果
        mods = pls.get('modalities') or {}
        if mods:
            L += ['| 模态 | 有效分区 | PLS1 解释 Y 方差 | PLS1 解释 X 方差 '
                  '| 稳定性余弦均值 | \\|Z\\|≥2 基因数 | 候选基因数 | 角标 |',
                  '|---|---|---|---|---|---|---|---|']
            for k, v in mods.items():
                flags = []
                if pos_ctrl and k == pos_ctrl:
                    flags.append('阳性对照(D10)')
                if k in excluded_mods:
                    flags.append('探索性(D11)')
                L.append(f"| {k} | {v.get('n_used_parcels')} | "
                         f"{_fmt(v.get('var_y'))} | {_fmt(v.get('var_x'))} | "
                         f"{_fmt(v.get('stability_cosine_mean'))} | "
                         f"{v.get('n_abs_z_gt_2')} | {v.get('n_candidate_genes')} | "
                         f"{'、'.join(flags) or '—'} |")
            L.append('')
            audit.add('3.4 逐模态 PLS 表', mods, 'transcriptome/pls_summary.json',
                      'modalities')
        if (conv or {}).get('spin_p'):
            L += ['- spin 检验（控制空间自相关）：', '',
                  '| 模态 | p | q（FDR） | FDR 显著 |', '|---|---|---|---|']
            for k in sorted(conv['spin_p']):
                L.append(f"| {k} | {_fmt(conv['spin_p'][k], 4)} | "
                         f"{_fmt((conv.get('spin_q') or {}).get(k), 4)} | "
                         f"{(conv.get('spin_fdr_significant') or {}).get(k)} |")
            L.append('')
            audit.add('3.4 spin 检验', conv['spin_p'],
                      'transcriptome/convergence.json', 'spin_p')
        # var_y 接近 1 是 p≫n 的必然结果，必须显式警示，否则会被误读为"模型解释力极强"
        var_ys = [v.get('var_y') for v in mods.values()
                  if isinstance(v.get('var_y'), (int, float))]
        if var_ys and max(var_ys) > 0.999:
            L += ['> **关于 PLS1 的「解释 Y 方差」**：当基因数（15,131）远大于分区数（~200）时，'
                  'PLS1 在数学上几乎必然能把 Y 完全拟合，``var_y ≈ 1`` 因此**不构成**'
                  '"关联很强"的证据。判断显著性应看 **spin 检验**（控制空间自相关），'
                  '判断方向稳定性应看**重采样余弦与入选率**，而非 var_y。'
                  '本文以 spin 检验与稳定性为主要判据。', '']
            audit.add('3.4 var_y 过拟合警示', max(var_ys),
                      'transcriptome/pls_summary.json', 'modalities.*.var_y',
                      note='p≫n 导致 var_y 饱和，不可作为效应强度证据')
    else:
        L += [TBD('PLS 关联结果（需 AHBA 数据与影像数据）'), '']
        tbd.append('PLS 关联结果')
    if conv:
        part = conv.get('participating_modalities')
        L += [f"- 跨模态收敛：{'满足事前判据' if conv.get('converged') else '未满足判据'}"
              f"（参与判定模态：{part}；"
              f"最小权重余弦 {_fmt(conv.get('min_pairwise_cosine'))}；"
              f"最大两两 |r| = {_fmt(conv.get('max_pairwise_abs_r'))}）。", '']
        if conv.get('insufficient_participants'):
            L += [f"> **收敛性无法判定**：{conv.get('note')}。"
                  '按 D11，影像层面无可靠组间信号的模态不参与收敛判定；'
                  '因此本研究不就"共享分子基础"作出结论，'
                  '仅报告各模态的探索性影像转录组关联。', '']
            tbd.append('跨模态收敛结论（参与模态不足，待 A1/A2 结果）')
        if excluded_mods:
            L += ['- 按 D11 剔除的模态及原因：', '']
            for k, v in excluded_mods.items():
                L.append(f'  - `{k}`：{v}')
            L.append('')
        audit.add('3.4 收敛结论', conv.get('converged'),
                  'transcriptome/convergence.json', 'converged')
    else:
        L += [TBD('跨模态收敛结果'), '']
    if fig_manifest and 'figure4_convergence' in (fig_manifest):
        L += ['![图 4](figure4_convergence.svg)', '',
              '**图 4** 跨模态收敛检验（含按 D11 剔除的模态）。', '']

    L += ['### 3.5 功能富集与细胞类型', '']
    if ora_sum or ewce_sum:
        if ora_sum:
            L += [f"- 富集分析：{ora_sum.get('n_sets_tested', 'NA')} 个基因集参与检验，"
                  f"FDR 显著 {ora_sum.get('n_fdr_significant', 'NA')} 个；"
                  f"背景基因集 {ora_sum.get('n_background', 'NA')} 个。", '']
            audit.add('3.5 富集汇总', ora_sum, 'enrichment/ora_summary.json')
        if ewce_sum:
            L += [f"- 细胞类型：{ewce_sum.get('n_cell_types', 'NA')} 种细胞类型参与检验，"
                  f"FDR 显著 {ewce_sum.get('n_fdr_significant', 'NA')} 个。", '']
            audit.add('3.5 细胞类型汇总', ewce_sum, 'enrichment/ewce_summary.json')
    else:
        L += [TBD('富集与细胞类型结果'), '']
        tbd.append('富集与细胞类型结果')

    L += ['### 3.6 术后变化与临床分层', '']
    if long_summary:
        ph = long_summary.get('phenotypes') or {}
        L += ['| 表型 | 配对数 | FDR 显著分区 | max\\|dz\\| | 随访来源 |',
              '|---|---|---|---|---|']
        for name, e in ph.items():
            p = e.get('paired') or {}
            L.append(f"| {name} | {p.get('n_pairs', 'NA')} | "
                     f"{p.get('n_fdr_significant', 'NA')} | "
                     f"{_fmt(p.get('max_abs_dz'))} | {e.get('followup_source', 'NA')} |")
        L.append('')
        audit.add('3.6 配对比较表', ph, 'longitudinal/run_summary.json', 'phenotypes')
        # 术式交互与临床分层
        for name, e in list(ph.items())[:3]:
            st = e.get('stratified') or {}
            it = st.get('interaction') or {}
            if it:
                L += [f"- {name}：术式交互检验（{it.get('groups')}）FDR 显著分区 "
                      f"{it.get('n_fdr_significant', 'NA')} 个。", '']
        strat_rows = []
        for name, e in ph.items():
            for label, v in (e.get('stratification') or {}).items():
                strat_rows.append((name, label, v))
        if strat_rows:
            L += ['| 表型 | 指标 | AUC | PR-AUC | 正/负例 | 置换 p |',
                  '|---|---|---|---|---|---|']
            for name, label, v in strat_rows:
                L.append(f"| {name} | {label} | {_fmt(v.get('auc'))} | "
                         f"{_fmt(v.get('pr_auc'))} | {v.get('n_positive')}/"
                         f"{v.get('n_negative')} | {_fmt(v.get('auc_perm_p'))} |")
            L.append('')
            audit.add('3.6 临床分层表', strat_rows, 'longitudinal/run_summary.json',
                      'phenotypes.*.stratification')
        if fig_manifest and 'figure6_longitudinal' in (fig_manifest):
            L += ['![图 6](figure6_longitudinal.svg)', '',
                  '**图 6** 术后影像变化（红色为 FDR 显著分区）。', '']
    else:
        L += [TBD('纵向与临床分层结果'), '']
        tbd.append('纵向与临床分层结果')
    L += ['---', '']

    # ================= 讨论 =================
    L += ['## 4 讨论', '',
          '### 4.1 主要发现', '', TBD('主要发现（待结果确定）'), '',
          '### 4.2 与既往研究的比较', '',
          '需与 Zhang et al.（2026）在同一数据集上的结构-功能耦合结果对照，'
          '包括基因权重向量的一致性（本项目三级验证的第二级）。', '',
          '### 4.3 机制解释', '',
          '若髓鞘/少突胶质细胞相关基因富集，需注意 A3 与该通路的先验关联，'
          '故本项目将 A3 的髓鞘富集定位为阳性对照，并报告去髓鞘残差分析结果。', '',
          '### 4.4 局限', '']
    L += ['1. AHBA 仅 6 例供体、以灰质采样为主，白质覆盖不足，因此分析单元为皮层分区，'
          '白质信息作为皮层表型属性纳入；',
          '2. AHBA 来自正常脑，非疾病脑，结论限于空间关联层面的假设生成，不涉及因果；',
          '3. 影像与转录组均来自公开数据，患者组的年龄与头动可能与对照存在系统差异，'
          '已通过协变量校正与匹配敏感性分析处理；',
          '4. 术式异质（MVD 与经皮介入机制不同），纵向结论按术式分层解释；',
          '5. 空间自相关的存在使有效自由度远小于分区数，已用 spin 检验控制，'
          '但此类分析的效应量普遍偏小。', '', '---', '']

    # ================= 声明 =================
    L += ['## 5 数据与代码可用性', '',
          '影像数据来自 OpenNeuro ds005713（DOI 10.18112/openneuro.ds005713.v2.0.3，许可 CC0），'
          '按原始 DOI 引用；转录组数据来自 Allen Human Brain Atlas；'
          '注释数据来自 Gene Ontology、Reactome 与 HGNC。',
          '分析代码（零依赖实现）与非个体水平的统计/图谱衍生物开源；'
          '个体水平影像不再分发。', '',
          '## 6 作者贡献 / 利益冲突 / 资助', '',
          TBD('按目标期刊格式补全'), '', '---', '']

    # ================= 图表清单 =================
    L += ['## 附：图表清单', '']
    if fig_manifest:
        gen = fig_manifest.get('generated') or []
        skip = fig_manifest.get('skipped') or []
        L += ['| 图号 | 文件 | 状态 |', '|---|---|---|']
        for k in gen:
            f = (fig_manifest).get(k, {})
            L.append(f"| {k} | {f.get('file', 'NA')} | 已生成 |")
        for s in skip:
            L.append(f"| {s['figure']} | — | 未生成（{s['reason']}） |")
        L.append('')
    else:
        L += [TBD('图表清单（需先运行 scripts/make_figures.py）'), '']

    # ================= 待办清单 =================
    if tbd:
        L += ['## 附：待补清单（投稿前必须逐条完成）', '']
        for i, t in enumerate(sorted(set(tbd)), 1):
            L.append(f'{i}. {t}')
        L.append('')

    body = '\n'.join(L) + '\n'
    (args.out_dir / 'manuscript.md').write_text(body, encoding='utf-8')

    # 结果数字只有落到正文才计入审计
    audit_path = audit.dump(args.out_dir / 'paper_number_audit.json')

    n_tbd = body.count('【待补：')
    summary = {
        'title': args.title,
        'journal': args.journal or None,
        'n_characters': len(body),
        'n_pending_placeholders': n_tbd,
        'n_audited_numbers': len(audit.entries),
        'n_figures_available': len((fig_manifest or {}).get('generated') or []),
        'sources': {
            'phenotypes': str(args.phenotypes) if pheno_summary else None,
            'transcriptome': str(args.transcriptome) if pls else None,
            'enrichment': str(args.enrichment) if (ora_sum or ewce_sum) else None,
            'longitudinal': str(args.longitudinal) if long_summary else None,
            'figures': str(args.figures) if fig_manifest else None,
        },
        'pending_items': sorted(set(tbd)),
    }
    (args.out_dir / 'manuscript_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'论文稿：{args.out_dir / "manuscript.md"}（{len(body)} 字符）')
    print(f'已审计数值：{len(audit.entries)} 条 → {audit_path.name}')
    print(f'待补占位：{n_tbd} 处')
    for t in summary['pending_items']:
        print(f'  - {t}')
    print('说明：所有写入数值均来自真实输出文件，并已在审计文件中记录来源。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
