#!/usr/bin/env python -W ignore
# -*- coding: utf-8 -*-
"""投稿包生成（阶段十二）。

把投稿所需的各项材料汇总为一个目录，并把**需要人工填写的空位**明确列出，
避免临投时才发现缺件。

产出（``<out-dir>/``）
--------------------
- ``submission_checklist.md``   投稿清单（含期刊要求条目与"待人工填写"标记）
- ``cover_letter_template.md``  投稿信模板（含真实统计数字的占位符说明）
- ``declarations.md``           声明集合：伦理、数据可用性、代码可用性、利益冲突、
                                作者贡献、资助（按常见期刊结构，可按目标期刊调整）
- ``figure_legends.md``         图注（从 `figure_manifest.json` 生成）
- ``table_inventory.md``        表清单（从各阶段 CSV 汇总行数列数）
- ``reproducibility.md``        可复现性声明（复制自 provenance 产物）
- ``package_manifest.json``     包内文件清单 + 哪些环节尚未就绪

设计原则
--------
- **只汇总真实存在的产物**；缺失环节在清单中标为 `未就绪`，不生成占位内容。
- 不覆盖任何既有文件；输出目录独立。
- 期刊差异通过 ``--journal`` 控制；未指定时用通用结构并在清单中提示需核对期刊要求。
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 常见目标期刊的格式要点（仅作提示，投稿前必须核对期刊最新 Author Guidelines）
JOURNAL_PROFILES = {
    'pain': {
        'name': 'Pain',
        'abstract_words': 250,
        'structured_abstract': True,
        'abstract_sections': ['Abstract'],
        'main_text_words': 5000,
        'figures_max': 8,
        'tables_max': 6,
        'references_max': 60,
        'highlights_required': False,
        'cover_letter_required': True,
    },
    'jhp': {
        'name': 'The Journal of Headache and Pain',
        'abstract_words': 350,
        'structured_abstract': True,
        'abstract_sections': ['Background', 'Methods', 'Results', 'Conclusions'],
        'main_text_words': None,
        'figures_max': None,
        'tables_max': None,
        'references_max': None,
        'highlights_required': False,
        'cover_letter_required': True,
    },
    'hbm': {
        'name': 'Human Brain Mapping',
        'abstract_words': 250,
        'structured_abstract': False,
        'abstract_sections': ['Abstract'],
        'main_text_words': None,
        'figures_max': None,
        'tables_max': None,
        'references_max': None,
        'highlights_required': False,
        'cover_letter_required': True,
    },
    'tp': {
        'name': 'Translational Psychiatry',
        'abstract_words': 250,
        'structured_abstract': True,
        'abstract_sections': ['Abstract'],
        'main_text_words': None,
        'figures_max': None,
        'tables_max': None,
        'references_max': None,
        'highlights_required': False,
        'cover_letter_required': True,
    },
}

# 投稿清单条目：(类别, 条目, 判定依据文件或说明)
CHECKLIST = [
    ('正文', '论文稿 manuscript.md', 'paper/manuscript.md'),
    ('正文', '摘要符合期刊字数与结构要求', '需人工核对'),
    ('图表', '主图（矢量 SVG）', 'figures/figure*.svg'),
    ('图表', '图注 figure_legends.md', 'submission/figure_legends.md'),
    ('图表', '表清单 table_inventory.md', 'submission/table_inventory.md'),
    ('声明', '伦理与知情同意声明', 'submission/declarations.md'),
    ('声明', '数据可用性声明', 'submission/declarations.md'),
    ('声明', '代码可用性声明', 'submission/declarations.md'),
    ('声明', '利益冲突声明', 'submission/declarations.md'),
    ('声明', '作者贡献声明', '需人工填写'),
    ('声明', '资助信息', '需人工填写'),
    ('可复现性', '可复现性声明（环境/依赖/输入哈希/代码指纹）', 'submission/reproducibility.md'),
    ('投稿信', 'Cover letter', 'submission/cover_letter_template.md'),
    ('附件', '补充材料（敏感性分析、稳健性、LODO 等）', 'data/*/sensitivity_*, robustness.json'),
    ('合规', '数据使用遵循原数据许可（CC0）', '需人工确认'),
    ('合规', '伦理备案/豁免确认件', '需人工上传'),
    ('合规', '所有作者的最终稿确认', '需人工确认'),
]


def _exists(p: Path) -> bool:
    return p.exists() and (p.is_file() or any(p.iterdir()))


def _read_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return None


def build_figure_legends(figures_dir: Path) -> tuple[str, list[str]]:
    """从图件清单生成图注骨架（图注正文需人工撰写，此处给出编号、文件与面板信息）。"""
    manifest = _read_json(figures_dir / 'figure_manifest.json') or {}
    inner = manifest.get('figures', manifest)
    titles = {
        'figure1_effect_profiles': '三模态组间效应量空间分布',
        'figure2_collinearity': '共线性诊断与收敛前置检验',
        'figure3_pls': 'PLS1 基因权重分布与稳定性',
        'figure4_convergence': '跨模态收敛检验',
        'figure5_enrichment': '功能富集与细胞类型',
        'figure6_longitudinal': '术后影像变化',
        'figure7_subject_flow': '受试者纳入流程',
    }
    lines = ['# 图注（Figure legends）', '',
             '> 说明：编号、文件名与面板信息由 `figure_manifest.json` 自动生成；'
             '**图注正文需人工撰写**（每图 1–2 句，注明统计量与 n）。', '']
    gen = inner.get('generated') or []
    files: list[str] = []
    for i, key in enumerate(gen, 1):
        f = (inner.get('figures') or {}).get(key, {})
        fname = f.get('file') or f'{key}.svg'
        files.append(fname)
        lines += [f'**Figure {i}. {titles.get(key, key)}**（`{fname}`）', '',
                  '- 面板：见图中 A/B/… 标注',
                  '- 需补充：n、统计检验、效应量与多重比较校正方式',
                  '- 需补充：显著性标记的含义', '']
    skipped = inner.get('skipped') or []
    if skipped:
        lines += ['## 未生成的图（输入缺失）', '']
        for s in skipped:
            lines.append(f"- {s.get('figure')}：{s.get('reason')}")
        lines.append('')
    return '\n'.join(lines) + '\n', files


def build_table_inventory(stage_dirs: dict[str, Path]) -> str:
    """汇总各阶段 CSV 的行列数，作为表清单的基础。"""
    import csv as _csv
    lines = ['# 表清单（Table inventory）', '',
             '> 说明：行数/列数由各阶段 CSV 自动统计；**表题与脚注需人工撰写**。', '',
             '| 表 | 来源文件 | 行 | 列 |', '|---|---|---|---|']
    n = 0
    for stage, d in stage_dirs.items():
        if not d or not d.exists():
            continue
        for f in sorted(d.glob('*.csv')):
            try:
                with f.open('r', encoding='utf-8', newline='') as fh:
                    reader = _csv.reader(fh)
                    header = next(reader, [])
                    rows = sum(1 for _ in reader)
            except Exception:  # noqa: BLE001
                continue
            n += 1
            lines.append(f'| Table {n} | `{stage}/{f.name}` | {rows} | {len(header)} |')
    if n == 0:
        lines += ['', '**当前无可汇总的结果表**（各阶段数据尚未生成）。']
    return '\n'.join(lines) + '\n'


def build_declarations(provenance: dict | None, manuscripts: dict | None) -> str:
    code_fp = ((provenance or {}).get('code') or {}).get('code_fingerprint', '未生成')
    n_src = ((provenance or {}).get('code') or {}).get('n_source_files', '未知')
    n_ready = sum(1 for v in ((provenance or {}).get('inputs') or {}).values()
                  if v.get('exists'))
    n_in = len(((provenance or {}).get('inputs') or {}))
    L = ['# 声明集合（Declarations）', '',
         '> 以下文本按常见期刊结构准备；**方括号内容需人工确认或替换**。', '']
    L += ['## 伦理声明（Ethics approval）', '',
          '本研究为公开数据二次分析。影像数据来自 OpenNeuro 数据集 ds005713，'
          '其伦理批准由原研究单位伦理委员会给出（2021-05-25，协议 #7），'
          '全部受试者签署知情同意。本项目不涉及新受试者招募。'
          '**本项目已向 [单位名称] 伦理委员会提交二次分析备案，'
          '备案/豁免编号：[编号]，日期：[日期]。**', '']
    L += ['## 数据可用性（Data availability）', '',
          '- 影像数据：OpenNeuro ds005713，快照 v2.0.3'
          '（DOI 10.18112/openneuro.ds005713.v2.0.3），许可 CC0；按原始 DOI 引用，'
          '本项目不再分发个体水平影像。',
          '- 转录组数据：Allen Human Brain Atlas（公开）。',
          '- 注释数据：Gene Ontology、Reactome、HGNC（公开）。',
          '- 本项目产生的**非个体水平**衍生物（分区级统计图谱、效应量矩阵）'
          '与全部分析代码将随论文公开：[仓库/DOI 待填]。', '']
    L += ['## 代码可用性（Code availability）', '',
          f'- 分析代码：Python 3.12 实现；源码文件 {n_src} 个；'
          f'代码版本指纹 `{code_fp}`。',
          f'- 可复现性清单记录了全部关键输入的 sha256（当前 {n_ready}/{n_in} 就绪），'
          '见 `docs/provenance.json`。',
          '- 未使用 scipy / statsmodels / scikit-learn / nibabel 等依赖：'
          '关键统计函数（Welch t、卡方、不完全 Beta/Gamma、PLS、'
          '球面旋转置换、超几何检验）以零依赖方式自实现，'
          '并用已知值或文献值交叉验证（见 `docs/methodological_notes.md`）。', '']
    L += ['## 利益冲突（Competing interests）', '',
          '作者声明无利益冲突。[如存在需据实填写]', '']
    L += ['## 作者贡献（Author contributions）', '',
          '[按 CRediT 分类填写：Conceptualization / Methodology / Software / '
          'Formal analysis / Data curation / Writing – original draft / '
          'Writing – review & editing / Supervision / Funding acquisition]', '']
    L += ['## 资助（Funding）', '', '[资助项目名称与编号；无资助需说明]', '']
    L += ['## 致谢（Acknowledgements）', '',
          '感谢 OpenNeuro ds005713 数据集作者与 Allen Human Brain Atlas 的开放数据共享。'
          '[其他致谢待填]', '']
    return '\n'.join(L) + '\n'


def build_cover_letter(profile: dict | None, manuscripts: dict | None) -> str:
    journal = (profile or {}).get('name', '[目标期刊名称]')
    L = [f'# 投稿信（Cover letter）—— 致 {journal} 编辑', '',
         '> 方括号内容需人工填写。', '',
         '尊敬的编辑：', '',
         '我们谨向贵刊提交题为《[论文题目]》的原创研究论文，供审阅。', '',
         '## 研究问题与主要发现', '',
         '[用 2–3 句概括研究问题与主要发现，数字须与正文一致]', '',
         '## 与既往工作的差异', '',
         '既往基于同一公开数据集的研究聚焦于中枢结构-功能耦合的转录组关联；'
         '本研究以皮层分区为分析单元，构建三种白质相关影像表型，'
         '并把跨模态收敛性作为**事前设定判据**的检验对象，'
         '同时按术式分层探索术后可逆性。', '',
         '## 方法学严谨性', '',
         '- 组间比较采用效应量（Hedges\' g）而非仅 t 值，并做混淆校正'
         '（年龄含平方项、性别、头动、Fazekas 分级）；',
         '- 空间关联使用球形旋转置换（spin test）控制空间自相关，'
         '并用重采样评估 PLS 权重稳定性；',
         '- 富集分析同时报告表达量/入选概率匹配的置换零模型，避免共表达导致的富集膨胀；',
         '- 独立验证分三级进行（方法学稳健性、与外部权重向量一致性、外部数据库定性支持）。', '',
         '## 声明', '',
         '本文未一稿多投，全体作者已审阅并同意投稿。利益冲突与资助信息见稿件声明部分。', '',
         '此致', '', '敬礼', '',
         '[通讯作者姓名、单位、联系方式、日期]', '']
    if profile:
        L += ['## 拟投期刊格式要点（投稿前请核对最新 Author Guidelines）', '',
              f"- 摘要：{'结构化' if profile.get('structured_abstract') else '非结构化'}，"
              f"≤ {profile.get('abstract_words')} 词",
              f"- 摘要章节：{', '.join(profile.get('abstract_sections') or [])}",
              f"- 正文：≤ {profile.get('main_text_words') or '未限制'} 词",
              f"- 图：≤ {profile.get('figures_max') or '未限制'} 个",
              f"- 表：≤ {profile.get('tables_max') or '未限制'} 个",
              f"- 参考文献：≤ {profile.get('references_max') or '未限制'} 条",
              '- 以上为常见约定，**务必以期刊官网最新要求为准**。', '']
    return '\n'.join(L) + '\n'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='生成投稿包')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'submission')
    ap.add_argument('--paper', type=Path, default=ROOT / 'paper')
    ap.add_argument('--figures', type=Path, default=ROOT / 'figures')
    ap.add_argument('--provenance', type=Path, default=ROOT / 'docs' / 'provenance.json')
    ap.add_argument('--journal', default='', help='目标期刊（pain/jhp/hbm/tp 或自填名称）')
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    profile = JOURNAL_PROFILES.get(args.journal.strip().lower())
    if args.journal and not profile:
        profile = {'name': args.journal}

    provenance = _read_json(args.provenance)
    ms_summary = _read_json(args.paper / 'manuscript_summary.json')
    audit = _read_json(args.paper / 'paper_number_audit.json')

    # ---- 图注 ----
    legends, fig_files = build_figure_legends(args.figures)
    (args.out_dir / 'figure_legends.md').write_text(legends, encoding='utf-8')

    # ---- 表清单 ----
    stage_dirs = {
        'phenotypes': ROOT / 'data' / 'phenotypes',
        'longitudinal': ROOT / 'data' / 'longitudinal',
        'transcriptome': ROOT / 'data' / 'transcriptome',
        'enrichment': ROOT / 'data' / 'enrichment',
    }
    (args.out_dir / 'table_inventory.md').write_text(
        build_table_inventory(stage_dirs), encoding='utf-8')

    # ---- 声明 ----
    (args.out_dir / 'declarations.md').write_text(
        build_declarations(provenance, ms_summary), encoding='utf-8')

    # ---- 投稿信 ----
    (args.out_dir / 'cover_letter_template.md').write_text(
        build_cover_letter(profile, ms_summary), encoding='utf-8')

    # ---- 复制可复现性声明 ----
    repro_src = args.provenance.with_name('reproducibility.md')
    copied: list[str] = []
    if repro_src.exists():
        shutil.copy2(repro_src, args.out_dir / 'reproducibility.md')
        copied.append('reproducibility.md')

    # ---- 投稿清单 ----
    # 状态判定必须基于**本次调用传入的目录**，而不是仓库默认路径；
    # 否则用非默认目录（例如集成验证产物）时会全部误判为"未就绪"。
    stage_map = {'phenotypes': stage_dirs['phenotypes'],
                 'longitudinal': stage_dirs['longitudinal'],
                 'transcriptome': stage_dirs['transcriptome'],
                 'enrichment': stage_dirs['enrichment']}

    def resolve_status(ref: str) -> str:
        if ref.startswith('需人工'):
            return '需人工'
        # 把清单里的仓库相对路径重定向到传入目录
        for key, d in stage_map.items():
            if ref.startswith(f'data/{key}/'):
                sub = ref[len(f'data/{key}/'):]
                if '*' in sub:
                    return '已生成' if list(d.glob(sub)) else '未就绪'
                return '已生成' if (d / sub).exists() else '未就绪'
        if ref.startswith('paper/'):
            sub = ref[len('paper/'):]
            if '*' in sub:
                return '已生成' if list(args.paper.glob(sub)) else '未就绪'
            return '已生成' if (args.paper / sub).exists() else '未就绪'
        if ref.startswith('figures/'):
            sub = ref[len('figures/'):]
            if '*' in sub:
                return '已生成' if list(args.figures.glob(sub)) else '未就绪'
            return '已生成' if (args.figures / sub).exists() else '未就绪'
        if ref.startswith('submission/'):
            return '已生成'
        p = ROOT / ref
        if '*' in ref:
            return '已生成' if list(ROOT.glob(ref)) else '未就绪'
        return '已生成' if _exists(p) else '未就绪'

    rows = [(c, i, r, resolve_status(r)) for c, i, r in CHECKLIST]

    n_ready = sum(1 for *_, s in rows if s == '已生成')
    n_todo = sum(1 for *_, s in rows if s == '需人工')
    n_missing = sum(1 for *_, s in rows if s == '未就绪')

    L = ['# 投稿清单（Submission checklist）', '',
         f"生成时间：{time.strftime('%Y-%m-%dT%H:%M:%S')}",
         f"目标期刊：{(profile or {}).get('name', '[未指定]')}", '',
         f"- 已生成：{n_ready} 项；未就绪（缺数据）：{n_missing} 项；需人工：{n_todo} 项", '',
         '| 类别 | 条目 | 依据 | 状态 |', '|---|---|---|---|']
    for c, i, r, s in rows:
        L.append(f'| {c} | {i} | `{r}` | {s} |')
    if ms_summary:
        L += ['', '## 论文稿状态', '',
              f"- 字符数：{ms_summary.get('n_characters')}",
              f"- 待补占位：{ms_summary.get('n_pending_placeholders')} 处",
              f"- 审计数值：{ms_summary.get('n_audited_numbers')} 条",
              f"- 可用图件：{ms_summary.get('n_figures_available')} 张"]
        pend = ms_summary.get('pending_items') or []
        if pend:
            L += ['', '**尚未填充的环节**：']
            for t in pend:
                L.append(f'- {t}')
    if not (args.paper / 'manuscript.md').exists():
        L += ['', '> ⚠️ 论文稿尚未生成：请先运行 `scripts/write_paper.py`。']
    L += ['', '## 投稿前最后检查', '',
          '1. 逐条核对 `paper/paper_number_audit.json`：正文中每个数字能否在来源文件中找到；',
          '2. 核对摘要字数与结构是否符合目标期刊要求；',
          '3. 核对图注是否包含 n、统计量、校正方式与显著性标记含义；',
          '4. 确认补充材料已包含：敏感性分析、LODO 稳健性、坐标/归一化敏感性；',
          '5. 确认已上传伦理备案件与全体作者确认。', '']
    (args.out_dir / 'submission_checklist.md').write_text('\n'.join(L) + '\n',
                                                         encoding='utf-8')

    manifest = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'journal': (profile or {}).get('name'),
        'files': sorted(p.name for p in args.out_dir.iterdir() if p.is_file()),
        'checklist': {'ready': n_ready, 'missing': n_missing, 'manual': n_todo},
        'figure_files': fig_files,
        'inputs_present': {
            'manuscript': (args.paper / 'manuscript.md').exists(),
            'number_audit': audit is not None,
            'n_audited_numbers': (audit or {}).get('n_numbers'),
            'provenance': provenance is not None,
            'figure_manifest': (args.figures / 'figure_manifest.json').exists(),
        },
    }
    (args.out_dir / 'package_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=float),
        encoding='utf-8')

    print(f'投稿包：{args.out_dir}')
    for f in manifest['files']:
        print(f'  - {f}')
    print(f"清单：已生成 {n_ready} / 未就绪 {n_missing} / 需人工 {n_todo}")
    if n_missing:
        print('  未就绪项对应数据尚未产出；填充后重跑本脚本即可更新清单。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
