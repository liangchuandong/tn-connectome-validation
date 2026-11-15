#!/usr/bin/env python -W ignore
# -*- coding: utf-8 -*-
"""功能富集与细胞类型分析（阶段八）运行入口。

输入
----
- ``<transcriptome-dir>/candidate_genes_<modal>.json``  各模态候选基因（阶段六输出）
- ``<transcriptome-dir>/enrichment_input.json``         并集与说明
- ``<transcriptome-dir>/pls_stability_<modal>.csv``     若存在，用作入选概率加权零模型
- ``data/annotations/``                                 GO/Reactome 注释（fetch_annotations.py）
- 可选 ``--cell-type-reference``                        基因 × 细胞类型 表达矩阵（CSV，首列为基因名）
- 可选 ``--cell-proportions``                           分区 × 细胞类型 比例矩阵（CSV）
- 可选 ``--image-map``                                  效应量图谱，用于细胞比例空间相关

输出（``<out-dir>``）
--------------------
- ``ora_<modal>.csv``         超几何 ORA 结果（含 q 值与 FDR 标志）
- ``ora_permuted_<modal>.csv`` 表达量匹配/入选概率加权的置换校准结果
- ``gsea_<modal>.csv``        基因集水平分析
- ``ewce_<modal>.json``       细胞类型富集（若提供参考）
- ``cell_proportion_correlation_<modal>.csv`` 细胞比例空间相关（若提供比例与图谱）
- ``ora_summary.json`` / ``ewce_summary.json``  供论文与出图读取的汇总

背景基因集：**必须**显式提供（方案 5.5 要求为 AHBA 检出基因 ∩ 注释可注释基因）。
若未提供，脚本按"注释库全部基因 ∩ 候选基因所在注释库"退化为默认背景，
并在汇总中标注 ``background_source='fallback_all_annotated'``——这会导致富集膨胀，
不应直接用于论文。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tnwt import config_lite as cl            # noqa: E402
from tnwt import enrichment as en            # noqa: E402


def _load_candidates(tdir: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in sorted(tdir.glob('candidate_genes_*.json')):
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
        except Exception:  # noqa: BLE001
            continue
        modal = obj.get('modal') or p.stem.replace('candidate_genes_', '')
        genes = obj.get('genes') or []
        if genes:
            out[modal] = list(genes)
    if not out:
        # 退化：直接从 pls_weights 按 |Z| 阈值取
        for p in sorted(tdir.glob('pls_weights_*.csv')):
            df = pd.read_csv(p)
            if 'z' in df.columns and 'gene' in df.columns:
                modal = p.stem.replace('pls_weights_', '')
                out[modal] = df.loc[df['z'].abs() >= 2.0, 'gene'].astype(str).tolist()
    return out


def _load_selection_freq(tdir: Path, modal: str) -> dict[str, float] | None:
    p = tdir / f'pls_stability_{modal}.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if 'gene' not in df.columns or 'selection_freq' not in df.columns:
        return None
    return dict(zip(df['gene'].astype(str), df['selection_freq'].astype(float)))


def _load_expression_for_background(tdir: Path, ahba_dir: Path | None = None
                                    ) -> dict[str, float] | None:
    """从 AHBA 表达矩阵计算每个基因的平均表达（用于表达量匹配的零模型）。"""
    cands = []
    if ahba_dir is not None:
        cands.append(Path(ahba_dir))
    cands += [tdir.parent / 'ahba', tdir.parent.parent / 'data' / 'ahba',
              Path(str(tdir).replace('transcriptome', 'ahba'))]
    for cand in cands:
        if not cand or not cand.exists():
            continue
        npy = sorted(cand.glob('expression_*.npy'))
        gene_json = sorted(cand.glob('genes_*.json'))
        if npy and gene_json:
            expr = np.load(npy[0])
            genes = json.loads(gene_json[0].read_text(encoding='utf-8'))
            if expr.shape[1] == len(genes):
                return dict(zip(genes, np.nanmean(expr, axis=0)))
    return None


def _load_background(tdir: Path, annot_dir: Path, annotatable: set[str],
                     ahba_dir: Path | None = None) -> tuple[set[str], str]:
    """构造背景基因集：优先 AHBA 检出基因 ∩ 可注释基因（方案 5.5 的强制要求）。

    ``ahba_dir`` 必须能被定位到，否则背景会退化为"全部可注释基因"（约 3.2 万个），
    而候选基因只来自 AHBA 检出的约 1.5 万个——分母过大将**系统性低估期望重叠数**，
    使富集 fold enrichment 虚高。脚本会打印 warn，该情形下的结果不得用于论文。
    """
    cands = []
    if ahba_dir is not None:
        cands.append(Path(ahba_dir))
    cands += [tdir.parent / 'ahba', Path(str(tdir).replace('transcriptome', 'ahba'))]
    for cand in cands:
        if not cand or not cand.exists():
            continue
        for p in sorted(cand.glob('genes_*.json')):
            try:
                genes = set(json.loads(p.read_text(encoding='utf-8')))
            except Exception:  # noqa: BLE001
                continue
            if genes:
                return genes & annotatable, 'ahba_detected_intersect_annotatable'
    return set(annotatable), 'fallback_all_annotated'


def _load_cell_reference(path: Path) -> en.CellTypeExpression:
    df = pd.read_csv(path, index_col=0)
    return en.CellTypeExpression(df.to_numpy(dtype=float),
                                 [str(c) for c in df.columns],
                                 [str(i) for i in df.index])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='功能富集与细胞类型分析（阶段八）')
    ap.add_argument('--transcriptome', type=Path, required=True)
    ap.add_argument('--annotations', type=Path, default=ROOT / 'data' / 'annotations')
    ap.add_argument('--ahba-dir', type=Path, default=None,
                    help='AHBA 输出目录（含 expression_*.npy / genes_*.json）。'
                         '用于构造正确的背景基因集（AHBA 检出 ∩ 可注释）；'
                         '不提供且无法自动定位时，背景会退化为全部可注释基因，'
                         '导致富集膨胀，结果不得用于论文')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'data' / 'enrichment')
    ap.add_argument('--config', type=Path, default=ROOT / 'config' / 'analysis.yaml')
    ap.add_argument('--modality', default=None, help='只分析指定模态（默认全部）')
    ap.add_argument('--min-set-size', type=int, default=10)
    ap.add_argument('--max-set-size', type=int, default=500)
    ap.add_argument('--n-perm', type=int, default=None)
    ap.add_argument('--gsea-perm', type=int, default=500)
    ap.add_argument('--no-gsea', action='store_true',
                    help='跳过 GSEA。实测 GSEA 单模态约 26 分钟（ORA 仅数秒），'
                         '而 GSEA 属二级分析，可在需要时单独补跑')
    ap.add_argument('--ewce-perm', type=int, default=None)
    ap.add_argument('--cell-type-reference', type=Path, default=None)
    ap.add_argument('--cell-proportions', type=Path, default=None)
    ap.add_argument('--image-map', type=Path, default=None)
    ap.add_argument('--atlas-centroids', type=Path,
                    default=ROOT / 'data' / 'atlas'
                            / 'Schaefer2018_200Parcels_7Networks_centroids_MNI.csv')
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args(argv)

    cfg = cl.load(args.config) if args.config.exists() else {}
    seed = args.seed if args.seed is not None else int(cl.dig(cfg, 'seeds.global', 0))
    n_perm = args.n_perm or 1000
    ewce_perm = args.ewce_perm or 1000

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates(args.transcriptome)
    if not candidates:
        print(f'{args.transcriptome} 下没有候选基因文件；请先运行 '
              'scripts/run_transcriptome.py', file=sys.stderr)
        return 2
    if args.modality:
        candidates = {k: v for k, v in candidates.items() if k == args.modality}
    print(f'候选基因：' + '，'.join(f'{k}={len(v)}' for k, v in candidates.items()))

    # ---------- 读取注释 ----------
    if not (args.annotations / 'go_gene_sets.tsv').exists():
        print(f'{args.annotations} 缺少 GO 注释；请先运行 scripts/fetch_annotations.py',
              file=sys.stderr)
        return 2
    go = en.read_go_sets(args.annotations, namespaces=('biological_process',))
    react = (en.read_reactome_sets(args.annotations)
             if (args.annotations / 'reactome_gene_sets.tsv').exists() else None)
    print(f'注释：GO-BP {go.n_sets} 个集合'
          + (f'；Reactome {react.n_sets} 个集合' if react else ''))

    annotatable = go.genes() | (react.genes() if react else set())
    background, bg_source = _load_background(args.transcriptome, args.annotations,
                                           annotatable, args.ahba_dir)
    print(f'背景基因集：{len(background)} 个（来源 {bg_source}）')
    if len(background) == 0:
        print('背景基因集为空：候选基因的命名空间与注释库不匹配 '
              '（例如使用合成基因名）。富集分析无法进行。', file=sys.stderr)
        (args.out_dir / 'ora_summary.json').write_text(json.dumps(
            {'error': 'empty_background', 'background_source': bg_source,
             'n_candidates': {k: len(v) for k, v in candidates.items()},
             'note': '候选基因与注释库无交集；若使用真实 AHBA 数据请检查基因名格式'},
            ensure_ascii=False, indent=2), encoding='utf-8')
        (args.out_dir / 'ewce_summary.json').write_text(json.dumps(
            {'error': 'empty_background'}, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'已写出说明文件：{args.out_dir}')
        return 3
    if bg_source == 'fallback_all_annotated':
        print('  [warn] 未找到 AHBA 检出基因列表，背景退化为全部可注释基因；'
              '这会导致富集膨胀，结果不应直接用于论文', file=sys.stderr)

    go_f = en.filter_gene_sets(go, background, args.min_set_size, args.max_set_size)
    react_f = (en.filter_gene_sets(react, background, args.min_set_size, args.max_set_size)
               if react else None)
    print(f'筛后集合：GO-BP {go_f.n_sets}' + (f'；Reactome {react_f.n_sets}'
                                             if react_f else ''))

    expr_map = _load_expression_for_background(args.transcriptome, args.ahba_dir)
    if expr_map:
        print(f'表达量数据可用于分层零模型（{len(expr_map)} 个基因）')

    # ---------- 逐模态富集 ----------
    ora_sum: dict = {'background_source': bg_source, 'n_background': len(background),
                     'n_perm': n_perm, 'go_bp_sets': go_f.n_sets,
                     'reactome_sets': react_f.n_sets if react_f else 0,
                     'modalities': {}}
    # 出图脚本读取 glob('ora_*.csv')，故文件名统一前缀 ora_
    for modal, genes in candidates.items():
        print(f'\n[{modal}] 候选基因 {len(genes)} 个')
        entry: dict = {'n_candidates': len(genes)}

        res_h = en.ora(genes, go_f, background, min_overlap=3)
        rows = res_h.table
        if rows:
            en.write_table(args.out_dir / f'ora_{modal}.csv', rows)
            n_sig = sum(1 for r in rows if r.get('fdr_significant'))
            entry['ora_go'] = {'n_tested': len(rows), 'n_fdr_significant': n_sig,
                               'min_q': float(min(r['q'] for r in rows))}
            if react_f:
                rr = en.ora(genes, react_f, background, min_overlap=3)
                if rr.table:
                    en.write_table(args.out_dir / f'ora_reactome_{modal}.csv', rr.table)
                    entry['ora_reactome'] = {
                        'n_tested': len(rr.table),
                        'n_fdr_significant': sum(1 for r in rr.table
                                                 if r.get('fdr_significant'))}
            print(f'  ORA(GO-BP)：{len(rows)} 个集合，FDR 显著 {n_sig} 个')
        else:
            entry['ora_go'] = {'note': '无集合通过最小重叠数要求'}

        # 置换校准（优先用入选概率加权，其次表达量分层）
        freq = _load_selection_freq(args.transcriptome, modal)
        if freq is not None:
            rp = en.ora_permuted(genes, go_f, background, selection_prob=freq,
                                 n_perm=n_perm, seed=seed, min_overlap=3, n_bins=10)
            en.write_table(args.out_dir / f'ora_permuted_{modal}.csv', rp.table)
            entry['ora_permuted'] = {
                'zero_model': 'selection_prob_weighted', 'n_tested': len(rp.table),
                'n_fdr_significant': sum(1 for r in rp.table if r.get('fdr_significant'))}
            print(f'  ORA 置换（入选概率加权）：FDR 显著 '
                  f"{entry['ora_permuted']['n_fdr_significant']} 个")
        elif expr_map:
            rp = en.ora_permuted(genes, go_f, background, expression=expr_map,
                                 n_perm=n_perm, seed=seed, min_overlap=3, n_bins=10)
            en.write_table(args.out_dir / f'ora_permuted_{modal}.csv', rp.table)
            entry['ora_permuted'] = {
                'zero_model': 'expression_stratified', 'n_tested': len(rp.table),
                'n_fdr_significant': sum(1 for r in rp.table if r.get('fdr_significant'))}
            print(f'  ORA 置换（表达量分层）：FDR 显著 '
                  f"{entry['ora_permuted']['n_fdr_significant']} 个")
        else:
            entry['ora_permuted'] = {'note': '缺少入选概率与表达量，无法做置换校准'}

        # GSEA（基因集水平，需要基因权重向量）
        wfile = args.transcriptome / f'pls_weights_{modal}.csv'
        if args.no_gsea:
            entry['gsea'] = {'note': '按 --no-gsea 跳过（可在需要时单独补跑）'}
        elif wfile.exists():
            w = pd.read_csv(wfile)
            if 'gene' in w.columns and 'z' in w.columns:
                ranked = w['gene'].astype(str).tolist()
                weights = w['z'].to_numpy(dtype=float)
                gr = en.gsea(ranked, weights, go_f, n_perm=args.gsea_perm, seed=seed,
                             min_size=args.min_set_size, max_size=args.max_set_size)
                if gr.table:
                    en.write_table(args.out_dir / f'gsea_{modal}.csv', gr.table)
                    entry['gsea'] = {
                        'n_tested': len(gr.table),
                        'n_fdr_significant': sum(1 for r in gr.table
                                                 if r.get('fdr_significant'))}
                    print(f"  GSEA：{len(gr.table)} 个集合，FDR 显著 "
                          f"{entry['gsea']['n_fdr_significant']} 个")

        # 细胞类型富集
        if args.cell_type_reference is not None and args.cell_type_reference.exists():
            ce = _load_cell_reference(args.cell_type_reference)
            out = en.ewce(genes, ce, n_perm=ewce_perm, seed=seed, top_frac=0.1)
            (args.out_dir / f'ewce_{modal}.json').write_text(
                json.dumps(out, ensure_ascii=False, indent=2, default=float),
                encoding='utf-8')
            entry['ewce'] = {
                'n_cell_types': len(out['cell_types']),
                'n_fdr_significant': sum(1 for r in out['results']
                                         if r.get('fdr_significant')),
                'top_cell_type': (out['results'][0]['cell_type'] if out['results'] else None)}
            print(f"  EWCE：{entry['ewce']['n_cell_types']} 种细胞类型，"
                  f"FDR 显著 {entry['ewce']['n_fdr_significant']} 个")
        else:
            entry['ewce'] = {'note': '未提供细胞类型参考矩阵（--cell-type-reference）'}

        ora_sum['modalities'][modal] = entry

    (args.out_dir / 'ora_summary.json').write_text(
        json.dumps(ora_sum, ensure_ascii=False, indent=2, default=float), encoding='utf-8')

    # ---------- 细胞类型比例空间相关 ----------
    ewce_sum = {'n_modalities': len(ora_sum['modalities']),
                'modalities_with_ewce': [k for k, v in ora_sum['modalities'].items()
                                         if 'n_cell_types' in (v.get('ewce') or {})]}
    if args.cell_proportions is not None and args.cell_proportions.exists():
        props = pd.read_csv(args.cell_proportions, index_col=0)
        ct_names = [str(c) for c in props.columns]
        spins = None
        if args.atlas_centroids.exists():
            from tnwt.spin import gen_spinsamples
            cent = pd.read_csv(args.atlas_centroids)
            cols = {c.strip().upper(): c for c in cent.columns}
            x, y, z = cols.get('R') or cols.get('X'), cols.get('A') or cols.get('Y'), cols.get('S') or cols.get('Z')
            if x and y and z:
                coords = cent[[x, y, z]].to_numpy(dtype=float)
                spins = gen_spinsamples(coords, n_perm=min(n_perm, 2000), seed=seed)
        maps = {}
        for modal in candidates:
            f = args.transcriptome.parent / 'phenotypes' / f'effect_map_{modal}.csv'
            if args.image_map is not None:
                f = args.image_map
            if f.exists():
                df = pd.read_csv(f)
                col = 'effect' if 'effect' in df.columns else df.columns[-1]
                maps[modal] = df[col].to_numpy(dtype=float)
        for modal, img in maps.items():
            if len(img) != props.shape[0]:
                continue
            rows = en.proportion_map_correlation(props.to_numpy(dtype=float), ct_names,
                                                img, spins=spins)
            en.write_table(args.out_dir / f'cell_proportion_correlation_{modal}.csv', rows)
            print(f'  细胞比例空间相关（{modal}）：{len(rows)} 种细胞类型')
        ewce_sum['cell_proportion_analysis'] = {'n_modalities': len(maps),
                                                'spins_used': spins is not None}
    else:
        ewce_sum['cell_proportion_analysis'] = {
            'note': '未提供分区级细胞类型比例（--cell-proportions）；'
                    '方案 5.6 的"单核比例空间映射"待补'}
    (args.out_dir / 'ewce_summary.json').write_text(
        json.dumps(ewce_sum, ensure_ascii=False, indent=2, default=float),
        encoding='utf-8')

    print(f"\n总耗时 {round(time.time() - t0, 1)}s")
    print(f'输出：{args.out_dir}')
    if not args.cell_type_reference:
        print('提示：细胞类型分析需要 --cell-type-reference（基因 × 细胞类型 表达矩阵）；'
              '可从 Allen Brain Cell Atlas 导出。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
