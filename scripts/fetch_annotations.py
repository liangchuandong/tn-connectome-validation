#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载并冻结真实基因本体/通路注释（阶段八所需的基因集来源）。

来源（公开，无需登录）：
  - Gene Ontology 本体：https://purl.obolibrary.org/obo/go/go-basic.obo
  - GO 人类注释（GAF）：https://current.geneontology.org/annotations/goa_human.gaf.gz
  - Reactome 基因-通路映射：https://reactome.org/download/current/NCBI2Reactome_All_Levels.txt
  - Reactome 通路名称：https://reactome.org/download/current/ReactomePathways.txt

产物（``data/annotations/``）：
  go-basic.obo.gz          原始本体（压缩保存）
  goa_human.gaf.gz         原始注释（压缩保存）
  go_terms.tsv             term_id / name / namespace / is_obsolete
  go_gene_sets.tsv         gene_symbol / term_id（长表，仅保留非 obsolete 的 BP/CC/MF）
  reactome_gene_sets.tsv   gene_symbol / pathway_id / pathway_name（仅 Homo sapiens）
  annotation_manifest.json 来源 URL、体积、sha256、条目数
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'annotations'

SOURCES = {
    'go-basic.obo.gz': 'https://purl.obolibrary.org/obo/go/go-basic.obo',
    'goa_human.gaf.gz': 'https://current.geneontology.org/annotations/goa_human.gaf.gz',
    'NCBI2Reactome_All_Levels.txt': 'https://reactome.org/download/current/NCBI2Reactome_All_Levels.txt',
    'ReactomePathways.txt': 'https://reactome.org/download/current/ReactomePathways.txt',
    # HGNC 提供 NCBI Gene ID（entrez_id）→ 标准基因符号的权威映射。
    # 本环境的 goa_human.gaf 全部来自 UniProtKB（无 NCBI 列），故必须借助该文件
    # 才能把 Reactome 的 NCBI Gene ID 转成基因符号。
    'hgnc_complete_set.txt': 'https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt',
}


def load_hgnc_entrez_map(path: Path) -> dict[str, str]:
    """从 HGNC 完整表读取 NCBI Gene ID(entrez_id) → 标准 symbol。"""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open('r', encoding='utf-8', errors='replace') as fh:
        header = fh.readline().rstrip('\n').split('\t')
        try:
            i_entrez = header.index('entrez_id')
            i_symbol = header.index('symbol')
        except ValueError:
            return out
        for line in fh:
            f = line.rstrip('\n').split('\t')
            if len(f) <= max(i_entrez, i_symbol):
                continue
            gid, sym = f[i_entrez].strip(), f[i_symbol].strip()
            if gid and sym:
                out.setdefault(gid, sym)
    return out


def fetch(url: str, dest: Path, gzip_it: bool = False) -> dict:
    """下载并保存。

    ``gzip_it`` 仅在源文件**未压缩**时才压缩保存。若源文件本身已是 gzip
    （以 ``\\x1f\\x8b`` 开头），则原样保存——否则会造成 gzip-of-gzip，
    使用方解压一层后得到的仍是二进制，解析出 0 条记录（本项目曾实际踩到该坑）。
    """
    req = urllib.request.Request(url, headers={'User-Agent': 'tnwt-annot/1.0'})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
    already_gz = raw[:2] == b'\x1f\x8b'
    if gzip_it and not already_gz:
        payload = gzip.compress(raw, compresslevel=6)
    else:
        payload = raw
    dest.write_bytes(payload)
    return {'url': url, 'n_bytes_raw': len(raw), 'n_bytes_stored': len(payload),
            'source_already_gzipped': already_gz,
            'sha256_raw': hashlib.sha256(raw).hexdigest()}


def parse_obo(path: Path) -> list[tuple[str, str, str, bool]]:
    """解析 OBO 的 [Term] 段，返回 (id, name, namespace, is_obsolete)。"""
    terms = []
    cur: dict = {}
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line == '[Term]':
                cur = {}
                continue
            if not line:
                if cur.get('id'):
                    terms.append((cur['id'], cur.get('name', ''),
                                  cur.get('namespace', ''),
                                  cur.get('is_obsolete', 'false') == 'true'))
                cur = {}
                continue
            if line.startswith('id: '):
                cur['id'] = line[4:].strip()
            elif line.startswith('name: '):
                cur['name'] = line[6:].strip()
            elif line.startswith('namespace: '):
                cur['namespace'] = line[11:].strip()
            elif line.startswith('is_obsolete: '):
                cur['is_obsolete'] = line[13:].strip()
    if cur.get('id'):
        terms.append((cur['id'], cur.get('name', ''), cur.get('namespace', ''),
                      cur.get('is_obsolete', 'false') == 'true'))
    return terms


GAF_EVIDENCE_EXCLUDE = {'IEA', 'ND', 'NR', 'IC', 'IPI', 'ISS', 'ISA', 'ISM', 'IGI', 'IMP',
                        'IEP', 'HTP', 'HDA', 'HMP', 'HGI', 'HEP', 'IBA', 'IBD', 'IKR', 'IRD'}
GAF_EVIDENCE_EXPERIMENTAL = {'EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'HTP', 'HDA', 'HMP',
                            'HGI', 'HEP'}


def parse_gaf(path: Path, namespace_filter: set[str] | None = None,
              experimental_only: bool = False
              ) -> tuple[dict[str, set[str]], dict[str, set[str]], int]:
    """解析 GAF。返回 (gene → terms, term → genes, 跳过的行数)。"""
    gene2terms: dict[str, set[str]] = defaultdict(set)
    term2genes: dict[str, set[str]] = defaultdict(set)
    skipped = 0
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('!'):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 15:
                skipped += 1
                continue
            symbol, qualifier, term, evidence, aspect = f[2], f[3], f[4], f[6], f[8]
            if not symbol or symbol == '':
                skipped += 1
                continue
            if 'NOT' in qualifier.split('|'):
                continue
            if aspect not in ('P', 'C', 'F'):
                continue
            if experimental_only and evidence not in GAF_EVIDENCE_EXPERIMENTAL:
                continue
            gene2terms[symbol].add(term)
            term2genes[term].add(symbol)
    return gene2terms, term2genes, skipped


def build_ncbi_symbol_map(gaf_path: Path) -> dict[str, str]:
    """从 GAF 提取 NCBI Gene ID → gene symbol 映射。

    实测本环境的 goa_human.gaf 列结构与标准 GAF 2.2 一致但并不都是 NCBI 来源：
    ``[0]`` 为数据库名（多为 ``UniProtKB``），``[1]`` 为该库主 ID，
    ``[2]`` 为基因符号，``[10]`` 为同义短名。因此：

    - 若 ``[0] == 'NCBI'``，则 ``[1]`` 是 NCBI Gene ID，直接建立映射；
    - 若 ``[0] == 'UniProtKB'``，则 ``[10]`` 提供符号，可建立 UniProt→symbol 映射，
      但**不能**当作 NCBI Gene ID 使用。

    本函数只返回真正的 NCBI Gene ID 映射（可能为空，取决于注释文件来源）。
    """
    out: dict[str, str] = {}
    with gzip.open(gaf_path, 'rt', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('!'):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 3:
                continue
            db, dbid, symbol = f[0].strip(), f[1].strip(), f[2].strip()
            if db == 'NCBI' and dbid and symbol:
                out.setdefault(dbid, symbol)
    return out


def build_gene_id_maps(gaf_path: Path) -> dict[str, dict[str, str]]:
    """建立多种来源的基因 ID → symbol 映射（NCBI / UniProtKB）。"""
    maps: dict[str, dict[str, str]] = {'NCBI': {}, 'UniProtKB': {}}
    with gzip.open(gaf_path, 'rt', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('!'):
                continue
            f = line.rstrip('\n').split('\t')
            if len(f) < 3:
                continue
            db, dbid, symbol = f[0].strip(), f[1].strip(), f[2].strip()
            short = f[10].strip() if len(f) > 10 else ''
            name = symbol or short
            if db in maps and dbid and name:
                maps[db].setdefault(dbid, name)
    return maps


def parse_reactome(ncbi2r: Path, names: Path) -> tuple[list[tuple[str, str, str]], int]:
    """解析 Reactome 的 NCBI2Reactome_All_Levels.txt（仅 Homo sapiens）。

    实测列结构（6 列，**无基因符号列**）：:

        [0] NCBI gene id        [1] Reactome pathway id    [2] pathway URL
        [3] pathway name        [4] evidence              [5] species

    因此基因符号需要由 NCBI Gene ID 映射得到（见 ``load_ncbi_symbol_map``）。
    返回 [(gene_symbol_or_id, pathway_id, pathway_name)]。
    """
    name_map: dict[str, str] = {}
    with names.open('r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            f = line.rstrip('\n').split('\t')
            if len(f) >= 3:
                name_map[f[0]] = f[1]
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0
    with ncbi2r.open('r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            f = line.rstrip('\n').split('\t')
            if len(f) < 6:
                skipped += 1
                continue
            gene = f[0].strip()
            pathway = f[1].strip()
            pname = f[3].strip() or name_map.get(pathway, '')
            species = f[5].strip().lower()
            if 'homo sapiens' not in species:
                continue
            if not gene or not pathway:
                skipped += 1
                continue
            key = (gene, pathway)
            if key in seen:
                continue
            seen.add(key)
            rows.append((gene, pathway, pname))
    return rows, skipped


def load_ncbi_symbol_map(path: Path) -> dict[str, str]:
    """读取 NCBI Gene ID → gene symbol 映射（每行：id\\tsymbol）。"""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open('r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            f = line.rstrip('\n').split('\t')
            if len(f) >= 2 and f[0] and f[1]:
                out[f[0].strip()] = f[1].strip()
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {'files': {}}

    for name, url in SOURCES.items():
        dest = OUT / name
        if dest.exists() and dest.stat().st_size > 0:
            print(f'[skip] {name} 已存在（{dest.stat().st_size/1024/1024:.1f} MB）')
            continue
        try:
            gz = name.endswith('.gz')
            info = fetch(url, dest, gzip_it=gz)
            manifest['files'][name] = info
            print(f'[ok]   {name} 原始 {info["n_bytes_raw"]/1024/1024:.1f} MB '
                  f'→ 存储 {info["n_bytes_stored"]/1024/1024:.1f} MB')
        except Exception as exc:  # noqa: BLE001
            print(f'[FAIL] {name}: {exc}', file=sys.stderr)
            manifest['files'][name] = {'url': url, 'error': str(exc)}

    # --- 解析 GO 本体 ---
    obo = OUT / 'go-basic.obo.gz'
    if obo.exists():
        terms = parse_obo(obo)
        with (OUT / 'go_terms.tsv').open('w', encoding='utf-8', newline='') as fh:
            fh.write('term_id\tname\tnamespace\tis_obsolete\n')
            for tid, nm, ns, obs in terms:
                fh.write(f'{tid}\t{nm}\t{ns}\t{int(obs)}\n')
        ns_count: dict[str, int] = defaultdict(int)
        for _, _, ns, obs in terms:
            if not obs:
                ns_count[ns] += 1
        manifest['go_terms'] = {'total': len(terms),
                                'by_namespace_nonobsolete': dict(ns_count)}
        print(f'[ok]   GO 本体：{len(terms)} 个 term，非 obsolete 按 namespace {dict(ns_count)}')

    # --- 解析 GAF ---
    gaf = OUT / 'goa_human.gaf.gz'
    id2sym: dict[str, str] = {}
    if gaf.exists():
        gene2terms, term2genes, skipped = parse_gaf(gaf)
        with (OUT / 'go_gene_sets.tsv').open('w', encoding='utf-8', newline='') as fh:
            fh.write('gene_symbol\tterm_id\n')
            for g in sorted(gene2terms):
                for t in sorted(gene2terms[g]):
                    fh.write(f'{g}\t{t}\n')
        manifest['go_annotations'] = {'n_genes': len(gene2terms), 'n_terms': len(term2genes),
                                      'n_pairs': sum(len(v) for v in gene2terms.values()),
                                      'skipped_lines': skipped}
        print(f'[ok]   GO 注释：{len(gene2terms)} 基因 × {len(term2genes)} term，'
              f'{sum(len(v) for v in gene2terms.values())} 个基因-term 对'
              + (f'（跳过 {skipped} 行）' if skipped else ''))

    # --- NCBI Gene ID → symbol（用于 Reactome）---
    hgnc = OUT / 'hgnc_complete_set.txt'
    id2sym = load_hgnc_entrez_map(hgnc)
    if id2sym:
        with (OUT / 'ncbi_gene_symbol.tsv').open('w', encoding='utf-8', newline='') as fh:
            for gid, sym in sorted(id2sym.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0):
                fh.write(f'{gid}\t{sym}\n')
        manifest['ncbi_symbol_map'] = {'source': 'HGNC hgnc_complete_set',
                                       'n_entries': len(id2sym)}
        print(f'[ok]   NCBI Gene ID → symbol（HGNC）：{len(id2sym)} 条')
    else:
        print('[warn] HGNC 映射不可用，Reactome 将无法转换为基因符号', file=sys.stderr)

    # --- 解析 Reactome ---
    n2r = OUT / 'NCBI2Reactome_All_Levels.txt'
    names = OUT / 'ReactomePathways.txt'
    if n2r.exists() and names.exists():
        rows, skipped = parse_reactome(n2r, names)
        mapped = [(id2sym[g], p, nm) for g, p, nm in rows if g in id2sym]
        unmapped = len(rows) - len(mapped)
        with (OUT / 'reactome_gene_sets.tsv').open('w', encoding='utf-8', newline='') as fh:
            fh.write('gene_symbol\tpathway_id\tpathway_name\n')
            for sym, pid, pname in mapped:
                fh.write(f'{sym}\t{pid}\t{pname}\n')
        manifest['reactome'] = {'n_pairs_raw': len(rows), 'n_pairs_mapped': len(mapped),
                                'n_unmapped_ncbi_ids': unmapped,
                                'n_genes': len({r[0] for r in mapped}),
                                'n_pathways': len({r[1] for r in mapped}),
                                'skipped': skipped}
        print(f'[ok]   Reactome：{len(mapped)} 对（{len({r[0] for r in mapped})} 基因 × '
              f'{len({r[1] for r in mapped})} 通路），未映射 NCBI ID {unmapped} 对')

    (OUT / 'annotation_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nmanifest: {OUT / "annotation_manifest.json"}')
    return 0 if manifest.get('go_annotations') else 1


if __name__ == '__main__':
    raise SystemExit(main())
