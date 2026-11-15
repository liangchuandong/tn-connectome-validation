#!/usr/bin/env python -W ignore
# -*- coding: utf-8 -*-
"""可复现性清单（provenance snapshot）。

投稿时需要的"可复现性声明"必须能回答三件事：**用了什么环境、什么版本的数据、什么代码**。
本脚本把这三件事固化成一份带哈希的清单，便于审稿人核验与后续复现。

产出
----
- ``docs/provenance.json``        机器可读：环境、依赖版本、输入文件哈希、目录体量、运行产物
- ``docs/reproducibility.md``     人可读：可直接粘贴进论文"数据与代码可用性"章节

记录内容
--------
1. **运行环境**：Python 版本与实现、平台、可执行文件路径、CPU/内存（若可取）
2. **依赖版本**：本环境实际安装的包及版本（含缺失的关键包清单）
3. **输入文件指纹**：对关键输入逐个记录 ``sha256`` + 字节数 + 修改时间
   （冻结的临床表、数据集清单、图集、注释、以及用户侧派生数据若存在）
4. **目录体量**：各数据目录的文件数与总字节数
5. **分析产物**：各阶段输出目录存在性与文件数（判断跑到哪一步）
6. **代码指纹**：本仓库自有源码（``src/``、``scripts/``、``tests/``、顶层 ``verify_*.py``）
   的逐个 sha256，便于确认"结果对应哪一版代码"

注意
----
- 本脚本**只读**，不修改任何被记录的文件。
- 对**非本工作流创建**的文件同样只做只读指纹记录（不解释、不修改）。
- 大目录默认不逐文件哈希（只统计体量与文件数），可通过 ``--hash-dir`` 显式开启。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 关键输入（存在则记录，缺失则在报告中标注"未就绪"）
KEY_INPUTS = [
    'data/frozen/participants_frozen.tsv',
    'data/frozen/participants_derived.tsv',
    'data/frozen/clinical_inventory.md',
    'data/frozen/group_stats.md',
    'data/manifests/ds005713_files.csv',
    'data/manifests/ds005713_summary.json',
    'data/raw_metadata/dataset_description.json',
    'data/raw_metadata/participants.xlsx',
    'data/raw_metadata/README',
    'data/raw_metadata/CHANGES',
    'data/atlas/Schaefer2018_200Parcels_7Networks_centroids_MNI.csv',
    'data/atlas/atlas_manifest.json',
    'data/annotations/annotation_manifest.json',
    'data/annotations/go_gene_sets.tsv',
    'data/annotations/reactome_gene_sets.tsv',
    'config/analysis.yaml',
]

# 分析产物目录（判断跑到哪一步）
OUTPUT_DIRS = [
    'data/phenotypes', 'data/longitudinal', 'data/transcriptome',
    'data/enrichment', 'figures', 'paper',
    'build/chain_verify', 'build/guardrail',
]

# 需上报版本的依赖（缺失也上报，便于说明为什么某些步骤是自实现）
WATCHED_PACKAGES = [
    'numpy', 'pandas', 'scipy', 'statsmodels', 'scikit-learn', 'matplotlib',
    'nibabel', 'nilearn', 'netneurotools', 'abagen', 'pyls', 'openpyxl',
    'python-docx', 'lxml', 'Pillow', 'XlsxWriter',
]

CODE_DIRS = ['src', 'scripts', 'tests']
CODE_TOP_FILES = ['run_all_checks.py', 'verify_chain.py', 'verify_figures.py',
                  'verify_ora_calibration.py', 'verify_pipeline_guardrails.py',
                  'verify_spin_real_atlas.py', 'requirements.txt']


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def describe_file(path: Path, do_hash: bool = True) -> dict:
    if not path.exists():
        return {'path': str(path.relative_to(ROOT)), 'exists': False}
    st = path.stat()
    info = {
        'path': str(path.relative_to(ROOT)),
        'exists': True,
        'bytes': st.st_size,
        'modified': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(st.st_mtime)),
    }
    if do_hash:
        try:
            info['sha256'] = sha256_file(path)
        except OSError as exc:
            info['sha256_error'] = str(exc)
    return info


def describe_dir(path: Path, hash_files: bool = False) -> dict:
    if not path.exists():
        return {'path': str(path.relative_to(ROOT)), 'exists': False}
    n_files = 0
    total = 0
    hashes: dict[str, str] = {}
    for p in sorted(path.rglob('*')):
        if not p.is_file():
            continue
        n_files += 1
        try:
            total += p.stat().st_size
        except OSError:
            pass
        if hash_files:
            try:
                hashes[str(p.relative_to(path))] = sha256_file(p)
            except OSError:
                pass
    out = {'path': str(path.relative_to(ROOT)), 'exists': True,
           'n_files': n_files, 'bytes': total}
    if hashes:
        out['file_sha256'] = hashes
    return out


def collect_environment() -> dict:
    env = {
        'python_version': sys.version.split()[0],
        'python_implementation': platform.python_implementation(),
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor() or None,
        'cpu_count': os.cpu_count(),
        'cwd': str(Path.cwd()),
        'path_separator': os.sep,
    }
    # 内存（Windows 用 ctypes，其他平台读 /proc）
    try:
        if os.name == 'nt':
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong),
                            ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong),
                            ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong),
                            ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong),
                            ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            env['memory_total_gb'] = round(m.ullTotalPhys / 1024 ** 3, 1)
            env['memory_available_gb'] = round(m.ullAvailPhys / 1024 ** 3, 1)
        else:
            with open('/proc/meminfo') as fh:
                for line in fh:
                    if line.startswith('MemTotal'):
                        env['memory_total_gb'] = round(int(line.split()[1]) / 1024 ** 2, 1)
    except Exception:  # noqa: BLE001
        pass
    return env


def collect_packages() -> dict:
    import importlib

    found, missing = {}, []
    for name in WATCHED_PACKAGES:
        mod_name = {'scikit-learn': 'sklearn', 'Pillow': 'PIL'}.get(name, name)
        try:
            mod = importlib.import_module(mod_name)
            found[name] = getattr(mod, '__version__', 'unknown')
        except Exception:  # noqa: BLE001
            missing.append(name)
    return {'installed': found, 'missing': missing,
            'note': '缺失的包说明对应步骤需自实现或改用独立环境'}


def collect_code() -> dict:
    out: dict = {'files': {}}
    for d in CODE_DIRS:
        p = ROOT / d
        if not p.exists():
            continue
        for f in sorted(p.rglob('*.py')):
            info = describe_file(f)
            out['files'][str(f.relative_to(ROOT))] = {
                'sha256': info.get('sha256'), 'bytes': info['bytes'],
                'modified': info['modified']}
    for name in CODE_TOP_FILES:
        f = ROOT / name
        if f.exists():
            info = describe_file(f)
            out['files'][f.name] = {'sha256': info.get('sha256'),
                                    'bytes': info['bytes'],
                                    'modified': info['modified']}
    # 源码总指纹：对所有文件哈希排序后再哈希，得到单一"代码版本"标识
    joined = ''.join(f'{k}:{v["sha256"]}' for k, v in sorted(out['files'].items())
                     if v.get('sha256'))
    out['code_fingerprint'] = hashlib.sha256(joined.encode()).hexdigest()[:32]
    out['n_source_files'] = len(out['files'])
    return out


def render_markdown(snap: dict) -> str:
    env = snap['environment']
    pkg = snap['packages']
    L: list[str] = []
    L += ['# 可复现性声明（可复现性清单）', '',
          f"生成时间：{snap['generated_at']}", '']
    L += ['## 1 运行环境', '',
          f"- Python：{env['python_version']}（{env['python_implementation']}）",
          f"- 解释器路径：`{env['python_executable']}`",
          f"- 平台：{env['platform']}",
          f"- CPU 核数：{env.get('cpu_count')}"]
    if 'memory_total_gb' in env:
        L.append(f"- 内存：{env['memory_total_gb']} GB"
                 + (f"（可用 {env['memory_available_gb']} GB）"
                    if 'memory_available_gb' in env else ''))
    L += ['', '## 2 依赖版本', '', '| 包 | 版本 |', '|---|---|']
    for k, v in sorted(pkg['installed'].items()):
        L.append(f'| {k} | {v} |')
    if pkg['missing']:
        L += ['', '**未安装**：' + '、'.join(pkg['missing']),
              '', '> 上述缺失的包对应步骤以零依赖方式自实现，并用已知值/文献值交叉验证'
                  '（见 `docs/methodological_notes.md`）。']

    L += ['', '## 3 关键输入指纹', '',
          '| 文件 | 状态 | 字节数 | sha256（前 16 位） |', '|---|---|---|---|']
    not_ready = []
    for k, v in snap['inputs'].items():
        if not v.get('exists'):
            L.append(f'| `{k}` | 未就绪 | — | — |')
            not_ready.append(k)
        else:
            L.append(f"| `{k}` | 就绪 | {v['bytes']} | {str(v.get('sha256'))[:16]}… |")
    if not_ready:
        L += ['', f"> 共 {len(not_ready)} 个输入尚未就绪；这些环节的结果尚未产生。"]
    L += ['', '## 4 数据目录体量', '', '| 目录 | 文件数 | 体积 |', '|---|---|---|']
    for k, v in snap['directories'].items():
        if v.get('exists'):
            mb = v['bytes'] / 1024 ** 2
            L.append(f"| `{k}` | {v['n_files']} | {mb:.1f} MB |")
        else:
            L.append(f'| `{k}` | 未创建 | — |')
    L += ['', '## 5 分析产物进度', '', '| 阶段输出 | 状态 | 文件数 |', '|---|---|---|']
    for k, v in snap['outputs'].items():
        state = '已生成' if v.get('exists') and v.get('n_files', 0) > 0 else (
            '空目录' if v.get('exists') else '未生成')
        L.append(f"| `{k}` | {state} | {v.get('n_files', 0)} |")

    code = snap['code']
    L += ['', '## 6 代码指纹', '',
          f"- 源码文件数：{code['n_source_files']}",
          f"- **代码版本指纹**：`{code['code_fingerprint']}`",
          '',
          '该指纹由 `src/`、`scripts/`、`tests/` 与顶层 `verify_*.py` 的全部文件哈希'
          '拼接后再哈希得到，可用于确认"某次结果对应哪一版代码"。', '',
          '## 7 如何在论文中引用', '',
          '建议在"数据与代码可用性"章节写明：', '',
          '> 全部分析由 Python 3.12 实现，未使用 scipy/statsmodels/scikit-learn/nibabel 等'
          '依赖，关键统计函数以零依赖方式自实现并用已知值交叉验证；'
          f'冻结代码版本指纹为 `{code["code_fingerprint"]}`。'
          '影像数据来自 OpenNeuro ds005713（快照 v2.0.3，DOI '
          '10.18112/openneuro.ds005713.v2.0.3，许可 CC0）；'
          '转录组数据来自 Allen Human Brain Atlas；'
          '注释数据来自 Gene Ontology、Reactome 与 HGNC。', '']
    return '\n'.join(L) + '\n'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='生成可复现性清单（只读操作）')
    ap.add_argument('--docs-dir', type=Path, default=ROOT / 'docs')
    ap.add_argument('--hash-dir', action='store_true',
                    help='同时逐文件哈希数据目录（较慢）')
    ap.add_argument('--no-hash-inputs', action='store_true',
                    help='跳过输入文件哈希（仅记录大小与时间）')
    args = ap.parse_args(argv)

    args.docs_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    do_hash = not args.no_hash_inputs

    snap: dict = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'project': 'tn-whiteterm',
        'environment': collect_environment(),
        'packages': collect_packages(),
        'inputs': {k: describe_file(ROOT / k, do_hash=do_hash) for k in KEY_INPUTS},
        'directories': {k: describe_dir(ROOT / k, hash_files=args.hash_dir)
                        for k in ['data', 'data/frozen', 'data/annotations', 'data/atlas',
                                  'derivatives']},
        'outputs': {k: describe_dir(ROOT / k) for k in OUTPUT_DIRS},
        'code': collect_code(),
        'elapsed_seconds': None,
    }
    snap['elapsed_seconds'] = round(time.time() - t0, 1)

    jp = args.docs_dir / 'provenance.json'
    jp.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding='utf-8')
    mp = args.docs_dir / 'reproducibility.md'
    mp.write_text(render_markdown(snap), encoding='utf-8')

    n_ready = sum(1 for v in snap['inputs'].values() if v.get('exists'))
    n_out = sum(1 for v in snap['outputs'].values()
                if v.get('exists') and v.get('n_files', 0) > 0)
    print(f'可复现性清单：{jp}')
    print(f'可读声明：{mp}')
    print(f'关键输入 {n_ready}/{len(KEY_INPUTS)} 就绪；'
          f'分析产物 {n_out}/{len(OUTPUT_DIRS)} 个目录有文件')
    print(f"代码指纹：{snap['code']['code_fingerprint']}"
          f"（{snap['code']['n_source_files']} 个源文件）")
    print(f"缺失依赖：{len(snap['packages']['missing'])} 个 "
          f"({', '.join(snap['packages']['missing'][:6])}…)" if
          snap['packages']['missing'] else '依赖齐全')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
