#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
冻结 OpenNeuro ds005713 的数据清单，生成分析用 manifest。

步骤：
1) 下载数据集根目录元数据（dataset_description.json、participants.*、CHANGES、README 等）
2) 全量枚举 S3 对象（仅路径/大小/ETag，不含影像内容），保存原始 XML
3) 汇总：被试数、随访数、各 datatype 文件数与体积、派生/非 BIDS 文件、异常命名
4) 输出 manifest CSV，供后续分析读取（不解析任何影像像素）

用法：python scripts/fetch_manifest.py
"""
from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

BUCKET = 'https://s3.amazonaws.com/openneuro.org'
DATASET = 'ds005713'
ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / 'data' / 'raw_metadata'
MANIFEST_DIR = ROOT / 'data' / 'manifests'
S3_NS = '{http://s3.amazonaws.com/doc/2006-03-01/}'

ROOT_FILES = [
    'README', 'CHANGES', 'dataset_description.json', 'participants.tsv',
    'participants.json', 'participants.xlsx', 'participants_psychology.tsv',
    '.bidsignore', '.gitattributes',
]


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'dsh-manifest/1.0'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def download_metadata() -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    for name in ROOT_FILES:
        url = f'{BUCKET}/{DATASET}/{name}'
        try:
            data = fetch_bytes(url)
        except Exception as exc:  # noqa: BLE001
            print(f'  [skip] {name}: {exc}')
            continue
        (META_DIR / name).write_bytes(data)
        print(f'  [ok] {name} ({len(data)} bytes)')


def list_objects() -> list[dict]:
    objects: list[dict] = []
    marker = ''
    page = 0
    while True:
        url = f'{BUCKET}/?prefix={DATASET}/&max-keys=1000'
        if marker:
            url += f'&marker={urllib.parse.quote(marker)}'
        xml = fetch_bytes(url)
        (MANIFEST_DIR / f's3_listing_page{page:03d}.xml').write_bytes(xml)
        root = ET.fromstring(xml)
        for node in root.findall(f'{S3_NS}Contents'):
            key = node.findtext(f'{S3_NS}Key')
            size = int(node.findtext(f'{S3_NS}Size') or 0)
            etag = (node.findtext(f'{S3_NS}ETag') or '').strip('"')
            modified = node.findtext(f'{S3_NS}LastModified')
            if key.endswith('/'):
                continue
            objects.append({'key': key, 'size': size, 'etag': etag, 'last_modified': modified})
        truncated = (root.findtext(f'{S3_NS}IsTruncated') or 'false').lower() == 'true'
        nxt = root.findtext(f'{S3_NS}NextMarker')
        page += 1
        print(f'  page {page}: {len(objects)} objects (truncated={truncated})')
        if not truncated:
            break
        marker = nxt or (objects[-1]['key'] if objects else '')
        if not marker:
            break
    return objects


SUBJ_RE = re.compile(r'^' + DATASET + r'/(sub-[A-Za-z0-9]+)(?:/(.*))?$')


def summarize(objects: list[dict]) -> dict:
    subs: set[str] = set()
    fu_ids: set[str] = set()
    two_digit: set[str] = set()
    three_digit: set[str] = set()
    other_ids: set[str] = set()
    per_sub_files: dict[str, int] = defaultdict(int)
    per_sub_bytes: dict[str, int] = defaultdict(int)
    datatype_counts: Counter = Counter()
    datatype_bytes: Counter = Counter()
    suffix_counts: Counter = Counter()
    unusual: list[str] = []
    non_bids_top: Counter = Counter()

    for obj in objects:
        key = obj['key']
        rel = key[len(DATASET) + 1:]
        m = SUBJ_RE.match(key)
        if not m:
            non_bids_top[rel.split('/')[0]] += 1
            continue
        sub, rest = m.group(1), (m.group(2) or '')
        subs.add(sub)
        stem = sub[len('sub-'):]
        is_fu = stem.endswith('fu')
        core = stem[:-2] if is_fu else stem
        # 注意：README 声明患者用 sub-000（3 位）、志愿者用 sub-00（2 位），
        # 但这是作者自述，真正的分组必须以 participants.tsv 的 Clinical_finding 为准。
        # 这里只按 ID 位数收集候选集合，供人工核对命名规则是否自洽。
        bucket = two_digit if len(core) == 2 else (three_digit if len(core) == 3 else other_ids)
        if is_fu:
            fu_ids.add(core)
        else:
            bucket.add(core)
        per_sub_files[sub] += 1
        per_sub_bytes[sub] += obj['size']
        parts = rest.split('/')
        if len(parts) >= 2:
            dt = parts[0]
            datatype_counts[dt] += 1
            datatype_bytes[dt] += obj['size']
            fname = parts[-1]
            suffix = 'nii.gz' if fname.endswith('.nii.gz') else (fname.rsplit('.', 1)[-1] if '.' in fname else fname)
            suffix_counts[f'{dt}/{suffix}'] += 1
        if rest and not any(seg in rest.split('/') for seg in ('anat', 'dwi', 'func')):
            unusual.append(key)

    return {
        'n_objects': len(objects),
        'total_bytes': sum(o['size'] for o in objects),
        'n_subject_dirs': len(subs),
        'n_two_digit_ids': len(two_digit),
        'n_three_digit_ids': len(three_digit),
        'two_digit_ids': sorted(two_digit),
        'three_digit_ids': sorted(three_digit),
        'other_ids': sorted(other_ids),
        'followup_core_ids': sorted(fu_ids),
        'datatype_counts': dict(datatype_counts),
        'datatype_bytes': dict(datatype_bytes),
        'suffix_counts': dict(sorted(suffix_counts.items())),
        'non_bids_top_level': dict(non_bids_top),
        'per_subject_files': dict(sorted(per_sub_files.items())),
        'per_subject_bytes': dict(sorted(per_sub_bytes.items())),
        'unusual_paths_sample': unusual[:50],
        'n_unusual_paths': len(unusual),
    }


def write_manifest(objects: list[dict]) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    header = 'key,size_bytes,etag,last_modified\n'
    lines = [
        f'"{o["key"]}",{o["size"]},{o["etag"]},{o["last_modified"]}' for o in objects
    ]
    (MANIFEST_DIR / 'ds005713_files.csv').write_text(header + '\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    print('[1/4] 下载根目录元数据…')
    download_metadata()

    print('[2/4] 全量枚举 S3 对象…')
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    objects = list_objects()
    if not objects:
        print('未获取到任何对象，请检查网络。', file=sys.stderr)
        return 1

    print('[3/4] 生成 manifest CSV…')
    write_manifest(objects)

    print('[4/4] 汇总…')
    summary = summarize(objects)
    (MANIFEST_DIR / 'ds005713_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    gb = summary['total_bytes'] / 1024 ** 3
    print(f"\n对象总数: {summary['n_objects']}  总体积: {gb:.2f} GB")
    print(f"被试目录: {summary['n_subject_dirs']}")
    print(f"  2 位 ID（README 称志愿者）: {summary['n_two_digit_ids']}")
    print(f"  3 位 ID（README 称患者）  : {summary['n_three_digit_ids']}")
    print(f"  随访覆盖的受试者核心 ID    : {len(summary['followup_core_ids'])}")
    print(f"  其他命名                   : {summary['other_ids']}")
    print('各 datatype 文件数/体积:')
    for dt, n in sorted(summary['datatype_counts'].items(), key=lambda x: -x[1]):
        print(f'  {dt:12s} {n:6d} files  {summary["datatype_bytes"][dt] / 1024 ** 3:8.2f} GB')
    print('非被试路径顶层:', summary['non_bids_top_level'])
    print('异常路径数:', summary['n_unusual_paths'])
    print(f"\n输出目录: {MANIFEST_DIR}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
