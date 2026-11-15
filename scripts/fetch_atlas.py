#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载并冻结 Schaefer2018 200 分区（7 网络）的 MNI 质心与标签。

用途：
  1. 为 spin test 提供真实皮层几何（合成球面点不能代表脑分区分布）；
  2. 提供分析用图集分区清单（parcel_index + parcel_label）。

来源（公开）：
  - 质心：CBIG 仓库 Centroid_coordinates/*.Centroid_RAS.csv
  - 标签：CBIG 仓库 MNI/*_order.txt

产物：
  data/atlas/Schaefer2018_200Parcels_7Networks_centroids_MNI.csv
  data/atlas/Schaefer2018_200Parcels_7Networks_order.txt
  data/atlas/atlas_manifest.json
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / 'data' / 'atlas'
BASE = ('https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/'
        'brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI')

SOURCES = {
    'Schaefer2018_200Parcels_7Networks_centroids_MNI.csv':
        f'{BASE}/Centroid_coordinates/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv',
    'Schaefer2018_200Parcels_7Networks_order.txt':
        f'{BASE}/Schaefer2018_200Parcels_7Networks_order.txt',
}


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'tnwt-atlas/1.0'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> int:
    ATLAS.mkdir(parents=True, exist_ok=True)
    manifest = {'source_base': BASE, 'files': {}}
    for name, url in SOURCES.items():
        try:
            raw = fetch(url)
        except Exception as exc:  # noqa: BLE001
            print(f'[FAIL] {name}: {exc}', file=sys.stderr)
            continue
        path = ATLAS / name
        path.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        manifest['files'][name] = {'url': url, 'bytes': len(raw), 'sha256': sha}
        print(f'[ok] {name} ({len(raw)} bytes, sha256={sha[:16]}…)')

    # 质心表规范化：统一列名
    cpath = ATLAS / 'Schaefer2018_200Parcels_7Networks_centroids_MNI.csv'
    if cpath.exists():
        rows = list(csv.DictReader(io.StringIO(cpath.read_text(encoding='utf-8'))))
        if rows:
            print('质心表列名:', list(rows[0].keys()))
            print('行数:', len(rows))
            print('首行:', rows[0])
            print('末行:', rows[-1])

    # 标签表
    tpath = ATLAS / 'Schaefer2018_200Parcels_7Networks_order.txt'
    if tpath.exists():
        lines = [ln for ln in tpath.read_text(encoding='utf-8').splitlines() if ln.strip()]
        print('标签行数:', len(lines))
        print('首行:', lines[0])
        print('末行:', lines[-1])
        manifest['n_labels'] = len(lines)

    (ATLAS / 'atlas_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nmanifest: {ATLAS / "atlas_manifest.json"}')
    return 0 if len(manifest['files']) == len(SOURCES) else 1


if __name__ == '__main__':
    raise SystemExit(main())
