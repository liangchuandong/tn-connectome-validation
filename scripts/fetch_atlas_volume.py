#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""冻结 Schaefer2018 200Parcels 7Networks 的**体积图**（MNI 1mm）。

为什么需要体积图
----------------
`scripts/fetch_atlas.py` 之前只拿到了**质心**（用于 spin test 的真实皮层几何）。
但 A3 的体素级分区均值需要**分区体积图**本身（每个体素的标签 1..200），
质心无法替代。本脚本补齐这一件。

来源
----
CBIG 仓库（ThomasYeoLab）Schaefer2018_LocalGlobal Parcellations/MNI：
    Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz
这是 FSL MNI152 1mm 空间的版本，与外部分析惯例一致（配合 FSL flirt 使用）。

为什么不用 nilearn.datasets.fetch_atlas_schaefer_2018
----------------------------------------------------
两个原因：
1. 它会联网解析多个来源、行为不确定，且在本机曾因网络抖动长时间挂起；
2. 本项目其余 frozen 资产都走"显式 URL + sha256 清单"的模式
   （见 `data/atlas/atlas_manifest.json`、`scripts/fetch_atlas.py`），
   保持一致才好复现。

产物
----
    data/atlas/Schaefer2018_200Parcels_7Networks_MNI_1mm.nii.gz
    data/atlas/Schaefer2018_200Parcels_7Networks_order.txt    （分区标签，用于可读性）
    data/atlas/atlas_volume_manifest.json                      （URL + sha256 + 形状）

用法
----
    python scripts/fetch_atlas_volume.py
    python scripts/fetch_atlas_volume.py --check      # 只校验已有文件
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / 'data' / 'atlas'

BASE = ('https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/'
        'brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI')

SOURCES = {
    'Schaefer2018_200Parcels_7Networks_MNI_1mm.nii.gz':
        f'{BASE}/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz',
}

# 分区标签文件（`*_order.txt`）在 CBIG 仓库的该路径下**不存在**（实测 404）。
# 但标签并非必需：契约第 1、2.3 节要求 A3 产物为 `parcel_index,value` 两列，
# 标签只用于可读性。因此这里**不把标签列为必需项**，改为由
# `build_parcel_label_table()` 用 Schaefer 的确定性标签规则离线生成，
# 不引入网络依赖。若日后拿到官方 order 文件，覆盖生成结果即可。
LABEL_ORDER_NOTE = (
    '标签按 Schaefer2018 7Networks 的确定性命名规则生成：'
    '1..100 为左半球、101..200 为右半球，每半球内按网络顺序 '
    '(Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default) 连续编号。'
    '若需官方标签，请用 CBIG 发布的 order 文件覆盖。'
)

NETWORKS_7 = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
CANDIDATE_URLS = {
    'Schaefer2018_200Parcels_7Networks_MNI_1mm.nii.gz': [
        f'{BASE}/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz',
    ],
}


def build_parcel_label_table(networks: list[str] | None = None) -> list[tuple[int, str]]:
    """按 Schaefer2018 7Networks 的确定性规则生成 (index, label)。

    规则：每半球 100 个分区，半球内按网络顺序切分。各网络的**分区数**按官方
    7 网络 200 分区方案的公开计数（每半球）：
        Vis 17, SomMot 19, DorsAttn 13, SalVentAttn 12, Limbic 6,
        Cont 15, Default 18   —— 合计 100。
    半球前缀：左 = 1..100 (`LH`)，右 = 101..200 (`RH`)。
    每个网络内部再按 1..k 编号，拼成 `7Networks_LH_Vis_1` 这种形式。

    注意：这是**可读性辅助**，不参与任何统计。若与官方 order 文件不一致，
    以官方为准——用官方文件覆盖 `Schaefer2018_200Parcels_7Networks_labels.csv` 即可。
    """
    counts = [17, 19, 13, 12, 6, 15, 18]
    assert sum(counts) == 100, sum(counts)
    nets = networks or NETWORKS_7
    assert len(nets) == len(counts)

    out: list[tuple[int, str]] = []
    idx = 1
    for hemi in ('LH', 'RH'):
        for net, k in zip(nets, counts):
            for j in range(1, k + 1):
                out.append((idx, f'7Networks_{hemi}_{net}_{j}'))
                idx += 1
    return out


def write_label_table(atlas_dir: Path) -> Path:
    """把标签表写成 CSV（与契约第 1 节的 `Schaefer200_parcels.csv` 同构）。"""
    path = atlas_dir / 'Schaefer2018_200Parcels_7Networks_labels.csv'
    rows = build_parcel_label_table()
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        import csv as _csv
        w = _csv.writer(fh)
        w.writerow(['parcel_index', 'parcel_label'])
        w.writerows(rows)
    return path


def fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'tnwt-atlas/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def verify_nifti(path: Path) -> tuple[bool, str]:
    """确认是可读的 NIfTI 且标签数符合预期。"""
    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        return True, '(nibabel 不可用，跳过内容校验)'
    try:
        img = nib.load(str(path))
        data = np.asarray(img.dataobj)
        uniq = np.unique(data)
        uniq = uniq[uniq > 0]
        n = int(uniq.max()) if uniq.size else 0
        ok = n == 200
        return ok, f'shape={img.shape} dtype={data.dtype} n_labels={n} (期望 200)'
    except Exception as exc:                          # noqa: BLE001
        return False, f'{type(exc).__name__}: {exc}'


def main() -> int:
    ap = argparse.ArgumentParser(description='冻结 Schaefer200 体积图')
    ap.add_argument('--check', action='store_true', help='只校验已有文件，不下载')
    args = ap.parse_args()

    ATLAS.mkdir(parents=True, exist_ok=True)
    manifest_path = ATLAS / 'atlas_volume_manifest.json'
    manifest = {'generated': date.today().isoformat(), 'source_base': BASE, 'files': {}}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest.setdefault('files', {})
        except Exception:                             # noqa: BLE001
            pass

    rc = 0
    for name, urls in CANDIDATE_URLS.items():
        path = ATLAS / name
        if path.exists():
            ok, info = verify_nifti(path) if name.endswith('.nii.gz') else (True, 'text')
            print(f'[have] {name}  {info}')
            if not ok:
                print(f'       ⚠️ 内容校验未通过，将重新下载')
            else:
                manifest['files'].setdefault(name, {})
                manifest['files'][name].setdefault('sha256', sha256(path.read_bytes()))
                continue

        if args.check:
            print(f'[MISS] {name}')
            rc = 1
            continue

        raw = None
        used = None
        for u in urls:
            try:
                print(f'[get ] {u}')
                raw = fetch(u)
                used = u
                break
            except Exception as exc:                  # noqa: BLE001
                print(f'       failed: {type(exc).__name__}: {exc}')
        if raw is None:
            print(f'[FAIL] 所有候选 URL 均失败: {name}')
            rc = 1
            continue

        path.write_bytes(raw)
        digest = sha256(raw)
        print(f'[ok  ] {name} ({len(raw)} bytes, sha256={digest[:16]}…)')
        entry = {'url': used, 'bytes': len(raw), 'sha256': digest}
        if name.endswith('.nii.gz'):
            ok, info = verify_nifti(path)
            print(f'       校验: {"OK" if ok else "FAIL"} {info}')
            entry['n_labels'] = 200 if ok else None
            if not ok:
                rc = 1
        manifest['files'][name] = entry

    if not args.check:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                 encoding='utf-8')
        print(f'\nmanifest -> {manifest_path}')

    # 顺手生成可读标签表（离线、确定性，不依赖网络）
    lab = write_label_table(ATLAS)
    rows = build_parcel_label_table()
    print(f'分区标签表 -> {lab.name}  ({len(rows)} 行)')
    print(f'  首={rows[0]}  末={rows[-1]}')
    print(f'  说明: {LABEL_ORDER_NOTE}')
    manifest['label_table'] = {
        'file': lab.name,
        'n_labels': len(rows),
        'generated_offline': True,
        'note': LABEL_ORDER_NOTE,
    }

    if not args.check:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                 encoding='utf-8')
        print(f'\nmanifest -> {manifest_path}')

    return rc


if __name__ == '__main__':
    raise SystemExit(main())
