#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成与 Schaefer 图集**完全同网格**的 MNI152 T1 参考脑。

为什么需要这个脚本（彻底解决 MNI305/MNI152 空间差异）
--------------------------------------------------
A3 需要把 Schaefer 图集（定义在 **MNI152** 空间）映射到被试原生空间。
配准需要一张 MNI152 的 T1 参考脑，但存在两个具体障碍：

1. **FreeSurfer 自带的参考脑是 MNI305，不是 MNI152**
   （`$FREESURFER_HOME/average/mni305.cor.mgz`）。
   MNI152 与 MNI305 之间存在真实的仿射差异（约 4–10 mm 量级），
   对分区级分析不可忽略 —— 不能靠"说明一下"带过。

2. **FSL 的 MNI152 模板取不到**：官方站点返回 HTML 页面而非文件，
   `fsl/fsl` 镜像不存在，raw.githubusercontent 等路径均 404 或 TLS 超时。

**解决办法**：`nilearn` **随包自带**一份 ICBM152（1mm）T1 模板
（`nilearn/datasets/data/mni_icbm152_t1_tal_nlin_sym_09a_converted.nii.gz`），
它是真正的 MNI152 空间影像，无需联网。

它与 Schaefer 图集的差异**仅是方向约定**（图集为 FSL 放射学方向，
affine 的 x 轴为 −1；nilearn 模板为神经学方向，x 轴为 +1），
这属于**可以精确重采样消除**的差异，而非空间定义不同。

本脚本把 nilearn 模板重采样到图集网格，产出：
    data/atlas/MNI152_T1_1mm_in_atlas_grid.nii.gz

之后配准用它作 `--ref`，配准结果与图集处于同一网格，
`mri_vol2vol --lta-inv` 即可把图集精确搬到原生空间 —— 空间链条自洽。

用法
----
    python scripts/make_mni152_reference.py
    python scripts/make_mni152_reference.py --check    # 只校验已有文件
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:                                   # pragma: no cover
    print('[FATAL] 缺少 nibabel', file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]
ATLAS_DIR = ROOT / 'data' / 'atlas'
ATLAS = ATLAS_DIR / 'Schaefer2018_200Parcels_7Networks_MNI_1mm.nii.gz'
OUT = ATLAS_DIR / 'MNI152_T1_1mm_in_atlas_grid.nii.gz'
MANIFEST = ATLAS_DIR / 'mni152_reference_manifest.json'

# 可能的 nilearn 模板位置（不同版本路径略有差异）
NILEARN_CANDIDATES = [
    'nilearn/datasets/data/mni_icbm152_t1_tal_nlin_sym_09a_converted.nii.gz',
    'nilearn/datasets/data/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz',
    'nilearn/datasets/data/MNI152_T1_1mm.nii.gz',
    'nilearn/datasets/data/avg152T1_brain.nii.gz',
]


def find_nilearn_template() -> Path | None:
    """在已安装的 nilearn 里找 MNI152 T1 模板。"""
    try:
        import nilearn
    except ImportError:
        return None
    base = Path(nilearn.__file__).resolve().parent
    for rel in NILEARN_CANDIDATES:
        p = base.parent / rel
        if p.exists():
            return p
        p2 = Path(nilearn.__file__).resolve().parent / rel.split('nilearn/', 1)[-1]
        if p2.exists():
            return p2
    # 兜底：搜整个 nilearn 目录
    for p in base.rglob('*.nii.gz'):
        n = p.name.lower()
        if 'icbm152' in n and 't1' in n:
            return p
        if 'mni152' in n and 't1' in n:
            return p
    return None


def resample_to_grid(src: Path, ref: Path, out: Path) -> tuple[bool, str]:
    """把 src 重采样到 ref 的网格（三线性）。"""
    src_img = nib.load(str(src))
    ref_img = nib.load(str(ref))

    if src_img.shape == ref_img.shape and np.allclose(src_img.affine, ref_img.affine):
        nib.save(nib.Nifti1Image(np.asarray(src_img.dataobj, dtype=np.float32),
                                 ref_img.affine, ref_img.header), str(out))
        return True, 'identical-grid'

    # 用 nibabel 的 resample_from_to（取向由 affine 自动处理）
    from nibabel.processing import resample_from_to
    out_img = resample_from_to(src_img, (ref_img.shape, ref_img.affine),
                               order=1, cval=0.0)
    out_img = nib.Nifti1Image(out_img.get_fdata(dtype=np.float32),
                              ref_img.affine, ref_img.header)
    out_img.set_data_dtype(np.float32)
    nib.save(out_img, str(out))
    return True, 'resampled'


def sanity_check(path: Path, ref: Path) -> tuple[bool, str]:
    """合理性检查：网格一致、脑组织占比合理、强度分布正常。"""
    a = nib.load(str(path))
    r = nib.load(str(ref))
    if a.shape != r.shape or not np.allclose(a.affine, r.affine, atol=1e-4):
        return False, f'grid mismatch: {a.shape} vs {r.shape}'
    d = np.asarray(a.dataobj, dtype=np.float32)
    nz = d[d > 0]
    if nz.size == 0:
        return False, 'all-zero volume'
    frac = nz.size / d.size
    # 脑组织在 1mm 全头模板里通常占 15%–45%
    if not (0.05 <= frac <= 0.70):
        return False, f'suspicious non-zero fraction {frac:.3f}'
    return True, (f'shape={a.shape} nonzero={frac*100:.1f}% '
                  f'min={nz.min():.1f} max={nz.max():.1f}')


def main() -> int:
    ap = argparse.ArgumentParser(description='生成与 Schaefer 图集同网格的 MNI152 参考脑')
    ap.add_argument('--atlas', type=Path, default=ATLAS)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--check', action='store_true', help='只校验已有文件')
    args = ap.parse_args()

    if not args.atlas.exists():
        print(f'[FATAL] 图集不存在: {args.atlas}\n'
              f'        先运行 scripts/fetch_atlas_volume.py', file=sys.stderr)
        return 2

    if args.check:
        if not args.out.exists():
            print(f'[MISS] {args.out}')
            return 1
        ok, info = sanity_check(args.out, args.atlas)
        print(f'[{"OK" if ok else "FAIL"}] {args.out.name}  {info}')
        return 0 if ok else 1

    src = find_nilearn_template()
    if src is None:
        print('[FATAL] 在已安装的 nilearn 中找不到 MNI152 T1 模板。\n'
              '        nilearn 通常自带以下文件之一：\n'
              + '\n'.join(f'          {c}' for c in NILEARN_CANDIDATES)
              + f'\n        请确认 nilearn 已安装且完整（当前解释器: {sys.executable}）',
              file=sys.stderr)
        return 2

    print(f'源模板 : {src}')
    print(f'参考网格: {args.atlas.name}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok, how = resample_to_grid(src, args.atlas, args.out)
    if not ok:
        print(f'[FAIL] 重采样失败: {how}', file=sys.stderr)
        return 1
    print(f'[ok] 已写出 {args.out.name}  ({how})')

    good, info = sanity_check(args.out, args.atlas)
    print(f'[{"OK" if good else "FAIL"}] 合理性检查: {info}')
    if not good:
        return 1

    a_img = nib.load(str(args.atlas))
    manifest = {
        'generated': date.today().isoformat(),
        'purpose': ('与 Schaefer200 图集完全同网格的 MNI152 T1 参考脑，'
                    '用于 A3 配准，避免 FreeSurfer 自带 MNI305 参考带来的空间差异'),
        'source_template': str(src),
        'source_note': ('nilearn 随包自带的 ICBM152 1mm T1 模板（真 MNI152 空间），'
                        '无需联网；与 Schaefer 图集的差异仅为方向约定'),
        'atlas_grid': args.atlas.name,
        'output': args.out.name,
        'output_sha256': __import__('hashlib').sha256(args.out.read_bytes()).hexdigest(),
        'check': info,
        'resample_method': how,
        'atlas_shape': list(a_img.shape),
        'atlas_affine': [[float(v) for v in row] for row in a_img.affine],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding='utf-8')
    print(f'manifest -> {MANIFEST}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
