#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A3 上游取值逻辑的单元测试（合成数据，不依赖网络/容器/真实影像）。

覆盖点
------
1. `parcel_values` 的分区均值正确性（含空分区必须为 NaN，而不是 0）；
2. `cortical_band` 的灰白质交界带几何正确性（带内应含紧邻 WM 的皮层体素，
   且不含深部 WM 或远离 WM 的皮层）；
3. `extract_one` 端到端：CSV 严格符合契约（两列 `parcel_index,value`、
   **固定 200 行**、缺失写空），JSON 必备字段，且 `n_parcels_with_value` 与 CSV 自洽；
4. **形状不一致必须报错而非静默出错** —— 配准/重采样失效时唯一的防线；
5. `transfer_to_dest` 的幂等性（防重复拷贝）；
6. 被试发现与 T1w/T2w 选取规则。

为什么先做这一步
----------------
真实流程需要 Docker 分割 + 配准，耗时且依赖外部条件。取值逻辑是本项目自己
写的、最容易出细错的部分（索引偏移、NaN vs 0、带定义），先用合成数据锁死，
之后接真实数据时一旦出错就能立刻定位到"是分割/配准的问题"而不是自己的代码。

本机环境踩过的三个坑（都已在本文件规避）
---------------------------------------
1. `from scipy import ndimage` 在 DSH 沙箱下偶发挂起数分钟
   -> 文件开头先设死 BLAS/OpenMP 线程数；
2. `tempfile.TemporaryDirectory.cleanup()` 抛 `PermissionError [WinError 5]`
   （在 onerror 回调内，`ignore_cleanup_errors=True` 也挡不住）
   -> 不用 TemporaryDirectory；
3. **在"新建的"目录里写文件会被沙箱拒绝**（`PermissionError Errno 13`），
   而预先存在的目录可正常写
   -> 始终复用同一个固定目录 `build/test_tmp/work`，每次用前清空，不做删除收尾。

用法
----
    python tests/run_tests_a3.py
"""
from __future__ import annotations

# ⚠️ 必须在任何科学计算库之前设置线程环境变量（见文件头"坑 1"）。
import os as _os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    _os.environ.setdefault(_v, '1')

import csv
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import run_a3_pipeline as a3  # noqa: E402

SHAPE = (40, 40, 40)
_TMP_ROOT = ROOT / 'build' / 'test_tmp'
_TMP_ROOT.mkdir(parents=True, exist_ok=True)
_WORK = _TMP_ROOT / 'work'

_passed = 0
_failed: list[str] = []


class WorkDir:
    """固定的测试工作目录：每次用前清空，**不做删除收尾**（见文件头"坑 2、3"）。"""

    def __enter__(self) -> Path:
        if _WORK.exists():
            for p in sorted(_WORK.rglob('*'), key=lambda x: -len(x.parts)):
                try:
                    p.unlink() if p.is_file() else p.rmdir()
                except OSError:
                    pass
        _WORK.mkdir(parents=True, exist_ok=True)
        return _WORK

    def __exit__(self, *exc) -> bool:
        return False


def check(name: str, cond: bool, detail: str = '') -> None:
    global _passed
    if cond:
        _passed += 1
        print(f'PASS  {name}')
    else:
        _failed.append(name)
        print(f'FAIL  {name}  {detail}')


# ------------------------------------------------------------------ 1. 分区均值

def test_parcel_values_basic():
    scalar = np.arange(SHAPE[0] * SHAPE[1] * SHAPE[2], dtype=float).reshape(SHAPE)
    parcel = np.zeros(SHAPE, dtype=np.int16)
    parcel[:10, :, :] = 1
    parcel[10:20, :, :] = 2
    mask = np.ones(SHAPE, dtype=bool)

    v = a3.parcel_values(scalar, parcel, mask, n_parcels=4)
    exp1 = scalar[:10, :, :].mean()
    exp2 = scalar[10:20, :, :].mean()
    check('parcel_values: 分区1均值正确', abs(v[0] - exp1) < 1e-9, f'{v[0]} vs {exp1}')
    check('parcel_values: 分区2均值正确', abs(v[1] - exp2) < 1e-9, f'{v[1]} vs {exp2}')
    check('parcel_values: 空分区为 NaN 而非 0',
          np.isnan(v[2]) and np.isnan(v[3]), f'{v[2]}, {v[3]}')
    check('parcel_values: 返回长度等于 n_parcels', v.shape == (4,), str(v.shape))


def test_parcel_values_respects_mask():
    scalar = np.ones(SHAPE, dtype=float)
    parcel = np.ones(SHAPE, dtype=np.int16)
    mask = np.zeros(SHAPE, dtype=bool)
    mask[:5, :, :] = True
    scalar[30:, :, :] = 999.0          # mask 外的高值不得被计入
    v = a3.parcel_values(scalar, parcel, mask, n_parcels=2)
    check('parcel_values: 只在 mask 内取值', abs(v[0] - 1.0) < 1e-9, f'{v[0]}')


def test_parcel_values_ignores_out_of_range_labels():
    scalar = np.ones(SHAPE, dtype=float)
    parcel = np.zeros(SHAPE, dtype=np.int16)
    parcel[:10] = 1
    parcel[10:20] = 99                  # 超出 n_parcels
    v = a3.parcel_values(scalar, parcel, np.ones(SHAPE, bool), n_parcels=2)
    check('parcel_values: 越界标签被忽略',
          abs(v[0] - 1.0) < 1e-9 and np.isnan(v[1]), f'{v}')


def test_parcel_values_default_is_200():
    """生产默认必须是 200 —— 契约固定 200 分区，防止意外产出错误维度。"""
    check('parcel_values: 默认 n_parcels == 200（契约）', a3.N_PARCELS == 200,
          str(a3.N_PARCELS))


# ------------------------------------------------------------- 2. 交界带几何

def test_cortical_band_geometry():
    aseg = np.zeros(SHAPE, dtype=np.int16)
    # 白质核心在 z=0..9，皮层包一层 z=10..19
    aseg[:, :, 0:10] = a3.LBL_WM_LH
    aseg[:, :, 10:20] = a3.LBL_CTX_LH

    band = a3.cortical_band(aseg, band_mm=2.0, vox_mm=np.array([1.0, 1.0, 1.0]))
    check('cortical_band: 只含皮层体素',
          not (band & (aseg == a3.LBL_WM_LH)).any())
    check('cortical_band: 含紧邻 WM 的皮层 (z=10..11)',
          band[:, :, 10].all() and band[:, :, 11].all())
    check('cortical_band: 排除远离 WM 的皮层 (z=15..19)',
          not band[:, :, 15:20].any())
    zs = np.unique(np.where(band)[2])
    check('cortical_band: 带宽约 2mm', zs.size <= 3, f'unique z={zs}')


def test_cortical_band_no_wm_returns_empty():
    aseg = np.zeros(SHAPE, dtype=np.int16)
    aseg[:, :, 10:20] = a3.LBL_CTX_LH         # 只有皮层，没有 WM
    band = a3.cortical_band(aseg, 2.0, np.array([1.0, 1.0, 1.0]))
    check('cortical_band: 无 WM 时返回空带', band.sum() == 0)


# ------------------------------------------------------------ 3. 契约格式

def _make_synth_subject(tmp: Path, shape=SHAPE, atlas_shift=0):
    """造一组几何一致的 T1w / T2w / aseg / atlas 合成影像。

    分区数固定用生产常量 `a3.N_PARCELS`（=200）：契约要求产物就是 200 行，
    `extract_one` 也写死了该常量（有意的防错设计）。
    受限于小的合成脑体积，只铺 10 个分区标签，**刻意留下 190 个空分区**，
    以验证空分区被正确写成缺失（NaN -> 空字符串）而非 0 —— 这正是真实数据里
    "AHBA 供体覆盖不足的分区"会遇到的场景。
    """
    aff = np.diag([1.0, 1.0, 1.0, 1.0])

    t1 = np.full(shape, 100.0, dtype=np.float32)
    t1[:, :, 0:10] = 200.0            # 白质亮
    t1[:, :, 10:20] = 120.0
    t2 = np.full(shape, 50.0, dtype=np.float32)
    t2[:, :, 0:10] = 60.0
    t2[:, :, 10:20] = 55.0

    aseg = np.zeros(shape, dtype=np.int16)
    aseg[:, :, 0:10] = a3.LBL_WM_LH
    aseg[:, :, 10:20] = a3.LBL_CTX_LH

    parcel = np.zeros(shape, dtype=np.int16)
    n_filled = 0
    for i in range(0, shape[0], 4):
        n_filled += 1
        if n_filled > 10:
            break
        parcel[i:i + 4, :, 10:20] = n_filled
    if atlas_shift:
        parcel = np.roll(parcel, atlas_shift, axis=1)

    paths = {}
    for name, arr, dtype in (('t1', t1, np.float32), ('t2', t2, np.float32),
                             ('aseg', aseg, np.int16), ('parcel', parcel, np.int16)):
        p = tmp / f'{name}.nii.gz'
        nib.save(nib.Nifti1Image(arr.astype(dtype), aff), str(p))
        paths[name] = p
    return paths


def test_extract_one_end_to_end():
    with WorkDir() as tmp:
        p = _make_synth_subject(tmp)
        out = tmp / 'derivatives'
        ok, why = a3.extract_one('sub-001', p['t1'], p['t2'], p['aseg'],
                                 p['parcel'], out, 2.0, 'SyntheticAtlas')
        check('extract_one: 执行成功', ok, why)
        if not ok:
            return

        for metric in ('a3_gwc', 'a3_t1t2'):
            csvp = out / metric / 'sub-001.csv'
            jsonp = out / metric / 'sub-001.json'
            check(f'extract_one: {metric} CSV 存在', csvp.exists())
            check(f'extract_one: {metric} JSON 存在', jsonp.exists())
            if not csvp.exists():
                continue

            rows = list(csv.reader(open(csvp, encoding='utf-8')))
            check(f'extract_one: {metric} 表头为 parcel_index,value',
                  rows[0] == ['parcel_index', 'value'], str(rows[0]))
            check(f'extract_one: {metric} 行数为 200+1（契约固定 200 分区）',
                  len(rows) == a3.N_PARCELS + 1, f'{len(rows)}')

            idx = [int(r[0]) for r in rows[1:]]
            check(f'extract_one: {metric} 索引为 1..200 有序',
                  idx == list(range(1, a3.N_PARCELS + 1)), f'{idx[:5]}...')

            empties = [r[1] for r in rows[1:] if r[1] == '']
            zeros = [r for r in rows[1:] if r[1] == '0']
            check(f'extract_one: {metric} 缺失写空而非 0',
                  len(zeros) == 0, f'有 {len(zeros)} 个 "0" 值')
            check(f'extract_one: {metric} 确实存在空分区（覆盖不足场景）',
                  len(empties) > 0, f'空分区数={len(empties)}')

            n_valid = len(rows) - 1 - len(empties)
            check(f'extract_one: {metric} 至少一个分区有值',
                  n_valid > 0, f'n_valid={n_valid}')

            obj = json.loads(jsonp.read_text(encoding='utf-8'))
            for key in ('produced_by', 'date', 'n_parcels', 'atlas', 'metric', 'notes'):
                check(f'extract_one: {metric} JSON 含 {key}', key in obj)
            check(f'extract_one: {metric} JSON n_parcels=200',
                  obj.get('n_parcels') == a3.N_PARCELS, str(obj.get('n_parcels')))
            check(f'extract_one: {metric} JSON n_parcels_with_value 与 CSV 自洽',
                  obj.get('n_parcels_with_value') == n_valid,
                  f"{obj.get('n_parcels_with_value')} vs {n_valid}")
            check(f'extract_one: {metric} notes 声明了取带方式',
                  'band' in obj.get('notes', '').lower())
            check(f'extract_one: {metric} notes 声明未做组间比较',
                  'no group comparison' in obj.get('notes', '').lower())


def test_extract_one_unifies_t2_and_aseg_grids():
    """T2w/aseg 与 T1w 网格不一致时必须**自动统一**，而不是报错。

    这是实测得到的真实情况：T1w 与 T2w 是两次独立采集，FOV 不同
    （例如 sub-01 的 T1w 为 340×512×512、T2w 为 360×512×512），
    FastSurfer 的 aseg 又是它自己的 256³ 1mm 网格。
    因此 `extract_one` 会把 T2w（三线性）与 aseg（最近邻）重采样到 T1w 网格。
    """
    with WorkDir() as tmp:
        p = _make_synth_subject(tmp)
        # 造一个不同网格的 T2w：体积内容相同但形状不同
        t2i = nib.load(str(p['t2']))
        t2_big = tmp / 't2_big.nii.gz'
        arr = np.asarray(t2i.dataobj)
        big = np.zeros((SHAPE[0] + 20, SHAPE[1], SHAPE[2]), dtype=np.float32)
        big[:SHAPE[0]] = arr
        nib.save(nib.Nifti1Image(big, t2i.affine), str(t2_big))
        # 造一个不同网格的 aseg
        asgi = nib.load(str(p['aseg']))
        a_arr = np.asarray(asgi.dataobj)
        a_big = np.zeros((SHAPE[0] + 20, SHAPE[1], SHAPE[2]), dtype=np.int16)
        a_big[:SHAPE[0]] = a_arr
        aseg_big = tmp / 'aseg_big.nii.gz'
        nib.save(nib.Nifti1Image(a_big, asgi.affine), str(aseg_big))

        out = tmp / 'derivatives'
        ok, why = a3.extract_one('sub-004', p['t1'], t2_big, aseg_big,
                                 p['parcel'], out, 2.0, 'SyntheticAtlas')
        check('extract_one: T2w/aseg 网格不同也能成功统一', ok, why)
        if ok:
            rows = list(csv.reader(open(out / 'a3_t1t2' / 'sub-004.csv', encoding='utf-8')))
            n_valid = sum(1 for r in rows[1:] if r[1] != '')
            check('extract_one: 统一网格后仍有分区有值', n_valid > 0, f'n_valid={n_valid}')


def test_extract_one_rejects_atlas_grid_mismatch():
    """配准后的**图集**网格与 T1w 不一致时必须报错。

    与 T2w/aseg 不同（那两者可以自动重采样），图集是由配准环节产出的，
    它若与 T1w 不同网格，说明配准出了问题 —— 此时必须失败而不是静默重采样，
    否则会把配准错误掩盖成"看起来能跑"。
    """
    with WorkDir() as tmp:
        p = _make_synth_subject(tmp)
        bad = tmp / 'bad_atlas.nii.gz'
        arr = np.zeros((30, 30, 30), dtype=np.int16)
        arr[:, :, 5:15] = 1
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(bad))

        out = tmp / 'derivatives'
        ok, why = a3.extract_one('sub-002', p['t1'], p['t2'], p['aseg'],
                                 bad, out, 2.0, 'BadAtlas')
        check('extract_one: 图集网格不一致时返回失败', not ok, why)
        check('extract_one: 失败说明指出图集网格问题',
              'atlas' in why.lower() or '图集' in why, why)
        check('extract_one: 失败时不写任何产物',
              not (out / 'a3_gwc' / 'sub-002.csv').exists())


def test_extract_one_rejects_tiny_band():
    """带太小时应报错，而不是产出全 NaN 的静默结果。"""
    with WorkDir() as tmp:
        p = _make_synth_subject(tmp)
        aseg = np.zeros(SHAPE, dtype=np.int16)
        aseg[:, :, 0:10] = a3.LBL_WM_LH
        aseg[0, 0, 10] = a3.LBL_CTX_LH        # 单个体素皮层
        asegp = tmp / 'tiny_aseg.nii.gz'
        nib.save(nib.Nifti1Image(aseg, np.eye(4)), str(asegp))

        out = tmp / 'derivatives'
        ok, why = a3.extract_one('sub-003', p['t1'], p['t2'], asegp,
                                 p['parcel'], out, 2.0, 'A')
        check('extract_one: 带过小时失败', not ok, why)
        check('extract_one: 说明含 cortical band',
              'cortical band' in why.lower(), why)


# ------------------------------------------------------------ 4. 传送逻辑

def test_transfer_to_dest():
    with WorkDir() as tmp:
        out = tmp / 'out'
        dest = tmp / 'dest'
        for metric in ('a3_gwc', 'a3_t1t2'):
            (out / metric).mkdir(parents=True, exist_ok=True)
            (out / metric / 'sub-001.csv').write_text(
                'parcel_index,value\n1,0.5\n', encoding='utf-8')
            (out / metric / 'sub-001.json').write_text(
                '{"n_parcels": 200}', encoding='utf-8')

        ok, skip = a3.transfer_to_dest(out, dest, ['sub-001'])
        check('transfer_to_dest: 首次拷贝 4 个文件', ok == 4 and skip == 0,
              f'{ok}/{skip}')
        check('transfer_to_dest: 目标文件存在',
              (dest / 'a3_gwc' / 'sub-001.csv').exists())

        ok2, skip2 = a3.transfer_to_dest(out, dest, ['sub-001'])
        check('transfer_to_dest: 幂等（重复传送全部跳过）',
              ok2 == 0 and skip2 == 4, f'{ok2}/{skip2}')


# --------------------------------------------------------- 5. 发现与选取

def test_discover_and_pickers():
    with WorkDir() as tmp:
        raw = tmp / 'raw'
        (raw / 'sub-001' / 'anat').mkdir(parents=True)
        (raw / 'sub-001' / 'anat' / 'sub-001_run-01_T1w.nii.gz').write_bytes(b'x')
        (raw / 'sub-001' / 'anat' / 'sub-001_run-02_T1w.nii.gz').write_bytes(b'x')
        (raw / 'sub-001' / 'anat' / 'sub-001_T2w.nii.gz').write_bytes(b'x')
        # 只有 T1w 的被试（如 sub-009 的真实情况：清单里 16 个无 T2w 基线之一）
        (raw / 'sub-009' / 'anat').mkdir(parents=True)
        (raw / 'sub-009' / 'anat' / 'sub-009_run-01_T1w.nii.gz').write_bytes(b'x')
        (raw / 'sub-999').mkdir()          # 无 anat 目录

        check('discover: 只收有 anat 目录的被试',
              a3.discover(raw, None) == ['sub-001', 'sub-009'],
              str(a3.discover(raw, None)))
        check('discover: --subjects 过滤生效',
              a3.discover(raw, ['sub-009']) == ['sub-009'])

        t1 = a3.pick_t1w(raw, 'sub-001')
        check('pick_t1w: 多 run 时优先 run-01',
              t1 is not None and 'run-01' in t1.name, str(t1))
        check('pick_t2w: 找到 T2w',
              (a3.pick_t2w(raw, 'sub-001') or Path()).name == 'sub-001_T2w.nii.gz')
        check('pick_t2w: 无 T2w 时返回 None', a3.pick_t2w(raw, 'sub-009') is None)


def main() -> int:
    print('=' * 72)
    print('A3 上游取值逻辑单元测试（合成数据）')
    print('=' * 72)

    test_parcel_values_basic()
    test_parcel_values_respects_mask()
    test_parcel_values_ignores_out_of_range_labels()
    test_parcel_values_default_is_200()
    test_cortical_band_geometry()
    test_cortical_band_no_wm_returns_empty()
    test_extract_one_end_to_end()
    test_extract_one_unifies_t2_and_aseg_grids()
    test_extract_one_rejects_atlas_grid_mismatch()
    test_extract_one_rejects_tiny_band()
    test_transfer_to_dest()
    test_discover_and_pickers()

    print('-' * 72)
    print(f'{_passed}/{_passed + len(_failed)} 通过')
    if _failed:
        print('失败项:')
        for f in _failed:
            print(f'  - {f}')
    return 0 if not _failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
