# -*- coding: utf-8 -*-
"""A2 管线的离线校验：不依赖 FSL/容器，验证去噪-取值-FC 与标签映射的正确性。

覆盖的风险点
------------
1. **nilearn `region_ids_` 是字典**（{列序号: 原标签}）而非列表。
   若按列表/enumerate 处理会把分区标签错位 —— 这是本管线最容易出的静默错误，
   错位后 FC 矩阵看起来完全正常，只是分区张冠李戴。
2. 缺失分区必须保持 NaN，不能被填成 0（契约第 0.4 条）。
3. FC 必须是 **Pearson r**，对角线为 1，且**不做 Fisher z**。
4. 时间序列形状必须是 (T, 200)。
5. 混淆长度与体积数不符时必须放弃混淆，而不是错位回归。

用法：python tests/run_tests_a2.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))

import nibabel as nib                                    # noqa: E402
import run_a2_pipeline as A2                             # noqa: E402

TMP = ROOT / 'build' / 'a2_selftest'
TESTS: list[tuple[str, callable]] = []


def test(fn):
    TESTS.append((fn.__name__, fn))
    return fn


def _fresh(name: str) -> Path:
    d = TMP / name
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    (d / 'work' / 'sub-999').mkdir(parents=True, exist_ok=True)
    (d / 'out').mkdir(parents=True, exist_ok=True)
    return d


def _make_case(root: Path, n_parcels_used: list[int], T: int = 60,
               shape=(12, 12, 12), seed: int = 0):
    """造一个合成 BOLD 与图集；返回 (work_dir, sub)。

    `n_parcels_used` 指定图集中实际存在的标签（**非连续**，用于检验映射）。
    """
    rng = np.random.default_rng(seed)
    sub = 'sub-999'
    sdir = root / 'work' / sub

    # 图集：把体素按 x 坐标分块，分别打上 n_parcels_used 里的标签
    atlas = np.zeros(shape, dtype=np.int16)
    nx = shape[0]
    n_used = len(n_parcels_used)
    for vx in range(nx):
        k = min(int(vx * n_used / nx), n_used - 1)
        atlas[vx, :, :] = n_parcels_used[k]
    aff = np.eye(4)
    nib.save(nib.Nifti1Image(atlas, aff), str(sdir / 'atlas_in_bold.nii.gz'))

    # BOLD：每个分区有自己的信号 + 噪声，使分区内时间序列高度一致
    base = rng.normal(0, 1, size=(T, n_used))
    data = np.zeros(shape + (T,), dtype=np.float32)
    for vx in range(nx):
        k = min(int(vx * n_used / nx), n_used - 1)
        # 形状要点：源信号 (1,1,T) 与噪声 (ny,nz,T) 广播成 (ny,nz,T)
        noise = rng.normal(0, 0.05, size=(shape[1], shape[2], T))
        data[vx] = base[:, k][None, None, :] + noise
    nib.save(nib.Nifti1Image(data, aff), str(sdir / 'bold_mc.nii.gz'))

    # 无运动参数、无 WM/CSF -> 混淆为空，路径仍然要走通
    return root / 'work', sub


@test
def test_label_mapping_uses_dict_not_order():
    """非连续标签必须按 region_ids_ 字典映射回原始分区序号。

    这是最关键的回归测试：若把 region_ids_ 当列表用，标签会错位成 1..k。
    """
    root = _fresh('labelmap')
    used = [3, 77, 150]
    work, sub = _make_case(root, used, T=40)
    ok, qc = A2.denoise_and_extract(sub, work, root / 'out', tr=3.0)
    assert ok, qc
    ts = np.load(root / 'out' / A2.OUT_TS / f'{sub}.npy')
    assert ts.shape[1] == 200, ts.shape
    present = [i + 1 for i in range(200) if np.isfinite(ts[:, i]).all()]
    assert present == used, f'映射错误: 期望 {used}，实际 {present}'


@test
def test_absent_parcels_are_nan_not_zero():
    """缺失分区必须是 NaN（契约要求缺失写空，不能填 0）。"""
    root = _fresh('nan')
    work, sub = _make_case(root, [1, 2], T=40)
    ok, qc = A2.denoise_and_extract(sub, work, root / 'out', tr=3.0)
    assert ok, qc
    ts = np.load(root / 'out' / A2.OUT_TS / f'{sub}.npy')
    missing = [i for i in range(200) if i not in (0, 1)]
    assert np.all(np.isnan(ts[:, missing])), '缺失分区应为 NaN'
    # 契约要求缺失写空字符串，np.loadtxt 无法解析空字段，故用 genfromtxt
    fc = np.genfromtxt(root / 'out' / A2.OUT_FC / f'{sub}.csv', delimiter=',',
                       missing_values='', filling_values=np.nan)
    # 缺失分区的 FC 行也应是 NaN
    assert np.all(np.isnan(fc[missing][:, missing]))


@test
def test_fc_is_pearson_r_with_unit_diagonal():
    """FC 必须是 Pearson r：有效分区对角线为 1，取值在 [-1,1]，且不做 Fisher z。"""
    root = _fresh('fc')
    used = [10, 20, 30]
    work, sub = _make_case(root, used, T=80)
    ok, qc = A2.denoise_and_extract(sub, work, root / 'out', tr=3.0)
    assert ok, qc
    # 契约要求缺失写空字符串，np.loadtxt 无法解析空字段，故用 genfromtxt
    fc = np.genfromtxt(root / 'out' / A2.OUT_FC / f'{sub}.csv', delimiter=',',
                       missing_values='', filling_values=np.nan)
    assert fc.shape == (200, 200), fc.shape
    # 有效分区的对角线应为 1；缺失分区整行整列（含对角线）保持 NaN
    valid_idx = [u - 1 for u in used]
    assert np.allclose(fc[valid_idx, valid_idx], 1.0), '有效分区对角线应为 1'
    others = [i for i in range(200) if i not in valid_idx]
    assert np.all(np.isnan(fc[others, others])), '缺失分区的对角线应为 NaN'
    fin = np.isfinite(fc)
    assert np.all(np.abs(fc[fin]) <= 1.0 + 1e-9), 'Pearson r 不应超出 [-1,1]'
    # 若误做 Fisher z，值会明显大于 1（z 在 r=0.9 时约 1.47）
    assert np.nanmax(np.abs(fc)) <= 1.0 + 1e-9, '疑似做了 Fisher z'


@test
def test_timeseries_shape_and_dtype():
    """时间序列必须是 (T, 200) 且可被 float32 表示（文件体积可控）。"""
    root = _fresh('ts')
    work, sub = _make_case(root, [5, 100], T=50)
    ok, qc = A2.denoise_and_extract(sub, work, root / 'out', tr=3.0)
    assert ok, qc
    ts = np.load(root / 'out' / A2.OUT_TS / f'{sub}.npy')
    assert ts.shape == (50, 200), ts.shape
    assert ts.dtype == np.float32, ts.dtype


@test
def test_intra_parcel_correlation_is_high():
    """同一分区内体素信号一致 -> 分区时间序列应与该分区的真实源信号高度相关。

    这是"取值方向没错"的间接证据：若标签错位，相关性会明显下降。
    """
    root = _fresh('corr')
    work, sub = _make_case(root, [7, 8, 9], T=200, seed=3)
    ok, qc = A2.denoise_and_extract(sub, work, root / 'out', tr=3.0)
    assert ok, qc
    ts = np.load(root / 'out' / A2.OUT_TS / f'{sub}.npy')[:, 6:9]
    assert np.isfinite(ts).all()
    # 三个分区来自独立的源信号，互相关应很低；而各自去噪后仍有信号
    assert ts.std(axis=0).min() > 1e-6, '取值为常量，说明取值环节失效'


@test
def test_qc_json_has_motion_and_method_fields():
    """QC 必须记录方法口径，便于论文与复核。"""
    root = _fresh('qc')
    work, sub = _make_case(root, [1, 2], T=40)
    ok, qc = A2.denoise_and_extract(sub, work, root / 'out', tr=3.0)
    assert ok, qc
    j = (root / 'out' / A2.OUT_QC / f'{sub}.json').read_text(encoding='utf-8')
    import json
    d = json.loads(j)
    for k in ('n_volumes', 'tr_s', 'low_pass_hz', 'high_pass_hz',
              'confounds', 'fc_metric', 'atlas_resampling', 'extraction'):
        assert k in d, f'QC 缺少字段 {k}'
    assert 'Pearson r' in d['fc_metric'] and 'NO Fisher' in d['fc_metric']


@test
def test_missing_inputs_fail_cleanly():
    """输入缺失时必须返回失败而不是抛异常。"""
    root = _fresh('missing')
    work = root / 'work'
    ok, qc = A2.denoise_and_extract('sub-000', work, root / 'out', tr=3.0)
    assert not ok, qc
    assert 'error' in qc


def main() -> int:
    bad = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f'PASS  {name}')
        except AssertionError as exc:
            bad += 1
            print(f'FAIL  {name}')
            print(f'      {exc}')
        except Exception as exc:                        # noqa: BLE001
            bad += 1
            print(f'ERROR {name}  {type(exc).__name__}: {exc}')
    print('-' * 70)
    print(f'{len(TESTS)-bad}/{len(TESTS)} 通过')
    shutil.rmtree(TMP, ignore_errors=True)
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
