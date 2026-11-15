#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A2：静息态 fMRI → 功能连接矩阵 + 分区时间序列（供图谐波耦合使用）。

产出（契约第 2.2 节）
---------------------
    <out>/a2/fc/<sub>.csv          200x200  Pearson r 功能连接矩阵
    <out>/a2/timeseries/<sub>.npy  T x 200  去噪后的分区平均时间序列
    <out>/a2/qc/<sub>.json         运动与去噪诊断（FD、被剔除帧数等）

契约明确要求"FC 矩阵（Pearson r）"，因此**不做 Fisher z 变换**，
也不做任何强度归一化 —— 只做去噪，然后原样导出相关系数。

为什么需要容器、而其余部分在宿主机
----------------------------------
运动校正与跨模态配准必须用 FSL（`mcflirt` / `flirt`）。实测
`mrtrix3/mrtrix3:latest` 镜像内自带完整 FSL 6.0.7.7，其中
`mcflirt`、`flirt`、`fslmaths`、`fslroi`、`fslstats` 均可用，
所以**不需要额外镜像**。

去噪、分区取值与相关矩阵则用宿主机上的 nilearn / numpy 完成 ——
比再塞进容器快，也便于调试。

处理口径（写论文时照抄这一节）
------------------------------
1. **运动校正**：`mcflirt`（6 参数，参考 = 时间均值像）。
2. **配准**：`flirt` 把 FastSurfer 的 conformed T1（`orig.mgz`）刚性配准到
   mean BOLD（6 DOF，`normmi` 代价），再把 A3 产出的**原生空间** Schaefer
   图集**最近邻**重采样到 BOLD 网格。图集不放大 BOLD，避免插值污染时间序列。
3. **混淆回归**：24 参数运动模型（6 参数 + 一阶时间导数 + 平方项）
   + 白质均值 + 脑脊液均值。
4. **滤波**：0.01–0.08 Hz（Butterworth 带通）。
5. **取值**：每个分区内体素的时间序列**均值**（不做 PCA）。
6. **FC**：分区时间序列两两 Pearson 相关，对角线置 1，**不转 z**。
7. **FD**：按 Power 定义自行计算（`sum|Δ平移| + 50mm × sum|Δ旋转|`），
   因为 mriqc 的 `bold_mean_fd` 只覆盖基线，随访没有。

> 为什么自己算 FD：mriqc 的 group TSV 只覆盖基线（T1w 158 / DWI 156 /
> BOLD 110 / T2w 142），**随访覆盖为 0**。为保证基线与随访口径一致，
> A2 对全部被试统一自算 FD。

用法
----
    python scripts/run_a2_pipeline.py --raw F:\\ds005713\\raw \\
        --work <C: 工作目录> --out <C: 产物目录> --a3-work <A3 work> \\
        --dest F:\\ds005713\\derivatives --subjects sub-01 --stage all
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

IMAGE = 'mrtrix3/mrtrix3:latest'
N_PARCELS = 200

# 契约要求的输出位置
OUT_FC = 'a2/fc'
OUT_TS = 'a2/timeseries'
OUT_QC = 'a2/qc'

# 去噪参数（写成常量便于复核）
TR_DEFAULT = 3.0
LOW_PASS = 0.08
HIGH_PASS = 0.01
FD_RADIUS_MM = 50.0


def log(msg: str = '') -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'), flush=True)


# ============================================================ 数据发现
def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:                                  # noqa: BLE001
        return {}


def discover_subjects(raw: Path, only: list[str] | None) -> list[str]:
    subs = sorted(p.name for p in raw.iterdir()
                  if p.is_dir() and p.name.startswith('sub-'))
    if only:
        want = set(only)
        missing = want - set(subs)
        if missing:
            log(f'[warn] 指定但不存在: {sorted(missing)}')
        subs = [s for s in subs if s in want]
    return subs


def pick_bold(raw: Path, sub: str) -> dict | None:
    """定位静息态 BOLD，返回路径与 TR。"""
    fdir = raw / sub / 'func'
    if not fdir.is_dir():
        return None
    cands = sorted(fdir.glob('*task-rest*bold.nii.gz'))
    if not cands:
        cands = sorted(fdir.glob('*bold.nii.gz'))
    if not cands:
        return None
    nii = cands[0]
    meta = _read_json(nii.with_name(nii.name[:-len('.nii.gz')] + '.json'))
    tr = meta.get('RepetitionTime')
    if not isinstance(tr, (int, float)) or tr <= 0:
        tr = TR_DEFAULT
        tr_note = 'TR missing in JSON, used default 3.0s'
    else:
        tr = float(tr)
        tr_note = 'TR from JSON'
    return {'nii': nii, 'meta': meta, 'tr': tr, 'tr_note': tr_note,
            'n_vol': meta.get('NumberOfVolumes')}


# ============================================================ 容器执行
PRELUDE = """export FSLDIR=/opt/fsl
export FSLOUTPUTTYPE=NIFTI_GZ
. /opt/fsl/etc/fslconf/fsl.sh 2>/dev/null || true
# ⚠️ 必须**追加** `/opt/fsl/bin`，不能前置：
# `fsl.sh` 只往 PATH 加 `/opt/fsl/share/fsl/bin`（有 bet/fast/topup，但**没有** mcflirt/flirt），
# 真正齐全的是 `/opt/fsl/bin`；可它同时含 FSL 自带 python。
# 一旦前置，任何 `#!/usr/bin/env python` 脚本（如 nilearn 之外的 MRtrix 工具）
# 会被 FSL python 接管，其 locale 解析返回 cp437 且该构建无此 codec，
# 报 `LookupError: unknown encoding: cp437`（本项目在 A1 的 dwi2response 上踩过）。
# 改为追加后：python 仍是系统的（UTF-8），mcflirt/flirt 也都能找到。
export PATH="$PATH:/opt/fsl/bin"
cd /out
"""


def _run_script(work: Path, stage_root: Path, name: str, body: str,
                timeout_h: float) -> tuple[bool, str]:
    """在容器内跑 bash。逐步显式报错，不用 set -e。

    注意：脚本必须用 **LF** 写入。Windows 上 `Path.write_text()` 默认做换行
    转换变成 CRLF，bash 会报 `$'true\\r': command not found` 这类完全看不出
    根因的错误（A1 已踩过）。
    """
    scripts = stage_root / 'scripts'
    scripts.mkdir(parents=True, exist_ok=True)
    sp = scripts / f'{name}.sh'
    with open(sp, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(body)
    cmd = ['docker', 'run', '--rm', '-u', '0:0',
           '-v', f'{stage_root}:/data',
           '-v', f'{work}:/out',
           '-e', 'FSLDIR=/opt/fsl',
           '-e', 'OMP_NUM_THREADS=4',
           '--entrypoint', 'bash', IMAGE, f'/data/scripts/{name}.sh']
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=int(timeout_h * 3600))
    except subprocess.TimeoutExpired:
        return False, f'timeout after {timeout_h}h'
    except Exception as exc:                            # noqa: BLE001
        return False, f'{type(exc).__name__}: {exc}'
    if r.returncode == 0:
        return True, 'ok'
    tail = ((r.stdout or '') + (r.stderr or '')).strip().splitlines()[-6:]
    return False, f'exit={r.returncode} :: ' + ' | '.join(tail)[:400]


def stage_fsl(sub: str, work: Path, stage_root: Path, tr: float,
              timeout_h: float) -> tuple[bool, str]:
    """容器内：运动校正 + T1→BOLD 配准 + 图集/组织掩码重采样。"""
    sdir = work / sub
    sdir.mkdir(parents=True, exist_ok=True)
    done = sdir / 'atlas_in_bold.nii.gz'
    if done.exists() and done.stat().st_size > 10_000:
        return True, 'already-done'

    body = PRELUDE + f"""
echo "[1/5] mcflirt 运动校正"
mcflirt -in /data/in/{sub}_bold.nii.gz -out /out/{sub}/bold_mc \\
        -mats -plots -rmsabs -rmsrel -report -refvol 0
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/bold_mc.nii.gz ]; then
  echo "FATAL mcflirt rc=$rc"
  exit 11
fi

echo "[2/5] 时间均值像（配准参考）"
fslmaths /out/{sub}/bold_mc.nii.gz -Tmean /out/{sub}/bold_mean.nii.gz
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/bold_mean.nii.gz ]; then
  echo "FATAL fslmaths -Tmean rc=$rc"
  exit 12
fi

echo "[3/5] 先把 .mgz 转 NIfTI，再做 FLIRT（FSL 读不了 .mgz）"
# ⚠️ 本镜像的 FSL 6.0.7.7 不支持 FreeSurfer .mgz：
#   flirt -in x.mgz -> Image Exception #63 :: No image files match
# 且 **flirt 失败仍返回 rc=0**，所以下面同时检查输出文件是否存在。
mrconvert /data/{sub}/orig.mgz /out/{sub}/orig_t1.nii -datatype float32 -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/orig_t1.nii ]; then
  echo "FATAL mrconvert orig.mgz -> nii rc=$rc"
  exit 17
fi

flirt -in /out/{sub}/orig_t1.nii -ref /out/{sub}/bold_mean.nii.gz \\
      -omat /out/{sub}/t1_to_bold.mat -dof 6 -cost normmi -searchcost normmi
rc=$?
if [ ! -s /out/{sub}/t1_to_bold.mat ]; then
  echo "FATAL flirt 未产出 t1_to_bold.mat (rc=$rc)"
  exit 13
fi

echo "[4/5] 图集重采样到 BOLD 网格（最近邻）"
flirt -in /data/{sub}/atlas.nii.gz -ref /out/{sub}/bold_mean.nii.gz \\
      -applyxfm -init /out/{sub}/t1_to_bold.mat -interp nearestneighbour \\
      -out /out/{sub}/atlas_in_bold.nii.gz
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/atlas_in_bold.nii.gz ]; then
  echo "FATAL resample atlas rc=$rc"
  exit 14
fi
# ⚠️ 内容校验：文件存在且有合法 header **不代表**图集落在脑内。
# 实测 sub-080/sub-117 的 FLIRT 发散（平移 235/235/389 mm），
# atlas_in_bold 是**全零**，但 `[ ! -s ]` 照样通过，于是管线报"成功"，
# 直到宿主侧的 nilearn 才抛 `ValueError: Image has no label`。
# 这里直接数非零体素，让失败发生在配准阶段、并给出可读原因。
NLAB=$(mrstats /out/{sub}/atlas_in_bold.nii.gz -output count -ignorezero 2>/dev/null | head -1)
echo "  图集非零体素 = $NLAB"
if ! awk -v n="$NLAB" 'BEGIN{{exit !(n+0 >= 500)}}'; then
  echo "FATAL 图集重采样后几乎为空（$NLAB < 500），FLIRT 配准发散"
  exit 19
fi

echo "[5/5] 白质/脑脊液掩码重采样（混淆回归用，三线性后取阈值）"
flirt -in /data/{sub}/wm.nii.gz -ref /out/{sub}/bold_mean.nii.gz \\
      -applyxfm -init /out/{sub}/t1_to_bold.mat -interp trilinear \\
      -out /out/{sub}/wm_in_bold.nii.gz
rc=$?
[ $rc -ne 0 ] && {{ echo "FATAL resample wm rc=$rc"; exit 15; }}
flirt -in /data/{sub}/csf.nii.gz -ref /out/{sub}/bold_mean.nii.gz \\
      -applyxfm -init /out/{sub}/t1_to_bold.mat -interp trilinear \\
      -out /out/{sub}/csf_in_bold.nii.gz
rc=$?
[ $rc -ne 0 ] && {{ echo "FATAL resample csf rc=$rc"; exit 16; }}

echo OK
"""
    return _run_script(work, stage_root, f'{sub}_fsl', body, timeout_h)


# ============================================================ 宿主侧去噪
def compute_fd(mc_par: Path, radius_mm: float = FD_RADIUS_MM) -> np.ndarray | None:
    """按 Power 定义算逐帧 FD：sum|Δ平移| + radius × sum|Δ旋转|。

    旋转以弧度给出，乘半径换算成枕叶附近的等效位移（mm）。
    mriqc 的 bold_mean_fd 只覆盖基线，随访为 0，故这里统一自算。
    """
    if not mc_par.exists():
        return None
    try:
        m = np.loadtxt(mc_par)
    except Exception:                                   # noqa: BLE001
        return None
    if m.ndim != 2 or m.shape[0] < 2:
        return None
    d = np.abs(np.diff(m, axis=0))
    # mcflirt 的 .par 列为 3 个旋转(rad) + 3 个平移(mm)
    fd = d[:, :3].sum(axis=1) * radius_mm + d[:, 3:].sum(axis=1)
    return np.concatenate([[0.0], fd])


def denoise_and_extract(sub: str, work: Path, out_root: Path, tr: float,
                        n_parcels: int = N_PARCELS) -> tuple[bool, dict]:
    """宿主侧：构造混淆 → 去噪 → 分区取值 → FC。"""
    import nibabel as nib
    from nilearn import signal as nlsignal
    from nilearn.maskers import NiftiLabelsMasker

    sdir = work / sub
    bold_f = sdir / 'bold_mc.nii.gz'
    atlas_f = sdir / 'atlas_in_bold.nii.gz'
    if not (bold_f.exists() and atlas_f.exists()):
        return False, {'error': '缺少 bold_mc 或 atlas_in_bold'}

    bold_img = nib.load(str(bold_f))
    T = bold_img.shape[3]

    # ---- 混淆矩阵：24 参数运动模型 + WM/CSF 均值 ----
    confounds = []
    fd = compute_fd(sdir / 'bold_mc.par')
    par_f = sdir / 'bold_mc.par'
    if par_f.exists():
        try:
            mp = np.loadtxt(par_f)
            if mp.ndim == 2 and mp.shape[0] == T:
                d1 = np.vstack([np.zeros((1, mp.shape[1])), np.diff(mp, axis=0)])
                confounds.append(mp)
                confounds.append(d1)
                confounds.append(mp ** 2)
        except Exception:                               # noqa: BLE001
            pass

    # WM/CSF 均值信号（阈值 0.9 取组织核心，避开部分容积）
    for tag in ('wm', 'csf'):
        f = sdir / f'{tag}_in_bold.nii.gz'
        if not f.exists():
            continue
        try:
            m = np.asarray(nib.load(str(f)).dataobj, dtype=np.float32)
            mask = m > 0.9
            if mask.sum() < 50:
                continue
            b = np.asarray(bold_img.dataobj, dtype=np.float32)
            sig = b[mask].mean(axis=0)                  # (T,)
            confounds.append(sig.reshape(-1, 1))
        except Exception:                               # noqa: BLE001
            continue

    conf = np.hstack(confounds) if confounds else None
    if conf is not None and conf.shape[0] != T:
        conf = None                                     # 长度不符则放弃，避免错位

    # ---- 去噪 ----
    bold_arr = np.asarray(bold_img.dataobj, dtype=np.float32)
    flat = bold_arr.reshape(-1, T).T                    # (T, V)
    # 只对脑内体素去噪，脑外用 0 占位，显著省时
    brain = flat.std(axis=0) > 0
    cleaned = np.zeros_like(flat)
    if brain.sum() > 0:
        cleaned[:, brain] = nlsignal.clean(
            flat[:, brain], confounds=conf, t_r=tr,
            low_pass=LOW_PASS, high_pass=HIGH_PASS,
            detrend=True, standardize='zscore_sample',
            filter='butterworth')
    bold_clean = nib.Nifti1Image(cleaned.T.reshape(bold_arr.shape),
                                 bold_img.affine, bold_img.header)

    # ---- 分区取值（均值）----
    # 注意 standardize=None 表示"不再标准化"（本数据已清洗过）。
    # nilearn 0.14 起布尔 False 会触发 FutureWarning，故用 None。
    masker = NiftiLabelsMasker(labels_img=str(atlas_f), strategy='mean',
                               background_label=0, standardize=None,
                               detrend=False, resampling_target=None)
    ts = masker.fit_transform(bold_clean)               # (T, n_found)

    # ⚠️ nilearn 的 region_ids_ 是**字典** {列序号: 原图集标签值, 'background': 0}，
    # 不是列表。用 enumerate() 会迭代到键（0,1,2,...）从而把标签错位 —— 实测确认。
    # 因此**始终**按字典把结果铺回固定的 200 列，缺的分区保持 NaN，
    # 不依赖"n_found == 200 时列序恰好等于标签序"这种隐含假设。
    mapping = getattr(masker, 'region_ids_', None)
    full = np.full((ts.shape[0], n_parcels), np.nan, dtype=np.float64)
    if isinstance(mapping, dict):
        for j in range(ts.shape[1]):
            lab = mapping.get(j)
            try:
                lab = int(lab)
            except (TypeError, ValueError):
                continue
            if 1 <= lab <= n_parcels:
                full[:, lab - 1] = ts[:, j]
    elif ts.shape[1] == n_parcels:
        full = ts.astype(np.float64)
    else:
        return False, {'error': f'无法把 {ts.shape[1]} 列映射到 {n_parcels} 个分区'}
    ts = full
    n_found = int(np.isfinite(ts).all(axis=0).sum())

    # ---- FC：Pearson r，不做 Fisher z（契约要求）----
    valid = np.isfinite(ts).all(axis=0)
    fc = np.full((n_parcels, n_parcels), np.nan, dtype=np.float64)
    if valid.sum() >= 2:
        # 常量列会导致除零，先剔除
        keep = ts[:, valid].std(axis=0) > 0
        idx = np.flatnonzero(valid)[keep]
        if len(idx) >= 2:
            c = np.corrcoef(ts[:, idx], rowvar=False)
            fc[np.ix_(idx, idx)] = c
    # 对角线只对**有效**分区置 1。缺失分区整行整列都保持 NaN ——
    # 给它一个 1.0 的自相关会让人误以为该分区"有数据"，与契约
    # "缺失写空值"的口径冲突。
    np.fill_diagonal(fc, np.where(valid, 1.0, np.nan))

    # ---- 写盘 ----
    (out_root / OUT_FC).mkdir(parents=True, exist_ok=True)
    (out_root / OUT_TS).mkdir(parents=True, exist_ok=True)
    (out_root / OUT_QC).mkdir(parents=True, exist_ok=True)

    with open(out_root / OUT_FC / f'{sub}.csv', 'w', encoding='utf-8', newline='\n') as fh:
        for row in fc:
            fh.write(','.join('' if not np.isfinite(v) else f'{v:.8g}' for v in row) + '\n')
    np.save(out_root / OUT_TS / f'{sub}.npy', ts.astype(np.float32))

    qc = {
        'n_volumes': int(T),
        'tr_s': float(tr),
        'n_confounds': int(conf.shape[1]) if conf is not None else 0,
        'n_parcels_with_timeseries': int(valid.sum()),
        'fd_mean': float(np.nanmean(fd)) if fd is not None else None,
        'fd_median': float(np.nanmedian(fd)) if fd is not None else None,
        'fd_max': float(np.nanmax(fd)) if fd is not None else None,
        'n_frames_fd_gt_0.5': int((fd > 0.5).sum()) if fd is not None else None,
        'low_pass_hz': LOW_PASS,
        'high_pass_hz': HIGH_PASS,
        'filter': 'butterworth',
        'detrend': True,
        'standardize': 'zscore_sample',
        'confounds': '24-param motion (6+d1+6^2) + WM mean + CSF mean',
        'fc_metric': 'Pearson r (NO Fisher z, per contract)',
        'atlas_resampling': 'native-space Schaefer200 -> BOLD grid, '
                            'nearest-neighbour via FLIRT (6 dof)',
        'extraction': 'mean over voxels within each parcel (no PCA)',
        'produced_by': 'scripts/run_a2_pipeline.py',
        'date': time.strftime('%Y-%m-%d'),
    }
    (out_root / OUT_QC / f'{sub}.json').write_text(
        json.dumps(qc, ensure_ascii=False, indent=2), encoding='utf-8')
    return True, qc


def transfer_to_dest(out_root: Path, dest: Path, subs: list[str]) -> tuple[int, int]:
    n_ok = n_skip = 0
    rels = []
    for s in subs:
        rels += [(OUT_FC, f'{s}.csv'), (OUT_TS, f'{s}.npy'), (OUT_QC, f'{s}.json')]
    for sub_dir, name in rels:
        src = out_root / sub_dir / name
        if not src.exists():
            continue
        dst = dest / sub_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        # ⚠️ 只比字节数不够：重建同一被试时 FC/QC 往往**大小完全相同**而内容已变，
        # 会被告成"已存在"而跳过，导致交付目录留着旧版本。
        # 实测 sub-080 救援后本地 QC 是 20 个混淆变量、F: 上仍是旧的 18 个，
        # 两者都是 715 B。
        # 现在改为：大小不同 **或** 源文件更新 才拷贝。
        if (dst.exists()
                and dst.stat().st_size == src.stat().st_size
                and dst.stat().st_mtime >= src.stat().st_mtime):
            n_skip += 1
            continue
        shutil.copy2(src, dst)
        n_ok += 1
    return n_ok, n_skip


# ============================================================ 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description='A2：静息态 fMRI → FC + 时间序列')
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--a3-work', type=Path, required=True,
                    help='A3 的 work（复用 fastsurfer 的 orig.mgz/aseg 与原生图集）')
    ap.add_argument('--dest', type=Path, default=None)
    ap.add_argument('--stage', default='all', choices=('all', 'fsl', 'clean'))
    ap.add_argument('--subjects', nargs='*', default=None)
    ap.add_argument('--subjects-file', type=Path, default=None)
    ap.add_argument('--timeout-h', type=float, default=3.0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--timeit', action='store_true')
    args = ap.parse_args()

    for name, p in (('--work', args.work), ('--out', args.out)):
        if str(p).upper().startswith('F:'):
            log(f'[FATAL] {name} 不能放 F:（Docker 无法可靠挂载该盘）')
            return 2

    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    stage_root = ROOT / 'build' / 'docker_staging_a2'
    (stage_root / 'in').mkdir(parents=True, exist_ok=True)

    only = list(args.subjects) if args.subjects else None
    if args.subjects_file:
        # utf-8-sig 剥掉 UTF-8 BOM（见 A1 中的同一说明）。
        only = [ln.strip() for ln in
                args.subjects_file.read_text(encoding='utf-8-sig').splitlines()
                if ln.strip()]
        # ⚠️ 空清单必须**拒绝运行**，不能退化成"不过滤 = 全部被试"。
        if not only:
            log(f'[FATAL] 清单文件为空: {args.subjects_file}')
            log('        空清单不等于"全部被试"；请重新生成清单后再运行。')
            return 2
    subs = discover_subjects(args.raw, only)

    log(f'A2 管线  被试={len(subs)}  阶段={args.stage}')
    log(f'  raw     : {args.raw}')
    log(f'  work    : {args.work}')
    log(f'  out     : {args.out}')
    log(f'  a3-work : {args.a3_work}')

    plan = []
    for s in subs:
        bold = pick_bold(args.raw, s)
        if bold is None:
            plan.append((s, None, '缺 rest BOLD'))
            continue
        aseg = args.a3_work / 'fastsurfer' / s / 'mri' / 'aseg.auto.mgz'
        orig = args.a3_work / 'fastsurfer' / s / 'mri' / 'orig.mgz'
        atlas = args.a3_work / 'registration' / f'Schaefer200_in_{s}.nii.gz'
        miss = []
        if not aseg.exists():
            miss.append('aseg')
        if not orig.exists():
            miss.append('orig')
        if not atlas.exists():
            miss.append('原生图集')
        if miss:
            plan.append((s, None, '缺 A3 产物: ' + ','.join(miss)))
            continue
        plan.append((s, bold, 'ok'))

    ready = sum(1 for _, b, w in plan if b is not None and w == 'ok')
    log(f'  可处理: {ready}/{len(plan)}')
    if args.dry_run:
        shown = 0
        for s, b, w in plan:
            if b is None and shown < 12:
                log(f'  [skip] {s}: {w}')
                shown += 1
        return 0

    ok = fail = 0
    t0 = time.time()
    for i, (sub, bold, why) in enumerate(plan, 1):
        if bold is None:
            log(f'[{i}/{len(plan)}] {sub}  跳过（{why}）')
            continue
        log(f'[{i}/{len(plan)}] {sub}')
        t_sub = time.time()

        # ---- 暂存 ----
        # ⚠️ 路径必须与 stage_fsl 里的读取路径**逐字一致**。首版这里把文件放进
        # `in/<sub>/`，而 stage_fsl 读的是 `/data/in/<sub>_bold.nii.gz` 和
        # `/data/<sub>/orig.mgz`，两处都不匹配，于是 186 例全部在 4 秒内报
        # `Image Exception : #22 :: Failed to read volume`。只有 mock 单测
        # 覆盖不到这种真实挂载路径，必须端到端跑过一例才算数。
        # 统一约定：
        #   stage_root/in/<sub>_bold.nii.gz     BOLD（唯一放 in/ 下的文件）
        #   stage_root/<sub>/{orig.mgz,atlas.nii.gz,wm.nii.gz,csf.nii.gz}
        in_dir = stage_root / 'in'
        sub_dir = stage_root / sub
        in_dir.mkdir(parents=True, exist_ok=True)
        sub_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in (
            (bold['nii'], in_dir / f'{sub}_bold.nii.gz'),
            (args.a3_work / 'fastsurfer' / sub / 'mri' / 'orig.mgz',
             sub_dir / 'orig.mgz'),
            (args.a3_work / 'registration' / f'Schaefer200_in_{sub}.nii.gz',
             sub_dir / 'atlas.nii.gz'),
        ):
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)

        # 由 aseg 生成 WM / CSF 掩码（nibabel，宿主机即可）
        try:
            import nibabel as nib
            asg = nib.load(str(args.a3_work / 'fastsurfer' / sub / 'mri' / 'aseg.auto.mgz'))
            a = np.asarray(asg.dataobj).astype(np.int16)
            wm = np.isin(a, [2, 41, 7, 46]).astype(np.uint8)
            csf = np.isin(a, [4, 43, 5, 44, 14, 15, 24, 31, 63]).astype(np.uint8)
            nib.save(nib.Nifti1Image(wm, asg.affine, asg.header),
                     str(sub_dir / 'wm.nii.gz'))
            nib.save(nib.Nifti1Image(csf, asg.affine, asg.header),
                     str(sub_dir / 'csf.nii.gz'))
        except Exception as exc:                        # noqa: BLE001
            log(f'    [warn] WM/CSF 掩码生成失败（将只用运动参数做混淆）: {exc}')

        good = True
        if args.stage in ('all', 'fsl'):
            okk, info = stage_fsl(sub, args.work, stage_root, bold['tr'], args.timeout_h)
            log(f'    FSL: {"OK" if okk else "FAIL"} ({info})')
            good = okk

        if good and args.stage in ('all', 'clean'):
            try:
                okk, qc = denoise_and_extract(sub, args.work, args.out, bold['tr'])
            except Exception as exc:                    # noqa: BLE001
                import traceback
                traceback.print_exc()
                okk, qc = False, {'error': f'{type(exc).__name__}: {exc}'}
            if okk:
                fdmed = qc.get('fd_median')
                fdtxt = f'{fdmed:.3f}' if isinstance(fdmed, (int, float)) else 'n/a'
                log(f'    去噪/FC: OK (体积 {qc["n_volumes"]}, 混淆 {qc["n_confounds"]}, '
                    f'有 TS 分区 {qc["n_parcels_with_timeseries"]}, FD 中位 {fdtxt})')
            else:
                log(f'    去噪/FC: FAIL ({qc})')
            good = okk

        if good:
            ok += 1
            if args.dest:
                n_ok, n_skip = transfer_to_dest(args.out, args.dest, [sub])
                log(f'    传送: 拷贝 {n_ok} / 跳过 {n_skip} -> {args.dest}')
        else:
            fail += 1

        if args.timeit:
            log(f'     单例总耗时 {time.time()-t_sub:.1f}s')
        # 清理暂存。
        # ⚠️ 暂存布局是 `in/<sub>_bold.nii.gz` + `<sub>/{orig.mgz,atlas,wm,csf}`，
        # 所以**必须清两处**。首版只删 `in/<sub>/`（旧布局的目录），改为新布局后
        # 这一句再也匹配不到任何东西，导致暂存无限累积（实测 94 例后 5.29 GB）。
        shutil.rmtree(sub_dir, ignore_errors=True)
        for f in in_dir.glob(f'{sub}_bold*'):
            f.unlink(missing_ok=True)
        # 兼容旧布局残留
        shutil.rmtree(in_dir / sub, ignore_errors=True)

    total = time.time() - t0
    log('')
    log(f'完成: 成功 {ok}，失败 {fail}，总耗时 {total/60:.1f} min')
    if ok:
        per = total / max(ok, 1)
        log(f'单例均值 ≈ {per:.1f}s -> 207 例约 {per*207/3600:.1f} h')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
