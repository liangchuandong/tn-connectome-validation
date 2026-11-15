#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A1：弥散加权 → 白质结构连接（同时产出 A2 所需的 SC 矩阵）。

产出（契约 `docs/derived_data_contract.md` 第 1、2.1、2.2 节）
----------------------------------------------------------------
    <out>/a2/sc/<sub>_streamlines.csv   200x200  流线数连接矩阵（原始计数）
    <out>/a2/sc/<sub>_fa.csv            200x200  FA 加权连接矩阵
    <out>/a1_streamlines/<sub>.csv      200      每分区流线数节点强度（行和，未归一）
    <out>/a1_fa/<sub>.csv               200      每分区 FA 加权节点强度（行和，未归一）

A1 与 A2 共用同一次纤维追踪：A1 的两个向量就是 SC 矩阵按行求和（**排除对角线**，
自连接无生理意义）。因此本脚本一次跑完，同时满足两处契约要求。

为什么这样设计
--------------
契约第 0.6 条明确要求"不要在导出前做组间比较、协变量回归、标准化"。
因此这里只做**原始计数**与**原始 FA 加权**的导出，节点强度直接用行和，
不做 log/强度归一化（契约把 `a1_streamlines_log` 列为"可选但不建议"）。

复用 A3 的产物（重要）
----------------------
A3 已经为每个被试算出了：
  * FastSurfer 的 `aseg.auto.mgz`
  * 配准后的**原生空间** Schaefer 图集 `Schaefer200_in_<sub>.nii.gz`

这两个正是 A1 需要的输入（`5ttgen` 需要分割；`tck2connectome` 需要原生空间图集）。
因此 A1 **不重复做分割与配准**，直接复用，省掉每例约 40 分钟。

镜像与工具
----------
用 `mrtrix3/mrtrix3:latest`。实测该镜像内**自带完整 FSL 6.0.7.7**：

    /opt/fsl/bin/{topup, applytopup, eddy, eddy_cpu, bet, fslmaths, fslstats}
    /opt/fsl/data/standard/MNI152_T1_1mm.nii.gz 等

注意 FSL 的 `eddy`/`applytopup` 只在 `/opt/fsl/bin/` 下，不在默认 PATH 上
（`which` 默认只命中 `/opt/fsl/share/fsl/bin/`），所以脚本里显式把
`/opt/fsl/bin` 前置到 PATH，并 source `fsl.sh`。

相位编码方向的来源（本数据集的一个坑）
--------------------------------------
BIDS JSON **没有** `PhaseEncodingDirection` 字段，只有
`InPlanePhaseEncodingDirectionDICOM = COL`（DICOM 约定，不足以区分 AP/PA）。
可靠信息在 `SeriesDescription` 里：

    DTI_64dir_b1500_AP  -> run-01  主序列, AP
    DTI_b01_PA          -> run-02  反向 b0, PA
    DTI_b02_PA          -> run-03  反向 b0, PA
    DTI_b0_AP           -> run-04  同向 b0, AP

因此从描述后缀推断：`_AP` -> `j`，`_PA` -> `j-`。
总读出时间优先用 JSON 的 `EstimatedTotalReadoutTime`；缺失时按
`(ReconMatrixPE - 1) * EstimatedEffectiveEchoSpacing` 推算；
再缺失则借用同被试同序列参数（EchoTrainLength/AcquisitionMatrixPE 一致）的 run 的值。
每一步来源都写进 JSON 溯源，绝不静默猜测。

用法
----
    python scripts/run_a1_pipeline.py --raw F:\\ds005713\\raw \\
        --work <C: 上的工作目录> --out <C: 上的产物目录> \\
        --a3-work <A3 的 work 目录> --dest F:\\ds005713\\derivatives \\
        --subjects sub-009 --stage all

分阶段：prep / preproc / fod / 5tt / tract / connectome / all
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'mrtrix3/mrtrix3:latest'
N_PARCELS = 200

# 契约要求的输出子目录
OUT_A1_SL = 'a1_streamlines'
OUT_A1_FA = 'a1_fa'
OUT_SC_SL = 'a2/sc'          # a2/sc/<sub>_streamlines.csv
OUT_SC_FA = 'a2/sc'          # a2/sc/<sub>_fa.csv


def log(msg: str = '') -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'), flush=True)


def looks_complete(p: Path, min_bytes: int = 1_000_000,
                   n_probe: int = 16, probe: int = 4096) -> bool:
    """判断一个容器产物是否**真正写完**，而不只是"文件存在且够大"。

    为什么需要
    ----------
    MRtrix 会**预分配**输出文件再往里写。如果进程被中途杀掉（容器被停、
    机器休眠、误杀），留下的是一个**满尺寸的零文件**：体积看着正常，内容全 0。

    实测事故：sub-024 的 `dwi2fod` 正在写时容器被杀，产出 191 MB 的
    `wm_fod.mif` 且 max = 0。旧的"已完成"判断只看 `st_size > 1MB`，
    于是把它当合格产物跳过 —— 下游 tckgen 从 10 亿种子产出 **0 条流线**，
    白白耗掉约 1 小时并产出一例失败。

    为什么**不能**只看尾部
    ----------------------
    首版只读最后 8 KB，结果把正常的 `dwi_bc.mif` / `5tt.mif` 也判成未完成
    （实测两者尾部全 0）—— 脑影像的最高 z 层和角落本来就是背景，
    文件末尾天然是 0。这条路走不通。

    现在的做法
    ----------
    在文件中**均匀取 n_probe 个 4 KB 片段**，只要**任意一个**非全零就认为
    写完了。脑影像占据体积的大部分，均匀采样必然命中真实数据；
    而预分配未写入的文件处处为 0，会被准确判为未完成。
    代价是读 16 x 4 KB = 64 KB，与文件总体积无关。
    """
    try:
        if not p.exists():
            return False
        n = p.stat().st_size
        if n <= min_bytes:
            return False
        with open(p, 'rb') as fh:
            if n <= probe:
                return any(fh.read())
            span = max(0, n - probe)
            for i in range(n_probe):
                off = int(span * i / max(n_probe - 1, 1))
                fh.seek(off)
                if any(fh.read(probe)):
                    return True
        return False
    except OSError:
        return False


# ============================================================ 数据发现
def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:                                  # noqa: BLE001
        return {}


def _pe_from_desc(desc: str) -> str | None:
    """从 SeriesDescription 后缀推断相位编码方向。

    本数据集的 BIDS JSON 没有 PhaseEncodingDirection，只能靠描述里的 _AP/_PA。
    返回 mrtrix/fsl 风格的方向串（'j' 表示 AP，'j-' 表示 PA），不认识则 None。
    """
    d = (desc or '').upper()
    if d.endswith('_AP') or '_AP_' in d or d.endswith('AP'):
        return 'j'
    if d.endswith('_PA') or '_PA_' in d or d.endswith('PA'):
        return 'j-'
    return None


def _readout_seconds(meta: dict) -> float | None:
    """按优先级取总读出时间（秒）。"""
    for k in ('EstimatedTotalReadoutTime', 'TotalReadoutTime'):
        v = meta.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    # 回退：用有效回波间距与重建矩阵推算
    ees = meta.get('EstimatedEffectiveEchoSpacing')
    npe = meta.get('ReconMatrixPE') or meta.get('AcquisitionMatrixPE')
    if isinstance(ees, (int, float)) and isinstance(npe, (int, float)) and npe > 1:
        return float((npe - 1) * ees)
    return None


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


def collect_protocol_readout(raw: Path) -> dict:
    """统计全数据集的读出时间，供缺失时做**协议级**回退。

    为什么需要（实测本数据集）
    --------------------------
    231 个有 DWI 的被试里，只有 12 例的 `run-01` JSON 自带读出时间；
    154 例可从反向 b0 借用；**65 例（全部是随访 `*fu`）两处都没有**。

    但采集协议是**统一**的 —— 全部 231 例的 `run-01` 参数逐项相同：

        SeriesDescription = DTI_64dir_b1500_AP
        ReconMatrixPE = 128, AcquisitionMatrixPE = 126, EchoTrainLength = 53
        PhaseEncodingAxis = j, MRAcquisitionType = 2D

    因此对这 65 例使用"有值被试的**中位数**"作为协议常数，是有依据的回退，
    而不是猜测。该值会被写进每例的溯源 JSON。

    返回值同时给出分级计数，便于在报告里如实说明有多少例用了回退。
    """
    import statistics as st
    vals: list[float] = []
    lvl = {1: 0, 2: 0, 3: 0, 4: 0}
    for sub in sorted(p.name for p in raw.iterdir()
                      if p.is_dir() and p.name.startswith('sub-')):
        ddir = raw / sub / 'dwi'
        if not ddir.is_dir():
            continue
        metas = {}
        for f in sorted(ddir.glob('*_dwi.json')):
            try:
                metas[f.name] = json.loads(f.read_text(encoding='utf-8'))
            except Exception:                           # noqa: BLE001
                continue
        if not metas:
            continue
        main = next((m for n, m in metas.items() if 'run-01' in n), None)
        revs = [m for n, m in metas.items() if 'run-01' not in n]
        got, level = None, 4
        if main:
            v = _readout_seconds(main)
            if v:
                got, level = v, 1
        if got is None:
            for m in revs:
                v = _readout_seconds(m)
                if v:
                    got, level = v, 2
                    break
        if got is None:
            got = _compute_readout_from_ees(main, revs)
            if got:
                level = 3
        lvl[level] += 1
        if got:
            vals.append(got)
    med = st.median(vals) if vals else None
    return {'median': med, 'n_with': len(vals), 'n_total': sum(lvl.values()),
            'levels': lvl, 'min': min(vals) if vals else None,
            'max': max(vals) if vals else None,
            'n_unique': len({round(v, 6) for v in vals})}


def _compute_readout_from_ees(main: dict | None, revs: list[dict]) -> float | None:
    """用 (ReconMatrixPE-1) * EstimatedEffectiveEchoSpacing 推算读出时间。"""
    for m in ([main] if main else []) + list(revs):
        if not m:
            continue
        ees = m.get('EstimatedEffectiveEchoSpacing')
        npe = m.get('ReconMatrixPE') or m.get('AcquisitionMatrixPE')
        if isinstance(ees, (int, float)) and isinstance(npe, (int, float)) and npe > 1:
            return float((npe - 1) * ees)
    return None


def pick_dwi(raw: Path, sub: str, protocol: dict | None = None) -> dict | None:
    """定位主 DWI 与反向相位 b0，并解析相位编码元数据。

    返回 dict 或 None（缺数据时）。所有推断都带来源标注，便于溯源。
    """
    ddir = raw / sub / 'dwi'
    if not ddir.is_dir():
        return None

    runs: dict[str, dict] = {}
    for nii in sorted(ddir.glob('*_dwi.nii.gz')):
        stem = nii.name[:-len('.nii.gz')]
        bval = ddir / f'{stem}.bval'
        bvec = ddir / f'{stem}.bvec'
        js = ddir / f'{stem}.json'
        if not (bval.exists() and bvec.exists()):
            continue
        meta = _read_json(js)
        desc = str(meta.get('SeriesDescription', ''))
        vals = [v for v in bval.read_text().split() if v]
        try:
            bvals = [float(v) for v in vals]
        except ValueError:
            continue
        run_tag = None
        for tok in stem.split('_'):
            if tok.startswith('run-'):
                run_tag = tok
        runs[run_tag or 'run-01'] = {
            'nii': nii, 'bval': bval, 'bvec': bvec, 'meta': meta,
            'desc': desc, 'bvals': bvals,
            'n_vol': len(bvals),
            'pe': _pe_from_desc(desc),
            'readout': _readout_seconds(meta),
        }

    if not runs:
        return None

    # 主序列 = 体积数最多的那个（本数据为 run-01，65 体积）
    main_tag = max(runs, key=lambda k: runs[k]['n_vol'])
    main = dict(runs[main_tag])
    main['run'] = main_tag

    # 反向相位 b0：收集**全部**单体积且方向与主序列相反的 run。
    # 本数据集每个被试都有两个（DTI_b01_PA = run-02、DTI_b02_PA = run-03），
    # 把它们**串联成 4D** 交给 -se_epi 有两个好处：
    #   1) 满足 dwifslpreproc -rpe_pair 对"4D 图像"的硬性要求
    #      （单体积 3D 会报 "image provided using -se_epi must be a 4D image"）；
    #   2) topup 用 2 个反向 b0 估计畸变场，比只用 1 个更稳。
    revs = []
    for tag, r in runs.items():
        if tag == main_tag:
            continue
        if r['n_vol'] > 3:
            continue
        if max(r['bvals']) > 50:
            continue
        if r['pe'] and r['pe'] != main['pe']:
            revs.append({**r, 'run': tag})
    revs.sort(key=lambda x: x['run'])
    main['revs'] = revs
    main['rev'] = revs[0] if revs else None      # 向后兼容单数形式

    # 读出时间补齐（四级链，每级都记录来源，绝不静默猜测）
    notes = []
    main_level = 1 if main['readout'] is not None else None
    main_note = 'L1 from run-01 JSON' if main_level else ''
    # L2：从任一反向 b0 借用（同协议下读出时间应相同）
    donor = next((r for r in revs if r.get('readout') is not None), None)
    if main['readout'] is None and donor is not None:
        main['readout'] = donor['readout']
        main_level = 2
        main_note = (f"L2 borrowed from reverse b0 {donor['run']} "
                     f"(same protocol: EchoTrainLength="
                     f"{main['meta'].get('EchoTrainLength')})")
        notes.append(main_note)
    if main['readout'] is None:
        computed = _compute_readout_from_ees(
            main['meta'], [main['meta']] + [r['meta'] for r in revs])
        if computed:
            main['readout'] = computed
            main_level = 3
            main_note = ('L3 computed as (ReconMatrixPE-1)*'
                         'EstimatedEffectiveEchoSpacing')
            notes.append(main_note)
    if main['readout'] is None and protocol and protocol.get('median'):
        main['readout'] = float(protocol['median'])
        main_level = 4
        main_note = (f"L4 dataset-wide protocol median {protocol['median']:.7f}s "
                     f"(from {protocol['n_with']} subjects; protocol verified "
                     f"identical: DTI_64dir_b1500_AP, ReconMatrixPE=128, "
                     f"EchoTrainLength=53)")
        notes.append('L4 protocol median used for main sequence')
    # 反向 b0 若缺读出时间，用主序列的值补（topup 需要同一读出时间）
    for r in revs:
        if r.get('readout') is None and main['readout'] is not None:
            r['readout'] = main['readout']
    main['notes'] = notes
    main['readout_level'] = main_level
    main['readout_note'] = main_note
    return main


# ============================================================ 容器执行
def _docker_common(work: Path, stage_root: Path) -> list[str]:
    return [
        'docker', 'run', '--rm', '-u', '0:0',
        '-v', f'{stage_root}:/data',
        '-v', f'{work}:/out',
        '-e', 'FSLDIR=/opt/fsl',
        '-e', 'OMP_NUM_THREADS=4',
        '--entrypoint', 'bash', IMAGE,
    ]


def _run_script(work: Path, stage_root: Path, name: str, body: str,
                timeout_h: float) -> tuple[bool, str]:
    """在容器内跑一段 bash，逐步显式报错（不用 set -e）。

    刻意不用 `set -e`：它会在首个失败处静默退出，并把 `echo $?` 的退出码
    变 0，使真正原因不可见（A3 为此多花了数轮排查）。
    """
    scripts = stage_root / 'scripts'
    scripts.mkdir(parents=True, exist_ok=True)
    sp = scripts / f'{name}.sh'
    # ⚠️ 必须显式用 LF 写入。Windows 上 `Path.write_text()` 默认会做
    # 换行转换变成 CRLF，bash 于是报
    #   `$'true\r': command not found` / `cd: $'/out\r': No such file`
    # 这类完全看不出根因的错误（本项目实测踩过）。
    with open(sp, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(body)

    cmd = _docker_common(work, stage_root) + [f'/data/scripts/{name}.sh']
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


PRELUDE = """export FSLDIR=/opt/fsl
# ⚠️ 必须是 NIFTI 而不是 NIFTI_GZ：FSL 的 `-out foo.nii` 在 NIFTI_GZ 下会**追加 .gz**，
# 实际写出 `foo.nii.gz`，于是脚本里对 `foo.nii` 的存在性检查必然失败
# （本项目在 5TT 重采样这一步踩过：产出 5tt_b0.nii.gz，检查 5tt_b0.nii 报了假失败）。
# 用 NIFTI 后文件名与脚本所写完全一致，行为可预测。
export FSLOUTPUTTYPE=NIFTI
. /opt/fsl/etc/fslconf/fsl.sh 2>/dev/null || true
# ⚠️ 必须**追加** `/opt/fsl/bin`，不能前置。
# `fsl.sh` 只把 `/opt/fsl/share/fsl/bin` 加进 PATH（那里有 topup/applytopup/bet/fast/eddy_cpu，
# 但**没有** flirt/eddy/fslmaths）。真正齐全的是 `/opt/fsl/bin`，可它同时含
# FSL 自带的 python——一旦前置就会抢在系统 python 之前。
# 而 MRtrix3 的 `dwi2response` 等脚本 shebang 是 `#!/usr/bin/env python`，
# 被 FSL 的 python 接管后，其 `locale.getpreferredencoding()` 返回 `cp437`，
# 该 python 构建又**不含 cp437 codec**，于是报
#   LookupError: unknown encoding: cp437
# 定位过程：不 source fsl.sh 时 rc=0；source 后必崩；把 /opt/fsl/bin 改为追加后恢复。
# 实测修复后：python -> /usr/local/bin/python（UTF-8），flirt/eddy/topup 仍全部可用，
# dwi2response rc=0，WM 响应函数第一本征值约 1063（合理）。
export PATH="$PATH:/opt/fsl/bin"
cd /out
"""


# ============================================================ 各阶段
def stage_prep(sub: str, dwi: dict, work: Path, stage_root: Path,
               threads: int, timeout_h: float) -> tuple[bool, str]:
    """降噪 + Gibbs 去环 + 转 mif。"""
    sdir = work / sub
    sdir.mkdir(parents=True, exist_ok=True)
    out = sdir / 'dwi_preproc.mif'
    if looks_complete(out):
        return True, 'already-done'

    main_rel = dwi['main_rel']
    body = PRELUDE + f"""
echo "[1/3] mrconvert"
mrconvert /data/{main_rel} /out/{sub}/dwi_raw.mif -fslgrad /data/{dwi['bvec_rel']} /data/{dwi['bval_rel']} -datatype float32
rc=$?
[ $rc -ne 0 ] && {{ echo "FATAL mrconvert rc=$rc"; exit 11; }}

echo "[2/3] dwidenoise"
dwidenoise /out/{sub}/dwi_raw.mif /out/{sub}/dwi_den.mif -noise /out/{sub}/noise.mif -nthreads {threads}
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/dwi_den.mif ]; then
  echo "FATAL dwidenoise rc=$rc"
  exit 12
fi

echo "[3/3] mrdegibbs"
mrdegibbs /out/{sub}/dwi_den.mif /out/{sub}/dwi_preproc.mif -axes 0,1 -nthreads {threads}
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/dwi_preproc.mif ]; then
  echo "FATAL mrdegibbs rc=$rc"
  exit 13
fi
rm -f /out/{sub}/dwi_raw.mif /out/{sub}/dwi_den.mif
echo OK
"""
    return _run_script(work, stage_root, f'{sub}_prep', body, timeout_h)


def stage_preproc(sub: str, dwi: dict, work: Path, stage_root: Path,
                  threads: int, timeout_h: float) -> tuple[bool, str]:
    """topup + eddy（经 dwifslpreproc），做磁化率畸变与涡流/头动校正。"""
    sdir = work / sub
    out = sdir / 'dwi_eddy.mif'
    if looks_complete(out):
        return True, 'already-done'

    revs = dwi.get('revs') or ([dwi['rev']] if dwi.get('rev') else [])
    if not revs:
        return False, '缺少反向相位 b0，无法做 topup（本数据集应存在 run-02/03）'
    pe = dwi['pe']
    readout = dwi['readout']
    if readout is None:
        return False, '无法确定总读出时间（JSON 缺字段且无法推算）'

    # 提取每个反向 b0 为独立 mif，随后沿第 4 轴串联成 4D
    rev_blocks = []
    for i, r in enumerate(revs, 1):
        rev_blocks.append(
            f'mrconvert /data/{r["nii_rel"]} /out/{sub}/b0_rev_{i}.mif '
            f'-fslgrad /data/{r["bvec_rel"]} /data/{r["bval_rel"]} -datatype float32\n'
            f'rc=$?\n'
            f'[ $rc -ne 0 ] && {{ echo "FATAL mrconvert rev{i} rc=$rc"; exit 21; }}\n'
        )
    cat_inputs = ' '.join(f'/out/{sub}/b0_rev_{i}.mif' for i in range(1, len(revs) + 1))

    body = PRELUDE + f"""
echo "[1/3] 提取 {len(revs)} 个反向 b0"
{''.join(rev_blocks)}
echo "[2/3] 串联成 4D（-rpe_pair 要求 -se_epi 必须是 4D）"
mrcat {cat_inputs} /out/{sub}/se_epi.mif -axis 3
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/se_epi.mif ]; then
  echo "FATAL mrcat rc=$rc"
  exit 23
fi
mrinfo /out/{sub}/se_epi.mif | head -4

echo "[3/3] dwifslpreproc (topup + eddy)"
# `--niter=3`：eddy 的默认迭代次数是 5。实测本机 eddy_cpu（无 GPU）默认设置下
# 单例耗时 85 分钟，是全流程最大的瓶颈。文献与 FSL 实践中 niter=3 已能收敛
# （eddy 的位移/涡流参数通常在第 2–3 轮即稳定），可把该步压到约 50 分钟。
# 该设置对**全样本统一**使用，保证被试间口径一致。
dwifslpreproc /out/{sub}/dwi_preproc.mif /out/{sub}/dwi_eddy.mif \\
  -pe_dir {pe} -rpe_pair -se_epi /out/{sub}/se_epi.mif \\
  -readout_time {readout:.8f} -nthreads {threads} \\
  -eddy_options " --niter=3 " -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/dwi_eddy.mif ]; then
  echo "FATAL dwifslpreproc rc=$rc"
  exit 22
fi
rm -f /out/{sub}/b0_rev_*.mif /out/{sub}/se_epi.mif
echo OK
"""
    return _run_script(work, stage_root, f'{sub}_preproc', body, timeout_h)


def stage_fod(sub: str, work: Path, stage_root: Path, threads: int,
              timeout_h: float) -> tuple[bool, str]:
    """掩码 + 响应函数 + 多壳 CSD（dhollander）。"""
    sdir = work / sub
    out = sdir / 'wm_fod.mif'
    if looks_complete(out):
        return True, 'already-done'

    body = PRELUDE + f"""
echo "[1/4] dwi2mask"
dwi2mask /out/{sub}/dwi_eddy.mif /out/{sub}/mask.mif -nthreads {threads} -force
rc=$?
[ $rc -ne 0 ] && {{ echo "FATAL dwi2mask rc=$rc"; exit 31; }}

echo "[2/4] dwibiascorrect"
dwibiascorrect ants /out/{sub}/dwi_eddy.mif /out/{sub}/dwi_bc.mif -mask /out/{sub}/mask.mif -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/dwi_bc.mif ]; then
  echo "FATAL dwibiascorrect rc=$rc"
  exit 32
fi

echo "[3/4] dwi2response dhollander"
dwi2response dhollander /out/{sub}/dwi_bc.mif /out/{sub}/resp_wm.txt /out/{sub}/resp_gm.txt /out/{sub}/resp_csf.txt \\
  -mask /out/{sub}/mask.mif -voxels /out/{sub}/resp_voxels.mif -nthreads {threads} -force
rc=$?
[ $rc -ne 0 ] && {{ echo "FATAL dwi2response rc=$rc"; exit 33; }}

echo "[4/4] dwi2fod msmt_csd"
dwi2fod msmt_csd /out/{sub}/dwi_bc.mif /out/{sub}/resp_wm.txt /out/{sub}/wm_fod.mif \\
  /out/{sub}/resp_gm.txt /out/{sub}/gm_fod.mif /out/{sub}/resp_csf.txt /out/{sub}/csf_fod.mif \\
  -mask /out/{sub}/mask.mif -nthreads {threads} -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/wm_fod.mif ]; then
  echo "FATAL dwi2fod rc=$rc"
  exit 34
fi
echo OK
"""
    return _run_script(work, stage_root, f'{sub}_fod', body, timeout_h)


def stage_5tt(sub: str, work: Path, stage_root: Path, a3_seg_rel: str,
              a3_orig_rel: str, timeout_h: float) -> tuple[bool, str]:
    """生成 5TT 并对齐到 DWI 空间（ACT 束约束必需）。

    为什么不是直接 `5ttgen freesurfer <subject_dir>`
    -----------------------------------------------
    A3 用的是 FastSurfer `--seg_only`，输出里**没有皮层表面**
    （只有 `surf/callosum.surf`，也没有 `label/`），因此 5ttgen 的
    "subject 目录" 用法走不通。实测 `5ttgen freesurfer` 的 input 可以是
    **任意分割图像**，直接吃 `aseg.auto.mgz` 即可（已在本数据上验证产出
    143x162x130x5 的合法 5TT）。

    为什么还要配准
    --------------
    ACT（`tckgen -act`）要求 5TT 与 FOD **同一空间**。5TT 来自 aseg，处于
    FastSurfer 的 conformed 空间；而 FOD 在涡流校正后的 DWI 空间。因此需要：
      1. 从预处理后的 DWI 提取 b0 作为参考；
      2. 用 FLIRT 把 conformed T1（`orig.mgz`，与 aseg 同网格）刚性配准到 b0；
      3. 用同一变换把 5TT 最近邻重采样到 b0 网格。
    用 `orig.mgz` 而不是原始 0.5mm T1w，是因为它与 aseg 同网格，省一次重采样。
    """
    out = work / sub / '5tt.mif'
    if looks_complete(out):
        return True, 'already-done'

    body = PRELUDE + f"""
echo "[1/4] 5ttgen freesurfer (直接吃 aseg)"
5ttgen freesurfer /data/{a3_seg_rel} /out/{sub}/5tt_native.mif -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/5tt_native.mif ]; then
  echo "FATAL 5ttgen rc=$rc"
  exit 41
fi

echo "[2/4] 提取 b0 作为配准参考"
dwiextract /out/{sub}/dwi_bc.mif -bzero - | mrconvert - /out/{sub}/b0.nii -datatype float32 -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/b0.nii ]; then
  echo "FATAL dwiextract rc=$rc"
  exit 42
fi

echo "[3/5] 转换 .mgz -> NIfTI，并由 aseg 生成脑掩码做脑提取"
# ⚠️ 为什么必须脑提取：
#   conformed T1 是 256³ 的**整头**（含颈部、眼球、头皮，世界 z 跨 255 mm），
#   而 DWI 只是 128×128×68 @2mm 的**平板**（世界 z 跨仅 137 mm）。
#   把整头直接配到平板上时，FLIRT 会收敛到错误位置（实测 z 偏约 19 mm，
#   表现为 5TT 只有 48% 落在脑掩码内，ACT 把流线在 18 mm 处就切断）。
#   这与 A3 里 mri_coreg 因颈部偏 10.7 mm 是同一类问题 —— 脑提取后即可。
#   aseg 是 256³、与 orig.mgz 完全同网格，所以直接当掩码用。
#
# ⚠️ 本镜像的 FSL 6.0.7.7 **不支持读 .mgz**（Image Exception #63），
# 且 flirt 读失败时**仍返回 rc=0**，所以这里统一先转成 .nii，
# 后续所有 FSL 步骤都只碰 .nii。
mrconvert /data/{a3_orig_rel} /out/{sub}/orig_t1.nii -datatype float32 -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/orig_t1.nii ]; then
  echo "FATAL mrconvert orig.mgz -> nii rc=$rc"
  exit 46
fi

mrconvert /data/{a3_seg_rel} /out/{sub}/aseg.nii -datatype int16 -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/aseg.nii ]; then
  echo "FATAL mrconvert aseg rc=$rc"
  exit 47
fi
mrthreshold /out/{sub}/aseg.nii -abs 0.5 /out/{sub}/brainmask.nii -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/brainmask.nii ]; then
  echo "FATAL mrthreshold brainmask rc=$rc"
  exit 48
fi
mrcalc /out/{sub}/orig_t1.nii /out/{sub}/brainmask.nii -mult \
  /out/{sub}/orig_t1_brain.nii -datatype float32 -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/orig_t1_brain.nii ]; then
  echo "FATAL 脑提取 rc=$rc"
  exit 49
fi

echo "[4/5] FLIRT 脑提取后的 T1 -> b0 (6 dof)"
flirt -in /out/{sub}/orig_t1_brain.nii -ref /out/{sub}/b0.nii \\
      -omat /out/{sub}/t1_to_b0.mat -dof 6 -cost normmi -searchcost normmi
rc=$?
if [ ! -s /out/{sub}/t1_to_b0.mat ]; then
  echo "FATAL flirt 未产出 t1_to_b0.mat (rc=$rc)"
  exit 43
fi

echo "[5/5] 5TT 先补到 T1 网格，再用同一矩阵重采样到 DWI 空间"
# ⚠️ 5ttgen 的输出是**裁剪过的**（本数据 147x138x179，而 T1 是 256³）。
# FLIRT 的 .mat 是相对**输入图像自身网格**定义的，把用 256³ 算出的矩阵
# 直接套到 147x138x179 的 5TT 上会错位。所以必须先 `mrgrid -template`
# 把 5TT 补齐到 orig_t1.nii 的网格（两者世界坐标一致、体素 1mm、无旋转，
# 因此这是纯填充，最近邻无插值误差），再施加矩阵。
mrgrid /out/{sub}/5tt_native.mif regrid -template /out/{sub}/orig_t1.nii \\
  -interp nearest /out/{sub}/5tt_conf.mif -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/5tt_conf.mif ]; then
  echo "FATAL mrgrid 5TT 补网格 rc=$rc"
  exit 50
fi

mrconvert /out/{sub}/5tt_conf.mif /out/{sub}/5tt_conf.nii -datatype float32 -force
flirt -in /out/{sub}/5tt_conf.nii -ref /out/{sub}/b0.nii \\
      -applyxfm -init /out/{sub}/t1_to_b0.mat -interp nearestneighbour \\
      -out /out/{sub}/5tt_b0.nii
rc=$?
if [ ! -s /out/{sub}/5tt_b0.nii ]; then
  echo "FATAL 重采样 5TT 未产出 (rc=$rc)"
  exit 44
fi
mrconvert /out/{sub}/5tt_b0.nii /out/{sub}/5tt.mif -datatype float32 -force
5ttcheck /out/{sub}/5tt.mif
rc=$?
if [ $rc -ne 0 ]; then
  echo "FATAL 5ttcheck rc=$rc（5TT 有重叠/空洞，ACT 结果不可信）"
  exit 45
fi

# ---- 对齐质量自检：脑掩码内五通道和的均值应接近 1 ----
# ⚠️ 5tt.mif 是 4D（5 个组织通道）。`mrstats -output mean` 对 4D 图像会**每个通道
# 输出一行**，`head -1` 只会拿到第一个通道（CGV ≈ 0.4），从而误判为配准失败。
# 正确做法：把 5 个通道的均值**求和**，得到"脑掩码内被 5TT 覆盖的比例"，
# 合格区间约 0.75–1.0；<0.6 说明配准错位。
# 注意：本脚本整体是 Python f-string，所以 bash/awk 里的花括号必须写成 {{{{ }}}}。
Q=$(mrstats /out/{sub}/5tt.mif -output mean -mask /out/{sub}/mask.mif 2>/dev/null \
    | awk '{{s+=$1}} END{{printf "%.4f", s}}')
echo "  5TT 五通道和在 DWI 脑掩码内的均值 = $Q （合格约 0.75-1.0；<0.6 说明配准失败）"
if ! awk -v q="$Q" 'BEGIN{{exit !(q+0 >= 0.6)}}'; then
  echo "FATAL 5TT 配准质量不合格（均值 $Q < 0.6）"
  exit 51
fi
echo OK
"""
    return _run_script(work, stage_root, f'{sub}_5tt', body, timeout_h)


def stage_tract(sub: str, work: Path, stage_root: Path, threads: int,
                n_streamlines: int, timeout_h: float) -> tuple[bool, str]:
    """纤维追踪 + SIFT2 加权。

    为什么**不用 ACT、也不用 -seed_dynamic**（本数据实测结论）
    -------------------------------------------------------
    在 sub-010 上做了三组公平对照（各 20 万流线，其余参数相同）：

      | 组合                    | 平均长度 | 中位  | 分区覆盖 | 非零有向边 |
      |-------------------------|---------|-------|---------|-----------|
      | ACT + -seed_dynamic     | 27.6 mm | 16.6  | 194/200 | 4781      |
      | ACT + -seed_image mask  | 42.7 mm | 34.5  | 194/200 | 6051      |
      | 无 ACT + -seed_image    | 57.7 mm | 44.6  | 198/200 | 6232      |

    1. `-seed_dynamic wm_fod.mif` 只在 FOD 峰值（深部白质核心）撒种子，
       本数据下平均流线仅 27.6 mm，远低于全脑追踪应有的 55-80 mm。
    2. ACT 进一步截短流线（57.7 -> 42.7 mm）。根源是 2mm 各向同性 DWI 的
       部分容积效应：5TT 的灰白质界面与 FOD 的白质边界在 2mm 网格上系统性
       错位，ACT 会把仍然有效的流线判为"进入灰质/CSF"而终止。
       （5TT 自身已修正对齐，脑掩码内五通道和均值 0.777，5ttcheck 通过；
         残余误差来自分辨率，不是配准。）
    3. 因此采用**全脑种子 + 无 ACT**，再用 SIFT2 校正流线密度偏差。
       这是临床 DWI 研究中常见且被接受的配置（ACT 是可选增强，不是必需）。

    ⚠️ 若将来换用 1.25mm 或更高分辨率的 DWI，应重新评估是否启用 ACT。
    """
    sdir = work / sub
    tck = sdir / 'tracks.tck'
    if looks_complete(tck):
        # SIFT2 可能还没做
        if (sdir / 'sift2_weights.txt').exists():
            return True, 'already-done'

    body = PRELUDE + f"""
echo "[1/2] tckgen (全脑种子，无 ACT)"
tckgen /out/{sub}/wm_fod.mif /out/{sub}/tracks.tck \\
  -seed_image /out/{sub}/mask.mif -backtrack -maxlength 250 \\
  -select {n_streamlines} -nthreads {threads} -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/tracks.tck ]; then
  echo "FATAL tckgen rc=$rc"
  exit 51
fi

echo "[2/2] tcksift2"
tcksift2 /out/{sub}/tracks.tck /out/{sub}/wm_fod.mif /out/{sub}/sift2_weights.txt \\
  -nthreads {threads} -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/sift2_weights.txt ]; then
  echo "FATAL tcksift2 rc=$rc"
  exit 52
fi
echo OK
"""
    return _run_script(work, stage_root, f'{sub}_tract', body, timeout_h)


def stage_connectome(sub: str, work: Path, stage_root: Path, atlas_rel: str,
                     threads: int, timeout_h: float,
                     t1w_rel: str = '') -> tuple[bool, str]:
    """生成 SC 矩阵（流线数 + FA 加权）。"""
    sdir = work / sub
    sl = sdir / 'connectome_streamlines.csv'
    fa = sdir / 'connectome_fa.csv'
    if sl.exists() and fa.exists():
        return True, 'already-done'

    body = PRELUDE + f"""
echo "[1/4] 拟合扩散张量（tensor2metric 的必需前置）"
# ⚠️ `tensor2metric` 的输入必须是**张量图像**（dwi2tensor 的输出，6 个分量），
# 不能直接喂 DWI。首版漏了这一步，把 dwi_bc.mif 传给 tensor2metric，报
#   tensor2metric: [ERROR] input tensor image is not a valid tensor
# 而且该错误发生在 eddy/FOD/5TT/tckgen 全部跑完之后，白耗约 2 小时/例。
# 已把 dwi2tensor 显式加入本阶段的第一步。
dwi2tensor /out/{sub}/dwi_bc.mif /out/{sub}/tensor.mif \\
  -mask /out/{sub}/mask.mif -nthreads {threads} -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/tensor.mif ]; then
  echo "FATAL dwi2tensor rc=$rc"
  exit 60
fi

echo "[2/4] 计算 FA"
tensor2metric /out/{sub}/tensor.mif -fa /out/{sub}/fa.mif \\
  -mask /out/{sub}/mask.mif -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/fa.mif ]; then
  echo "FATAL tensor2metric rc=$rc"
  exit 61
fi

echo "[3/5] 把图集显式映射进 DWI 空间"
# 为什么必须做，以及为什么必须是"原始 T1w 直接配准 b0"
# ------------------------------------------------
# 原做法把 **T1 空间**的图集直接交给 tck2connectome，MRtrix 按**世界坐标**
# 把它叠到流线上。前提是被试的 T1 与 DWI 世界坐标系一致；不一致时图集整体
# 错位，分区覆盖率确定性偏低，而且**不报任何错**。
#
# ⚠️ 曾经写过一版"图集 -> orig_t1 网格 -> 用 t1_to_b0.mat 搬到 b0"的两步法，
# 结果是**错的**：A3 的 Schaefer200_in_<sub>.nii.gz 在**原始 T1w** 网格
# （如 340x512x512 / 0.5mm），而 orig_t1.nii 来自 FastSurfer 的 **conformed**
# orig.mgz（256^3 / 1mm），两者世界原点并不相同。用世界坐标把图集重采样到
# conformed 网格，等于在第一步就引入新的错位。
# 实测同一改动方向相反：sub-029 54->121（变好），sub-109 176->106（变差）。
#
# 正确做法（sub-029 实测 54 -> 199/200）：
#   (a) 原始 T1w 降采样到 1mm（**保持世界坐标**，只为提速约 8 倍；
#       0.5mm 全网格直接做 FLIRT 要 30 分钟以上）；
#   (b) 1mm T1w 刚性配准到 b0；
#   (c) 用该矩阵把**原生网格**的图集重采样到 b0 空间；
#   (d) 图集转 uint32 —— flirt 输出是浮点，tck2connectome 会拒绝。
# 全过程不经过 conformed 网格。
ATLAS=/data/{atlas_rel}
MAPMODE=world
if [ -n "{t1w_rel}" ] && [ -s /data/{t1w_rel} ] && [ -s /out/{sub}/b0.nii ]; then
  mrgrid /data/{t1w_rel} regrid /out/{sub}/_t1w_1mm.nii -voxel 1.0 -interp cubic -force > /dev/null 2>&1
  if [ -s /out/{sub}/_t1w_1mm.nii ]; then
    flirt -in /out/{sub}/_t1w_1mm.nii -ref /out/{sub}/b0.nii \\
      -omat /out/{sub}/t1w_to_b0.mat -dof 6 -cost normmi -searchcost normmi > /dev/null 2>&1
    if [ -s /out/{sub}/t1w_to_b0.mat ]; then
      flirt -in /data/{atlas_rel} -ref /out/{sub}/b0.nii -applyxfm \\
        -init /out/{sub}/t1w_to_b0.mat -interp nearestneighbour \\
        -out /out/{sub}/atlas_in_dwi.nii > /dev/null 2>&1
      if [ -s /out/{sub}/atlas_in_dwi.nii ]; then
        mrconvert /out/{sub}/atlas_in_dwi.nii /out/{sub}/atlas_in_dwi_int.mif \\
          -datatype uint32 -force > /dev/null 2>&1
        if [ -s /out/{sub}/atlas_in_dwi_int.mif ]; then
          ATLAS=/out/{sub}/atlas_in_dwi_int.mif
          MAPMODE=explicit
        fi
      fi
    fi
  fi
fi
# 校验：重采样后图集应仍含 200 个标签（丢标签说明映射出错）
if [ "$MAPMODE" = "explicit" ]; then
  NLAB=$(mrstats "$ATLAS" -output count -ignorezero 2>/dev/null | head -1 | tr -d ' ')
  NMAX=$(mrstats "$ATLAS" -output max 2>/dev/null | head -1 | tr -d ' ')
  echo "图集映射=原始T1w直配  非零体素=$NLAB  最大标签=$NMAX"
else
  echo "图集映射=世界坐标（回退：t1w/b0 缺失或重采样失败）"
fi

echo "[4/5] tck2connectome 流线数"
tck2connectome /out/{sub}/tracks.tck "$ATLAS" /out/{sub}/connectome_streamlines.csv \\
  -tck_weights_in /out/{sub}/sift2_weights.txt -zero_diagonal -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/connectome_streamlines.csv ]; then
  echo "FATAL tck2connectome (streamlines) rc=$rc"
  exit 62
fi

echo "[5/5] tck2connectome FA 加权"
# ⚠️ 本镜像的 MRtrix3 3.0.8 **没有 `tck2connectome -metric` 选项**
# （报 `unknown option "-metric"`）。官方推荐的分两步做法（见 `tck2connectome -help`
# 的 -scale_file 示例原文）：
#   1) tcksample 取每条流线的平均 FA；
#   2) tck2connectome -scale_file 用这些标量给每条流线加权，-stat_edge mean 逐边取均值。
# 同时保留 -tck_weights_in（SIFT2），使 FA 加权与流线数加权口径一致（都经 SIFT2）。
tcksample /out/{sub}/tracks.tck /out/{sub}/fa.mif /out/{sub}/fa_per_streamline.txt \\
  -stat_tck mean -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/fa_per_streamline.txt ]; then
  echo "FATAL tcksample rc=$rc"
  exit 64
fi

tck2connectome /out/{sub}/tracks.tck "$ATLAS" /out/{sub}/connectome_fa.csv \\
  -tck_weights_in /out/{sub}/sift2_weights.txt \\
  -scale_file /out/{sub}/fa_per_streamline.txt -stat_edge mean \\
  -zero_diagonal -force
rc=$?
if [ $rc -ne 0 ] || [ ! -s /out/{sub}/connectome_fa.csv ]; then
  echo "FATAL tck2connectome (FA) rc=$rc"
  exit 63
fi
echo OK
"""
    ok, info = _run_script(work, stage_root, f'{sub}_connectome', body, timeout_h)
    if not ok:
        return ok, info

    # ---- 对称化（宿主机侧，容器内不做）----
    # ⚠️ `tck2connectome` 默认把每条流线**只计入一个方向**（起点节点 -> 终点节点），
    # 因此输出的 A 与 A^T 的支撑集**完全不相交**（实测 |A-A^T|/A = 2.0000，
    # 相对不对称度中位 1.0）。这不是噪声，而是有向计数矩阵的定义使然。
    # 结构连接矩阵的通行口径是**无向**，标准做法是对称化：
    #       S = A + A^T
    # 用求和而非平均，是为了保留"节点 i 与 j 之间共有多少条流线"的物理含义，
    # 也保持总量守恒（A 的总和不变）。对角已由 -zero_diagonal 置零，对称化后仍为 0。
    # 若下游需要平均口径，除以 2 即可，此处不再二次加工。
    n_sym = 0
    for name in ('connectome_streamlines.csv', 'connectome_fa.csv'):
        p = sdir / name
        if not p.exists():
            continue
        try:
            m = np.loadtxt(p, delimiter=',')
        except Exception:                               # noqa: BLE001
            continue
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            continue
        s = m + m.T
        np.fill_diagonal(s, 0.0)
        np.savetxt(p, s, delimiter=',', fmt='%.10g')
        n_sym += 1
    if n_sym:
        info = f'{info}; 已对称化 {n_sym} 个矩阵 (A + A^T)'
    return ok, info


# ============================================================ 导出契约产物
def export_contract(sub: str, work: Path, out_root: Path,
                    meta_extra: dict | None = None,
                    n_parcels: int = N_PARCELS) -> tuple[bool, str]:
    """把 SC 矩阵整理成契约要求的两类文件。

    契约要求（第 2.1 / 2.2 节）：
      a2/sc/<sub>_streamlines.csv   200x200 流线数矩阵
      a2/sc/<sub>_fa.csv            200x200 FA 加权矩阵
      a1_streamlines/<sub>.csv      200     行和（排除对角线），未归一
      a1_fa/<sub>.csv               200     行和（排除对角线），未归一
    """
    sdir = work / sub
    sl_src = sdir / 'connectome_streamlines.csv'
    fa_src = sdir / 'connectome_fa.csv'
    if not (sl_src.exists() and fa_src.exists()):
        return False, '缺少 connectome 矩阵'

    try:
        sl = np.loadtxt(sl_src, delimiter=',')
        fa = np.loadtxt(fa_src, delimiter=',')
    except Exception as exc:                            # noqa: BLE001
        return False, f'矩阵读取失败: {exc}'

    if sl.shape != (n_parcels, n_parcels):
        return False, f'流线矩阵形状 {sl.shape} != ({n_parcels},{n_parcels})'
    if fa.shape != (n_parcels, n_parcels):
        return False, f'FA 矩阵形状 {fa.shape} != ({n_parcels},{n_parcels})'

    # MRtrix 的 tck2connectome 允许对角线非零；契约要求节点强度排除自连接
    sl_off = sl.astype(float).copy()
    fa_off = fa.astype(float).copy()
    np.fill_diagonal(sl_off, 0.0)
    np.fill_diagonal(fa_off, 0.0)
    sl_strength = sl_off.sum(axis=1)
    fa_strength = fa_off.sum(axis=1)

    # ---- 覆盖率质量闸门（防止"跑完了但结果不可用"）----
    # 实测教训：sub-029 的全部阶段都报 OK、单例耗时 148.6 min 完全正常，
    # 但连接组只有 **54/200** 个节点有连接（其余 38 例是 191–200）。
    # 导出日志其实早就写着 `节点强度中位 0.0`，可当时只报告、不拒绝，
    # 于是坏结果被当成成功交付 —— 这类静默失败不做闸门根本发现不了。
    # 阈值取 180：实测分布是双峰的 —— 正常例 191-200，坏例 164 与 54。
    # 首版用 150 太宽松（sub-049 的 164 就漏了过去，靠周期审计才发现）。
    nz_nodes = int((sl_strength > 0).sum())
    if nz_nodes < 180:
        return False, (f'覆盖率过低：仅 {nz_nodes}/{n_parcels} 个分区有连接'
                       f'（阈值 180，正常 191-200）；节点强度中位 {np.median(sl_strength):.4g}')

    (out_root / OUT_SC_SL).mkdir(parents=True, exist_ok=True)
    (out_root / OUT_A1_SL).mkdir(parents=True, exist_ok=True)
    (out_root / OUT_A1_FA).mkdir(parents=True, exist_ok=True)

    def _write_matrix(path: Path, m: np.ndarray) -> None:
        with open(path, 'w', encoding='utf-8', newline='\n') as fh:
            for row in m:
                fh.write(','.join(f'{v:.6g}' for v in row) + '\n')

    def _write_vector(path: Path, v: np.ndarray) -> None:
        with open(path, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write('parcel_index,value\n')
            for i, x in enumerate(v, start=1):
                fh.write(f'{i},{x:.10g}\n')

    _write_matrix(out_root / OUT_SC_SL / f'{sub}_streamlines.csv', sl)
    _write_matrix(out_root / OUT_SC_FA / f'{sub}_fa.csv', fa)
    _write_vector(out_root / OUT_A1_SL / f'{sub}.csv', sl_strength)
    _write_vector(out_root / OUT_A1_FA / f'{sub}.csv', fa_strength)

    meta = {
        'produced_by': 'scripts/run_a1_pipeline.py',
        'date': time.strftime('%Y-%m-%d'),
        'subject': sub,
        'n_parcels': int(n_parcels),
        'metric_streamlines': 'streamline count (SIFT2-weighted), diagonal zeroed',
        'metric_fa': 'mean FA along streamlines (SIFT2-weighted), diagonal zeroed',
        'node_strength': 'row sum of the off-diagonal matrix entries; NOT normalized',
        'notes': ('A1 vectors are derived from the same tractography as the A2 SC '
                  'matrices; no group comparison / covariate regression / '
                  'normalization applied (contract section 0.6)'),
    }
    if meta_extra:
        meta.update(meta_extra)
    for d in (OUT_A1_SL, OUT_A1_FA):
        (out_root / d / f'{sub}.json').write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return True, (f'streamlines 节点强度中位 {np.median(sl_strength):.1f}; '
                  f'FA 加权中位 {np.median(fa_strength):.3f}')


def transfer_to_dest(out_root: Path, dest: Path, subs: list[str]) -> tuple[int, int]:
    """把某被试的产物拷到最终交付目录（按字节数幂等跳过）。"""
    n_ok = n_skip = 0
    rels = []
    for s in subs:
        rels += [
            (OUT_SC_SL, f'{s}_streamlines.csv'),
            (OUT_SC_FA, f'{s}_fa.csv'),
            (OUT_A1_SL, f'{s}.csv'),
            (OUT_A1_FA, f'{s}.csv'),
            (OUT_A1_SL, f'{s}.json'),
            (OUT_A1_FA, f'{s}.json'),
        ]
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
    ap = argparse.ArgumentParser(description='A1/A2-SC：DWI → 结构连接')
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True,
                    help='工作目录（**必须在容器可挂载的盘上，如 C:**）')
    ap.add_argument('--out', type=Path, required=True,
                    help='契约产物目录（同样必须在可挂载盘上）')
    ap.add_argument('--a3-work', type=Path, required=True,
                    help='A3 的 work 目录（复用其 fastsurfer 与 registration 结果）')
    ap.add_argument('--dest', type=Path, default=None, help='最终交付目录（可 F:）')
    ap.add_argument('--stage', default='all',
                    choices=('all', 'prep', 'preproc', 'fod', '5tt', 'tract', 'connectome'))
    ap.add_argument('--subjects', nargs='*', default=None)
    ap.add_argument('--subjects-file', type=Path, default=None)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--streamlines', type=int, default=1_000_000,
                    help='tckgen 目标流线数。默认 1M —— 实测本机 10M 需 >88 min/例，'
                         '1M 约 9 min；对 200 分区配 SIFT2 加权已足够，'
                         '且与 eddy --niter=3 共同把单例压到约 105 min（全量 186 例约 13.6 天）。'
                         '如需更高密度可设 10000000，但全量将达 29.7 天。')
    ap.add_argument('--timeout-h', type=float, default=8.0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--timeit', action='store_true')
    args = ap.parse_args()

    # Docker 不能可靠挂载 F:（实测容器写入宿主看不到），因此 work/out 不能放 F:
    for name, p in (('--work', args.work), ('--out', args.out)):
        if str(p).upper().startswith('F:'):
            log(f'[FATAL] {name} 不能放在 F:（Docker 无法可靠挂载该盘）。'
                f'请用 C: 上的目录，最终产物再用 --dest 拷到 F:')
            return 2

    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    stage_root = ROOT / 'build' / 'docker_staging_a1'
    stage_root.mkdir(parents=True, exist_ok=True)

    only = list(args.subjects) if args.subjects else None
    if args.subjects_file:
        # utf-8-sig 剥掉 UTF-8 BOM：若清单被记事本等编辑过，用 utf-8 读取会把
        # 首行变成 '\ufeffsub-XXX'（幽灵被试），既报警告又静默跳过该被试。
        only = [ln.strip() for ln in
                args.subjects_file.read_text(encoding='utf-8-sig').splitlines()
                if ln.strip()]
        # ⚠️ 空清单必须**拒绝运行**，不能退化成"不过滤 = 全部被试"。
        # 否则一个意外清空的清单文件会静默启动 200+ 例的多天作业。
        if not only:
            log(f'[FATAL] 清单文件为空: {args.subjects_file}')
            log('        空清单不等于"全部被试"；请重新生成清单后再运行。')
            return 2

    subs = discover_subjects(args.raw, only)
    log(f'A1 管线  被试={len(subs)}  阶段={args.stage}  '
        f'流线={args.streamlines/1e6:.0f}M  线程={args.threads}')
    log(f'  raw     : {args.raw}')
    log(f'  work    : {args.work}')
    log(f'  out     : {args.out}')
    log(f'  a3-work : {args.a3_work}')
    if args.dest:
        log(f'  dest    : {args.dest}')

    plan = []
    protocol = collect_protocol_readout(args.raw)
    if protocol['median']:
        log(f'  读出时间协议常数: 中位 {protocol["median"]:.7f}s '
            f'（有值 {protocol["n_with"]}/{protocol["n_total"]}，'
            f'分级 L1={protocol["levels"][1]} L2={protocol["levels"][2]} '
            f'L3={protocol["levels"][3]} 需回退={protocol["levels"][4]}）')
    for s in subs:
        dwi = pick_dwi(args.raw, s, protocol)
        if dwi is None:
            plan.append((s, None, '缺 DWI 主序列或相位编码元数据'))
            continue
        aseg = args.a3_work / 'fastsurfer' / s / 'mri' / 'aseg.auto.mgz'
        atlas = args.a3_work / 'registration' / f'Schaefer200_in_{s}.nii.gz'
        if not aseg.exists():
            plan.append((s, None, '缺 A3 的 FastSurfer aseg（先跑 A3）'))
            continue
        if not atlas.exists():
            plan.append((s, None, '缺 A3 的原生空间图集（先跑 A3）'))
            continue
        plan.append((s, dwi, 'ok'))

    ok_ready = sum(1 for _, d, w in plan if d is not None and w == 'ok')
    log(f'  可处理: {ok_ready}/{len(plan)}')
    if args.dry_run:
        for s, d, w in plan:
            if d is None:
                log(f'  [skip] {s}: {w}')
        return 0

    ok = fail = 0
    t0 = time.time()
    for i, (sub, dwi, why) in enumerate(plan, 1):
        if dwi is None:
            log(f'[{i}/{len(plan)}] {sub}  跳过（{why}）')
            continue
        log(f'[{i}/{len(plan)}] {sub}')
        t_sub = time.time()

        # 准备暂存：容器从 /data 读，所有输入集中到 stage_root/in
        in_dir = stage_root / 'in'
        in_dir.mkdir(parents=True, exist_ok=True)

        def stage_in(src: Path, tag: str) -> str:
            """把输入文件放到暂存目录，返回相对 stage_root 的路径。"""
            dst = in_dir / f'{tag}{src.suffix}'
            if src.suffix == '.gz':
                dst = in_dir / f'{tag}.nii.gz'
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
            return f'in/{dst.name}'

        dwi['main_rel'] = stage_in(dwi['nii'], f'{sub}_main_dwi')
        dwi['bval_rel'] = stage_in(dwi['bval'], f'{sub}_main_dwi')
        # bvec 必须与 nii 同名（mrconvert -fslgrad 用的是显式路径，名字随意，
        # 但保持一致避免混淆）
        dwi['bvec_rel'] = stage_in(dwi['bvec'], f'{sub}_main_dwi')
        # 暂存**全部**反向 b0（本数据集每例 2 个：run-02/run-03），
        # 后续在容器内串联成 4D 交给 -se_epi。
        for i, r in enumerate(dwi.get('revs') or [], 1):
            tag = f'{sub}_rev{i}_dwi'
            r['nii_rel'] = stage_in(r['nii'], tag)
            r['bvec_rel'] = stage_in(r['bvec'], tag)
            r['bval_rel'] = stage_in(r['bval'], tag)
        aseg_rel = stage_in(args.a3_work / 'fastsurfer' / sub / 'mri' / 'aseg.auto.mgz',
                            f'{sub}_aseg')
        orig_rel = stage_in(args.a3_work / 'fastsurfer' / sub / 'mri' / 'orig.mgz',
                            f'{sub}_orig')
        atlas_rel = stage_in(args.a3_work / 'registration' / f'Schaefer200_in_{sub}.nii.gz',
                             f'{sub}_atlas')

        # ---- 为图集映射挑选原始 T1w ----
        # A3 的 Schaefer200_in_<sub>.nii.gz 在**原始 T1w** 网格上，所以必须用
        # 同一网格的 T1w 去配准 b0。有些被试有多个 T1w run（如 sub-029 有
        # run-01/run-02，两者 head position 差 41 mm），选错就等于配到另一张
        # 头上。判据：affine 与图集 affine 最接近的那个 run。
        t1w_rel = ''
        atlas_path = args.a3_work / 'registration' / f'Schaefer200_in_{sub}.nii.gz'
        if atlas_path.exists():
            try:
                import nibabel as _nib
                _a = _nib.load(str(atlas_path))
                _best, _bestd = None, 1e9
                for _p in sorted((args.raw / sub).rglob('*_T1w.nii.gz')):
                    try:
                        _d = float(abs(_nib.load(str(_p)).affine - _a.affine).max())
                    except Exception:
                        continue
                    if _d < _bestd:
                        _best, _bestd = _p, _d
                if _best is not None:
                    t1w_rel = stage_in(_best, f'{sub}_t1w_raw')
                    print(f'    T1w({_best.name}) 与图集 affine 差 {_bestd:.4f} -> 用于图集映射')
            except Exception as _e:
                print(f'    T1w 挑选失败（将回退世界坐标）: {type(_e).__name__}: {_e}')

        steps = []
        if args.stage in ('all', 'prep'):
            steps.append(('降噪', lambda: stage_prep(sub, dwi, args.work, stage_root,
                                                    args.threads, args.timeout_h)))
        if args.stage in ('all', 'preproc'):
            steps.append(('涡流/畸变', lambda: stage_preproc(sub, dwi, args.work, stage_root,
                                                          args.threads, args.timeout_h)))
        if args.stage in ('all', 'fod'):
            steps.append(('FOD', lambda: stage_fod(sub, args.work, stage_root,
                                                   args.threads, args.timeout_h)))
        if args.stage in ('all', '5tt'):
            steps.append(('5TT', lambda: stage_5tt(sub, args.work, stage_root,
                                                  aseg_rel, orig_rel, args.timeout_h)))
        if args.stage in ('all', 'tract'):
            steps.append(('追踪', lambda: stage_tract(sub, args.work, stage_root, args.threads,
                                                     args.streamlines, args.timeout_h)))
        if args.stage in ('all', 'connectome'):
            steps.append(('连接组', lambda: stage_connectome(sub, args.work, stage_root,
                                                            atlas_rel, args.threads,
                                                            args.timeout_h, t1w_rel)))

        good = True
        t_step = time.time()
        stage_times: list[str] = []
        for label, fn in steps:
            t_one = time.time()
            okk, info = fn()
            dt = time.time() - t_one
            stage_times.append(f'{label}={dt:.0f}s')
            # 逐阶段计时：单例总耗时突然翻倍时，只有分阶段数字才能定位是哪一步。
            # 实测 sub-025 用了 9288 s（常规 3875 s），但没有分阶段记录，无法判断
            # 是 eddy、tckgen 还是别的步骤异常。以后每例都记下来。
            if dt >= 600:      # 超过 10 分钟的步骤单独提示
                log(f'    {label}: {"OK" if okk else "FAIL"} '
                    f'({info}) [耗时 {dt/60:.1f} min]')
            else:
                log(f'    {label}: {"OK" if okk else "FAIL"} ({info})')
            if not okk:
                good = False
                break
        if args.timeit and stage_times:
            log(f'     阶段耗时: ' + ' '.join(stage_times))
        if not good:
            fail += 1
            # 清理本例暂存，避免累积
            for f in in_dir.glob(f'{sub}_*'):
                f.unlink(missing_ok=True)
            continue

        if args.stage in ('all', 'connectome'):
            meta_extra = {
                'dwi_run': dwi['run'],
                'phase_encoding_direction': dwi['pe'],
                'total_readout_time_s': round(float(dwi['readout']), 8),
                'readout_source_level': dwi.get('readout_level'),
                'readout_source_note': dwi.get('readout_note'),
                'reverse_pe_runs': [r['run'] for r in (dwi.get('revs') or [])],
                'n_streamlines_target': args.streamlines,
                'atlas': 'Schaefer2018_200Parcels_7Networks_FSLMNI152_1mm '
                         '(A3 native-space warp)',
            }
            okk, info = export_contract(sub, args.work, args.out, meta_extra)
            log(f'    导出: {"OK" if okk else "FAIL"} ({info})')
            if not okk:
                fail += 1
                continue
            ok += 1
            if args.dest:
                n_ok, n_skip = transfer_to_dest(args.out, args.dest, [sub])
                log(f'    传送: 拷贝 {n_ok} / 跳过 {n_skip} -> {args.dest}')
        else:
            ok += 1

        if args.timeit:
            log(f'     单例总耗时 {time.time()-t_sub:.1f}s')
        # 每例结束清理暂存（单例输入约 70 MB，不清理 200 例会累积 14 GB）
        for f in in_dir.glob(f'{sub}_*'):
            f.unlink(missing_ok=True)

    total = time.time() - t0
    log('')
    log(f'完成: 成功 {ok}，失败 {fail}，总耗时 {total/60:.1f} min')
    if ok:
        per = total / max(ok, 1)
        log(f'单例均值 ≈ {per:.1f}s -> 206 例约 {per*206/3600:.1f} h')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
