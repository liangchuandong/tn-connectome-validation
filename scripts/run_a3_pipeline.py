#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段二-A3：从原始 T1w/T2w 生成契约要求的 A3 派生数据。

这是 `docs/derived_data_contract.md` 中"合作方影像预处理"那一段的**本机实现**。
项目原有的 `scripts/run_phenotypes.py` 从**已派生的 A3 向量**开始（下游），
本脚本负责它上游：原始影像 → 分割 → 配准 → 带状取值 → 分区均值。

产物（严格对齐契约第 1、2.3 节）
--------------------------------
    <out>/a3_gwc/sub-XXX.csv      200 长表: parcel_index,value   （灰白质对比度）
    <out>/a3_t1t2/sub-XXX.csv     200 长表: parcel_index,value   （T1w/T2w 比值）
    <out>/a3_gwc/sub-XXX.json     说明文件（produced_by/date/atlas/notes）
    <out>/a3_t1t2/sub-XXX.json

关键方法学决定（均已写入 json 的 notes，供论文追溯）
--------------------------------------------------
1. **分割**：FastSurfer `--seg_only --device cpu`（本机无 GPU）。
   跳过表面重建的理由：A3 只需灰白质交界带的**体素级**取值，用 volume-based
   映射即可满足契约第 4.3 节"surface-based（推荐）或 volume-based，二选一"。
   若日后要改 surface-based，只需换映射环节，取值逻辑不变。

2. **配准**：Schaefer 图集（MNI152 1mm）→ 被试原生空间。
   先做**脑提取**（用 FastSurfer 的 aseg 生成脑掩码），再用 `mri_coreg`（12 dof）
   把纯脑 T1w 配准到 MNI152 参考脑，最后 `mri_vol2vol --lta-inv --nearest`
   把图集重采样到原生网格。
   方向选这个而非"原生→MNI"，避免对每例影像做重采样、保住原始灰度值
   —— GWC 与 T1w/T2w 都是**强度比值**，任何插值都会改变比值。
   （脑提取为什么必需、参考脑为什么必须与图集同网格、以及不用 FSL 的原因，
   见 `_atlas_to_native` 文档与 `docs/a3_pipeline_notes.md`。）

3. **取带方式**：灰白质交界带 = 皮层带 ∩ (WM ∪ 距离 WM 2mm 内)。
   契约第 4.3 节要求"在皮层带内、gray-white boundary 附近取值并写明取带方式"。
   本实现写死为 `--band-mm`（默认 2.0），即从 WM 表面向灰质侧取 2 mm。
   带内的**每一个体素**都参与分区平均——而**不是**只取分割边界那一层，
   因为单层体素受分割误差影响过大。

4. **分区来源**：Schaefer2018 200Parcels 7Networks（与全项目一致），
   由 `scripts/fetch_atlas_volume.py` 冻结到 `data/atlas/`，含 sha256。

5. **不做任何组间比较/协变量回归/标准化**（契约第 6 条）。
   原样导出分区均值，统计由 `scripts/run_phenotypes.py` 统一做。

用法
----
    # 全流程：分割 + 取值（推荐）
    python scripts/run_a3_pipeline.py --raw F:\\ds005713\\raw ^
        --work F:\\ds005713\\work --out F:\\ds005713\\derivatives ^
        --subjects sub-009 sub-01

    # 只取分割（检查 FastSurfer 是否正常，或分离长任务）
    python scripts/run_a3_pipeline.py ... --stage seg

    # 只取值（分割已完成时）
    python scripts/run_a3_pipeline.py ... --stage extract

    # 先看要处理谁、缺什么，不执行
    python scripts/run_a3_pipeline.py --raw F:\\ds005713\\raw --work W --out O --dry-run

    # 单例计时（确定全量批次计划前必做）
    python scripts/run_a3_pipeline.py ... --subjects sub-009 --timeit
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:                                   # pragma: no cover
    print('[FATAL] 缺少 nibabel。请用分析环境：F:\\venvs\\tnwt\\Scripts\\python.exe',
          file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATLAS = ROOT / 'data' / 'atlas' / 'Schaefer2018_200Parcels_7Networks_MNI_1mm.nii.gz'
N_PARCELS = 200
FASTSURFER_IMAGE = 'deepmi/fastsurfer:latest'

# FastSurfer aseg 标签（FreeSurfer 约定）——只用这三个类别
LBL_CTX_LH, LBL_CTX_RH = 3, 42
LBL_WM_LH, LBL_WM_RH = 2, 41
BRAIN_MASK_LABEL = 255


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- FreeSurfer 许可证

# [!] 为什么需要许可证（实测得出的关键结论）
# ------------------------------------------
# FastSurfer 镜像**只打包了免许可证的自家工具**（FastSurferVINN 等 Python/NN 代码），
# 因此分割能跑通。但容器内 **没有 `FS_LICENSE`、也没有
# `/opt/freesurfer/license.txt`**，于是所有需要许可证的 FreeSurfer 命令
# （`mri_coreg`、`mri_robust_register`、`mri_vol2vol`、部分 `mri_convert` 用法）
# 会**跑完但不产出输出文件**，且**退出码为 0**——静默失败，极难察觉。
# 实测表现：`mri_coreg ... --reg /out/x.lta` 打印完初始化参数后直接结束，
# `/out/x.lta` 不存在，`$?` 却是 0。
#
# 取得许可证（学术免费，约 1 分钟）：
#   https://surfer.nmr.mgh.harvard.edu/registration.html
# 把页面/邮件里那段文本存成 license.txt 即可（须保留全部行）。
FS_LICENSE_IN_CONTAINER = '/opt/freesurfer/license.txt'


def license_mount_args(license_file: Path | None) -> list[str]:
    """生成把许可证挂进容器的 docker 参数。

    [!] 不同 FreeSurfer 命令找的**文件名不一样**（实测踩到）：
      * `mri_coreg`     认 `/opt/freesurfer/license.txt`
      * `mri_vol2vol`   认 `/opt/freesurfer/.license`（带点前缀！）
    只挂其中一个，会出现"前一步成功、后一步报
    `FreeSurfer license file /opt/freesurfer/.license not found`"的怪现象。
    因此这里**同时挂两个名字**，并设 `FS_LICENSE` 指向标准名。
    """
    if license_file is None:
        return []
    p = Path(license_file)
    if not p.exists():
        raise FileNotFoundError(f'许可证文件不存在: {p}')
    return [
        '-v', f'{p.resolve()}:/opt/freesurfer/license.txt:ro',
        '-v', f'{p.resolve()}:/opt/freesurfer/.license:ro',
        '-e', f'FS_LICENSE={FS_LICENSE_IN_CONTAINER}',
    ]


def to_docker_path(p: Path, mount_root: Path, inner: str) -> str:
    """把 Windows 路径映射成容器内路径。要求 p 位于 mount_root 之下。"""
    rel = Path(p).resolve().relative_to(Path(mount_root).resolve())
    return '/' + str(Path(inner) / rel).replace('\\', '/').lstrip('/')


def pick_t1w(raw: Path, sub: str) -> Path | None:
    """挑该被试的 T1w。多 run 时优先 run-01（真正的择优在 QC 表里由 cjv 决定，
    本函数只保证确定性选一个可跑通的；若要按 cjv 择优，用 --from-qc 传 QC 表）。"""
    d = raw / sub / 'anat'
    if not d.is_dir():
        return None
    cands = sorted(d.glob('*T1w.nii.gz'))
    if not cands:
        return None
    run01 = [c for c in cands if 'run-01' in c.name]
    return (run01 or cands)[0]


def pick_t2w(raw: Path, sub: str) -> Path | None:
    d = raw / sub / 'anat'
    if not d.is_dir():
        return None
    cands = sorted(d.glob('*T2w.nii.gz'))
    return cands[0] if cands else None


# ---------------------------------------------------------------- FastSurfer

# ---------------------------------------------------------------- 完整性校验
#
# [!] 为什么必须校验而不只是"检查文件存在"
# ----------------------------------------
# 本流程是 5 天量级的长时间批处理，且运行环境**可能被随时断电/关机**
# （用户明确说明无法保持常开）。如果只在写入过程中断电，磁盘上会留下
# **截断的文件**，而 `path.exists()` 对它**返回 True** —— 于是续跑时
# 会把这个损坏文件当成"已完成"而跳过，**静默产出错误结果**。
# 这类错误极难发现（结果看起来正常，只是数值错）。
#
# 因此所有"跳过"判定都必须基于**可读性 + 尺寸下限**，而不是存在性。
# 对于 `.mgz` / `.nii.gz`，nibabel 能打开并读出合理形状即视为完整。

def is_complete_volume(path: Path, min_bytes: int = 10_000,
                       expect_shape: tuple | None = None) -> tuple[bool, str]:
    """判断一个影像文件是否可安全当作"已完成"。

    返回 (是否可信, 说明)。任何异常都视为"不可信"，交由调用方重做。
    """
    if not path.exists():
        return False, 'missing'
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f'stat failed: {exc}'
    if size < min_bytes:
        return False, f'too small ({size} B < {min_bytes} B) — 疑似写入中断'
    try:
        img = nib.load(str(path))
        shape = img.shape
        if len(shape) != 3 or any(s <= 0 for s in shape):
            return False, f'bad shape {shape}'
        if expect_shape is not None and tuple(shape) != tuple(expect_shape):
            return False, f'shape {shape} != expected {expect_shape}'
        # 触发实际读取头部，能拦住截断的 gzip/mgz
        _ = img.header
    except Exception as exc:                          # noqa: BLE001
        return False, f'unreadable: {type(exc).__name__}: {exc}'
    return True, f'{size} B, shape={shape}'


def subject_already_extracted(sub: str, out_root: Path,
                              n_parcels: int = N_PARCELS) -> tuple[bool, str]:
    """判断某被试的契约产物是否**已完整产出**（可整例跳过）。

    [!] 为什么需要这一层（而不只靠配准缓存）
    --------------------------------------
    配准缓存能让续跑跳过分割与配准，但取值阶段仍会重新读入 T1w/T2w（各数十 MB）
    并重算分区均值，约 1 分钟/例。193 例续跑就是约 3 小时的纯浪费。
    这里直接检查两份 CSV 是否都是完整的 `n_parcels` 行，是则整例跳过。

    校验要点：行数必须是 n_parcels + 1（含表头）、表头正确、索引严格 1..N。
    断电留下的半截 CSV 会因行数不足而被判为"未完成"并重做。
    """
    for metric in ('a3_gwc', 'a3_t1t2'):
        p = out_root / metric / f'{sub}.csv'
        if not p.exists():
            return False, f'缺 {metric}/{sub}.csv'
        try:
            with open(p, encoding='utf-8') as fh:
                rows = list(csv.reader(fh))
        except Exception as exc:                      # noqa: BLE001
            return False, f'{metric} 读取失败: {exc}'
        if len(rows) != n_parcels + 1:
            return False, f'{metric} 行数 {len(rows)} != {n_parcels + 1}（疑似截断）'
        if rows[0] != ['parcel_index', 'value']:
            return False, f'{metric} 表头异常: {rows[0]}'
        try:
            idx = [int(r[0]) for r in rows[1:]]
        except (ValueError, IndexError):
            return False, f'{metric} 索引列无法解析'
        if idx != list(range(1, n_parcels + 1)):
            return False, f'{metric} 索引不是 1..{n_parcels} 有序序列'
    return True, 'ok'


def gzip_stream_ok(path: Path, chunk: int = 1 << 22) -> tuple[bool, str]:
    """完整读一遍 gzip 流，判断压缩数据是否完好。

    为什么必须有这一层
    ------------------
    **字节数正确不代表内容完好。** 本项目实测发现 `sub-055` 的两个 T1w
    在清单里记录的大小（270,111 B 与 14,422,419 B）本身就是截断的 ——
    也就是说源文件即损坏，重新下载也无济于事。字节数校验（`verify_download.py`
    第一层）对这种情况完全无效：它比对的是"清单里的期望值"，而期望值本身就是错的。

    这类文件交给 FastSurfer 会浪费约 30 分钟才失败。这里用约 1–3 秒的
    完整解压提前拦下，给出明确原因。

    返回 (是否完好, 说明)。非 .gz 文件一律视为通过（由其他层校验）。
    """
    import gzip as _gzip
    if path.suffix != '.gz':
        return True, 'not gzip'
    try:
        n = 0
        with _gzip.open(str(path), 'rb') as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                n += len(b)
    except Exception as exc:                            # noqa: BLE001
        return False, f'{type(exc).__name__}: {exc}'
    return True, f'{n/1e6:.1f} MB decompressed'


def run_fastsurfer(t1w: Path, work: Path, sub: str,
                   threads: int, timeout_h: float, stage_root: Path,
                   extra: list[str] | None = None) -> tuple[bool, str]:
    """跑 FastSurfer --seg_only。返回 (成功, 说明)。

    产物: <work>/fastsurfer/<sub>/mri/aseg.auto.mgz

    [!] 实测得出的两条硬约束（决定了本函数的形态）
    --------------------------------------------
    1. **Docker Desktop 的 WSL2 后端无法使用可移动盘 F:**
       实测：`-v F:\\...:/io` 挂载点为空；即便容器内写入成功并 `cat` 回来，
       宿主 F: 上也**不会**出现文件（写入不传播）。而 C: 挂载完全正常
       （容器内写入后宿主可读）。
       => 所有容器 I/O **必须**走 C:。结果再用普通文件拷贝传到 F:。
    2. **FastSurfer 默认以 nonroot 运行**，要求 `-u $(id -u):$(id -g)`；
       以 root 运行则必须显式加 `--allow_root`，否则直接报错退出。

    另一条易错点：`--seg_only` 模式下产出的是 **`aseg.auto.mgz`**，
    不是 `aseg.mgz`。用错文件名会把成功判成失败（本项目踩过）。
    """
    subj_dir = work / 'fastsurfer' / sub
    aseg = subj_dir / 'mri' / 'aseg.auto.mgz'
    stage_in = stage_root / 'in' / sub / 'anat'

    # 早退路径也必须清理暂存 —— 否则"续跑已分割被试"时上一轮留下的暂存会一直留着
    # （实测：单例残留 43.9 MB；213 例累计约 10 GB）。
    #
    # 用 is_complete_volume 而非 exists()：断电留下的截断文件必须被识别并重做。
    # 阈值 1 MB 远低于真实 aseg（约 0.5 MB？不 —— 实测 sub-01 的 aseg.auto.mgz 为
    # 0.50 MB 压缩后），所以下限取 100 KB 足够区分"完整"与"几乎为空"。
    _ok, _why = is_complete_volume(aseg, min_bytes=100_000)
    if _ok:
        shutil.rmtree(stage_in.parent, ignore_errors=True)
        return True, 'already-done'
    if aseg.exists():
        log(f'    [warn] 已有分割结果不可信（{_why}），将重做')
        shutil.rmtree(subj_dir, ignore_errors=True)

    subj_dir.mkdir(parents=True, exist_ok=True)
    fsroot = work / 'fastsurfer'
    fsroot.mkdir(parents=True, exist_ok=True)

    # 暂存输入到 stage_root（C:），容器只从这里读。
    #
    # [!] 空间管理（实测踩过）：每例 T1w 约 44–70 MB，若不清理，213 例会累积 10 GB+。
    # 调试期间就堆出了 4 份重复副本共 176 MB。故在返回前删除本例暂存。
    stage_in.mkdir(parents=True, exist_ok=True)
    staged = stage_in / t1w.name
    if not staged.exists() or staged.stat().st_size != t1w.stat().st_size:
        shutil.copy2(t1w, staged)

    # 容器内挂载点：stage_root/in -> /data（可读）, fsroot -> /out（可写）
    t1_in = to_docker_path(staged, stage_root / 'in', '/data')
    sd_in = '/out'

    cmd = [
        'docker', 'run', '--rm',
        '-u', '0:0',
        '-v', f'{Path(stage_root / "in").resolve()}:/data:ro',
        '-v', f'{Path(fsroot).resolve()}:/out',
        FASTSURFER_IMAGE,
        '--t1', t1_in,
        '--sid', sub,
        '--sd', sd_in,
        '--seg_only',
        '--allow_root',            # 见文档点 2
        '--device', 'cpu',         # 本机无 NVIDIA GPU
        '--threads', str(threads),
        '--no_cereb',
        '--no_hypothal',
        '--vox_size', '1',
    ]
    if extra:
        cmd.extend(extra)

    log(f'    [docker] FastSurfer --seg_only --device cpu --threads {threads}')
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=int(timeout_h * 3600))
    except subprocess.TimeoutExpired:
        return False, f'timeout after {timeout_h}h'
    el = time.time() - t0

    # 清理本例暂存输入（见上文空间管理说明）——无论成败都删，
    # 因为分割产物已落在 work/ 下，暂存副本不再需要。
    try:
        shutil.rmtree(stage_in.parent, ignore_errors=True)
    except Exception:                                 # noqa: BLE001
        pass

    if aseg.exists():
        return True, f'{el/60:.1f} min'
    tail = ((r.stderr or '') + (r.stdout or '')).strip().splitlines()[-6:]
    return False, f'exit={r.returncode} after {el/60:.1f} min :: ' + ' | '.join(tail)


# --------------------------------------------------------- registration + extract

def _atlas_to_native(atlas: Path, native_t1: Path, stage_root: Path, out_dir: Path,
                     sub: str, work: Path | None = None,
                     license_file: Path | None = None,
                     mni_ref: Path | None = None,
                     timeout_h: float = 2.0) -> tuple[Path | None, str]:
    """把 MNI 空间的 Schaefer 图集配准并重采样到被试原生空间。

    为什么不用 FSL flirt（实测两条路都断了）
    ----------------------------------------
    1. FastSurfer 镜像内**没有 FSL**（`which flirt` 无命中，只有 /opt/freesurfer）；
    2. `fsl/fsl:latest` 这个镜像名**不存在**：
       `pull access denied for fsl/fsl, repository does not exist`。

    为什么不用 mri_coreg（FreeSurfer 的常规配准命令）
    -----------------------------------------------
    它依赖 `$FREESURFER_HOME/average/RB_all_2020-01-02.gca` 作初始化，
    而实测该文件是 **0 字节空文件**，配准不可靠。

    采用的方案：mri_coreg 12dof + mri_vol2vol --lta-inv
    -----------------------------------------------------
    关键洞察：我们**已经有** Schaefer 图集，它本身就定义在 FSL MNI152 1mm 网格上
    （shape (182,218,182)）。因此不需要 FSL 的 MNI152 T1 模板来桥接空间，
    直接用图集作参考做仿射配准即可。

    只重采样**图集**（标签图，最近邻无损），**不重采样** T1w/T2w ——
    因为契约要求 GWC 与 T1w/T2w 是原生空间的强度比值，任何插值都会改变比值。
    因此先做 native→atlas 的配准，取逆得到 atlas→native 的变换。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    warped = out_dir / f'Schaefer200_in_{sub}.nii.gz'
    # 同样用完整性校验而非 exists()：断电留在半途的重采样图必须被识别并重做
    _ok, _why = is_complete_volume(warped, min_bytes=100_000)
    if _ok:
        return warped, 'cached'
    if warped.exists():
        log(f'    [warn] 已有配准结果不可信（{_why}），将重做')
        warped.unlink(missing_ok=True)
        # 同时清掉可能同源的 LTA，避免半截变换被复用
        for f in out_dir.glob(f'{sub}_native2mni.lta'):
            f.unlink(missing_ok=True)

    if mni_ref is None or not Path(mni_ref).exists():
        return None, ('缺少 MNI152 参考脑 —— 先运行 '
                      'scripts/make_mni152_reference.py 生成 '
                      'data/atlas/MNI152_T1_1mm_in_atlas_grid.nii.gz')
    mni_ref = Path(mni_ref)

    # ---- 网格一致性断言（核心正确性属性）----
    # 参考脑必须与 Schaefer 图集**同网格**：否则配准解出的变换虽在数学上自洽，
    # 但会隐含一个 MNI 变体之间的系统偏移，直接污染所有分区取值。
    # 实测 FSL 官方 MNI152_T1_1mm 与 Schaefer FSLMNI152 图集的 shape/affine
    # 逐元素相同，因此这里应该总是通过；一旦不通过说明参考文件被换错了。
    try:
        _a = nib.load(str(atlas))
        _r = nib.load(str(mni_ref))
        if _a.shape != _r.shape or not np.allclose(_a.affine, _r.affine, atol=1e-4):
            return None, (f'参考脑与图集不同网格，拒绝配准：'
                          f'atlas {_a.shape} vs ref {_r.shape}；'
                          f'affine 相同={np.allclose(_a.affine, _r.affine, atol=1e-4)}')
    except Exception as exc:                          # noqa: BLE001
        return None, f'参考脑读取失败: {type(exc).__name__}: {exc}'

    # 容器内挂载点：/in（T1w + 图集 + 脑掩码）、/ref（参考脑）、/out（可写）
    in_root = stage_root / 'reg_in'
    in_root.mkdir(parents=True, exist_ok=True)
    t1_staged = in_root / native_t1.name
    atlas_staged = in_root / atlas.name
    for src, dst in ((native_t1, t1_staged), (atlas, atlas_staged)):
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)

    # 脑提取：把 aseg（1mm，FastSurfer 内部网格）重采样到 T1w 原生网格，
    # 得到原生空间的脑掩码，并据此把 T1w 抠成纯脑。
    #
    # [!] 为什么必须做这一步（实测得出的关键结论）
    # ------------------------------------------
    # 原生 T1w 是 0.5mm **全头**影像（含颈部，z 范围 -129…131 mm），
    # 而参考脑是**脑提取过**的 1mm 模板。两者直接配准时 `mri_coreg` 的
    # 质心初始化会被非脑组织带偏，实测收敛到一个 z 轴差 **10.7 mm** 的解
    # （T1 配准后与参考脑的 Dice 仅 0.833；1 cm 足以把分区错配到相邻脑回）。
    # 抠出纯脑后两个影像的质心可比，配准才可靠。
    aseg_path = (work or out_dir.parent) / 'fastsurfer' / sub / 'mri' / 'aseg.auto.mgz'
    brain_staged = None
    if aseg_path.exists():
        from nibabel.processing import resample_from_to
        asg = nib.load(str(aseg_path))
        t1_img = nib.load(str(native_t1))
        # 最近邻把标签图搬到原生网格
        mask_native = resample_from_to(
            nib.Nifti1Image((np.asarray(asg.dataobj) > 0).astype(np.uint8),
                            asg.affine, asg.header),
            (t1_img.shape, t1_img.affine), order=0, cval=0)
        m = np.asarray(mask_native.dataobj) > 0
        if m.sum() > 100_000:                      # 合理性门槛
            brain = np.asarray(t1_img.dataobj, dtype=np.float32) * m
            brain_staged = in_root / f'{sub}_brain.nii.gz'
            nib.save(nib.Nifti1Image(brain.astype(np.float32), t1_img.affine,
                                     t1_img.header), str(brain_staged))

    mov_name = brain_staged.name if brain_staged else native_t1.name

    lta_fwd = f'/out/{sub}_native2mni.lta'
    # 参考脑：与 Schaefer 图集**完全同网格**的 MNI152 T1
    # （由 scripts/make_mni152_reference.py 从 nilearn 自带的 ICBM152 模板重采样得到）
    ref_name = mni_ref.name
    # [!] 用 **Python 脚本**驱动容器内流程，而不是 bash。
    # 原因：bash 解析器对含中文的字符串 + `$(...)` 嵌套命令替换会报
    # "syntax error: unexpected end of file"（实测），而这与配准本身无关，
    # 纯粹是引号/编码问题，排查它浪费时间。Python 处理 UTF-8 无此问题。
    py_script = f'''
import subprocess, sys, os, json
def run(cmd, **kw):
    print("RUN:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout: print(r.stdout[-3000:], flush=True)
    if r.stderr: print("STDERR:", r.stderr[-2000:], flush=True)
    return r.returncode

lic = "/opt/freesurfer/license.txt"
print("license.txt exists:", os.path.exists(lic), flush=True)
print("FS_LICENSE =", os.environ.get("FS_LICENSE"), flush=True)

lta = "{lta_fwd}"
warped = "/out/{warped.name}"

rc = run(["mri_coreg", "--mov", "/in/{mov_name}", "--ref", "/ref/{ref_name}",
          "--reg", lta, "--dof", "12", "--threads", "4"])
if not (os.path.exists(lta) and os.path.getsize(lta) > 0):
    print("FATAL: mri_coreg produced no LTA, rc=", rc, flush=True)
    sys.exit(11)
print("STEP1 OK lta bytes =", os.path.getsize(lta), flush=True)

rc = run(["mri_vol2vol", "--mov", "/in/{atlas.name}", "--targ", "/in/{native_t1.name}",
          "--lta-inv", lta, "--nearest", "--o", warped])
if not (os.path.exists(warped) and os.path.getsize(warped) > 0):
    print("FATAL: mri_vol2vol produced no output, rc=", rc, flush=True)
    sys.exit(12)
print("STEP2 OK warped bytes =", os.path.getsize(warped), flush=True)

import numpy as np, nibabel as nib
a = np.asarray(nib.load(warped).dataobj)
u = np.unique(a); u = u[u > 0]
qc = {{"warped_voxels": int((a > 0).sum()), "labels": int(len(u))}}
open("/out/{sub}_reg_qc.json", "w").write(json.dumps(qc))
print("QC:", qc, flush=True)
sys.exit(0)
'''
    script_path = stage_root / f'reg_{sub}.py'
    script_path.write_text(py_script, encoding='utf-8')

    cmd = [
        'docker', 'run', '--rm', '-u', '0:0',
        *license_mount_args(license_file),
        '-v', f'{Path(in_root).resolve()}:/in:ro',
        '-v', f'{Path(mni_ref.parent).resolve()}:/ref:ro',
        '-v', f'{Path(stage_root).resolve()}:/scripts:ro',
        '-v', f'{Path(out_dir).resolve()}:/out',
        '--entrypoint', 'python3',
        FASTSURFER_IMAGE, f'/scripts/reg_{sub}.py',
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=int(timeout_h * 3600))
    except Exception as exc:                          # noqa: BLE001
        return None, f'{type(exc).__name__}: {exc}'

    # 用完整性校验而非 exists()：确认重采样图真的可读、标签数合理
    ok, why = is_complete_volume(warped, min_bytes=100_000)
    if ok:
        # 清理本例暂存（T1w + 图集，数十 MB；不清理会随批次累积）
        shutil.rmtree(in_root, ignore_errors=True)
        return warped, f'mri_coreg ({why})'
    # 半截输出必须删掉，否则下一轮会被当成"已完成"
    if warped.exists():
        warped.unlink(missing_ok=True)
    shutil.rmtree(in_root, ignore_errors=True)
    tail = ((r.stderr or '') + (r.stdout or '')).strip().splitlines()[-4:]
    return None, f'exit={r.returncode} :: ' + ' | '.join(tail)[:300]


def cortical_band(aseg: np.ndarray, band_mm: float, vox_mm: np.ndarray) -> np.ndarray:
    """灰白质交界带掩码（体素级）。

    定义：皮层带 ∩ (WM ∪ 距离 WM <= band_mm 的体素)。
    用 3D 距离变换实现"距离 WM 多近"，避免逐层膨胀的尺度不一致问题。
    """
    from scipy import ndimage

    ctx = np.isin(aseg, [LBL_CTX_LH, LBL_CTX_RH])
    wm = np.isin(aseg, [LBL_WM_LH, LBL_WM_RH])
    if not ctx.any() or not wm.any():
        return np.zeros_like(aseg, dtype=bool)

    # 采样间距（mm）；各向同性时取均值即可满足"距离 <= band_mm"的判据
    spacing = tuple(float(np.mean(vox_mm)) for _ in range(3))
    dist = ndimage.distance_transform_edt(~wm, sampling=spacing)
    return ctx & (dist <= band_mm)


def parcel_values(scalar: np.ndarray, parcel: np.ndarray, mask: np.ndarray,
                  n_parcels: int = N_PARCELS) -> np.ndarray:
    """在 mask 内按分区求均值。返回 (n_parcels,)，缺失为 NaN。"""
    vals = np.full(n_parcels, np.nan, dtype=float)
    ids = parcel[mask]
    s = scalar[mask]
    fin = np.isfinite(s) & (ids >= 1) & (ids <= n_parcels)
    ids, s = ids[fin], s[fin]
    if ids.size == 0:
        return vals
    # bincount 求和 + 计数，避免逐分区布尔索引
    tot = np.bincount(ids, weights=s, minlength=n_parcels + 1)[1:]
    cnt = np.bincount(ids, minlength=n_parcels + 1)[1:]
    with np.errstate(invalid='ignore', divide='ignore'):
        v = tot / cnt
    v[cnt == 0] = np.nan
    vals[:] = v[:n_parcels]
    return vals


def write_contract_csv(path: Path, values: np.ndarray) -> None:
    """按契约第 3 条写长表：parcel_index,value（1..N），缺失写空。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['parcel_index', 'value'])
        for i, v in enumerate(values, start=1):
            w.writerow([i, '' if not np.isfinite(v) else f'{v:.10g}'])


def write_contract_json(path: Path, sub: str, metric: str, atlas: str,
                        n_parcels: int, notes: str, n_finite: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        'produced_by': 'scripts/run_a3_pipeline.py (FastSurfer seg_only + brain-extraction + mri_coreg + band sampling)',
        'date': date.today().isoformat(),
        'n_subjects': 1,
        'subject': sub,
        'n_parcels': n_parcels,
        'n_parcels_with_value': int(n_finite),
        'atlas': atlas,
        'metric': metric,
        'notes': notes,
    }
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def extract_one(sub: str, t1w: Path, t2w: Path, aseg_path: Path,
                atlas_native: Path, out_root: Path, band_mm: float,
                atlas_name: str) -> tuple[bool, str]:
    """取值并写契约产物。

    [!] 网格不一致是**常态**，不是异常（实测）：本数据集中 T1w 与 T2w 是两次独立
    采集，FOV 不同（例如 sub-01 的 T1w 为 340×512×512、T2w 为 360×512×512）。
    FastSurfer 的 aseg 又是它自己的 256^3 1mm 内部网格。
    因此必须先把 T2w 与 aseg 重采样到 **T1w 原生网格**（T1w 作为基准，
    因为配准后的图集已经在这个网格上），再逐体素取值。
    """
    from nibabel.processing import resample_from_to

    t1_img = nib.load(str(t1w))
    ref_grid = (t1_img.shape, t1_img.affine)

    t2_img_raw = nib.load(str(t2w))
    asg_img_raw = nib.load(str(aseg_path))
    par_img = nib.load(str(atlas_native))

    # 统一到 T1w 网格：T2w 用三线性（强度图），aseg 用最近邻（标签图）
    if t2_img_raw.shape != t1_img.shape or not np.allclose(t2_img_raw.affine,
                                                           t1_img.affine, atol=1e-4):
        t2_img = resample_from_to(t2_img_raw, ref_grid, order=1, cval=0.0)
    else:
        t2_img = t2_img_raw
    if asg_img_raw.shape != t1_img.shape or not np.allclose(asg_img_raw.affine,
                                                            t1_img.affine, atol=1e-4):
        asg_img = resample_from_to(asg_img_raw, ref_grid, order=0, cval=0)
    else:
        asg_img = asg_img_raw

    if par_img.shape != t1_img.shape or not np.allclose(par_img.affine,
                                                        t1_img.affine, atol=1e-4):
        return False, (f'配准后的图集网格与 T1w 不一致: atlas={par_img.shape} '
                       f't1={t1_img.shape}（配准环节有问题，不应发生）')

    t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
    t2 = np.asarray(t2_img.dataobj, dtype=np.float32)
    aseg = np.asarray(asg_img.dataobj).astype(np.int16)
    parcel = np.asarray(par_img.dataobj).astype(np.int16)

    vox_mm = np.abs(np.diag(t1_img.affine))[:3]

    # T1w/T2w 比值：分母过小处（脑外、CSF）会产生爆炸值，先设下限再取比值
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = t1 / np.where(t2 > 1e-6, t2, np.nan)

    band = cortical_band(aseg, band_mm, vox_mm)
    if band.sum() < 1000:
        return False, f'cortical band too small ({int(band.sum())} voxels)'

    # GWC：灰白质对比度。用带内 T1 强度相对全脑白质中位数的比值表示，
    # 这样是"未见对比度"的无量纲度量，可跨被试比较。
    wm_all = np.isin(aseg, [LBL_WM_LH, LBL_WM_RH])
    wm_med = float(np.median(t1[wm_all])) if wm_all.any() else float('nan')
    if not np.isfinite(wm_med) or wm_med <= 0:
        return False, 'invalid white-matter median intensity'
    gwc_map = t1 / wm_med

    gwc_vals = parcel_values(gwc_map, parcel, band)
    t1t2_vals = parcel_values(ratio, parcel, band)

    notes = (
        f'segment=FastSurfer --seg_only --device cpu (aseg.auto.mgz); '
        f'brain_extraction=aseg mask resampled to T1w native grid before registration '
        f'(critical: without it mri_coreg mis-initialises on the neck and converges '
        f'~10.7 mm off in z, Dice 0.833; with it Percent Overlap 98.4%%, Dice improves); '
        f'registration=mri_coreg 12dof native-brain -> MNI152 reference; '
        f'reference = FSL official MNI152_T1_1mm, whose grid is element-wise identical '
        f'to the Schaefer2018 FSLMNI152 atlas (same shape and affine), so no resampling '
        f'and no MNI305/MNI152 discrepancy is involved; '
        f'then mri_vol2vol --lta-inv --nearest warps the atlas to the T1w native grid; '
        f'band=gray-white boundary, cortex within {band_mm} mm of WM '
        f'({int(band.sum())} voxels); '
        f'GWC=T1/wm_median (wm_median={wm_med:.1f}); '
        f'T1w/T2w=raw intensity ratio with t2 floor 1e-6; '
        f'T2w and aseg resampled onto the T1w grid before sampling '
        f'(acquisitions have different FOV, e.g. T1w 340x512x512 vs T2w 360x512x512); '
        f'no group comparison / covariate regression / normalization applied '
        f'(per derived_data_contract section 0.6)'
    )

    write_contract_csv(out_root / 'a3_gwc' / f'{sub}.csv', gwc_vals)
    write_contract_csv(out_root / 'a3_t1t2' / f'{sub}.csv', t1t2_vals)
    write_contract_json(out_root / 'a3_gwc' / f'{sub}.json', sub, 'grey-white contrast (GWC)',
                        atlas_name, N_PARCELS, notes, int(np.isfinite(gwc_vals).sum()))
    write_contract_json(out_root / 'a3_t1t2' / f'{sub}.json', sub, 'T1w/T2w ratio',
                        atlas_name, N_PARCELS, notes, int(np.isfinite(t1t2_vals).sum()))

    return True, (f'band={int(band.sum())} vox; '
                  f'gwc valid={int(np.isfinite(gwc_vals).sum())}/{N_PARCELS}; '
                  f't1t2 valid={int(np.isfinite(t1t2_vals).sum())}/{N_PARCELS}')


# ---------------------------------------------------------------------- main

def discover(raw: Path, only: list[str] | None) -> list[str]:
    subs = []
    for d in sorted(raw.iterdir()):
        if not d.is_dir() or not d.name.startswith('sub-'):
            continue
        if only and d.name not in only:
            continue
        if (d / 'anat').is_dir():
            subs.append(d.name)
    return subs


def transfer_to_dest(out_root: Path, dest: Path, subs: list[str]) -> tuple[int, int]:
    """把契约产物从 out_root（C:）拷到 dest（F:）。

    为什么需要这一步：Docker Desktop 无法在 F: 上做挂载（实测写入不传播），
    因此容器相关的读写全部在 C: 完成，最终产物用**普通文件拷贝**送到 F:。
    只拷 a3_gwc / a3_t1t2 下的 csv+json（体积极小），不搬分割中间件。
    返回 (拷贝成功数, 跳过数)。
    """
    ok = skip = 0
    for metric in ('a3_gwc', 'a3_t1t2'):
        for sub in subs:
            for ext in ('csv', 'json'):
                src = out_root / metric / f'{sub}.{ext}'
                if not src.exists():
                    continue
                dst = dest / metric / f'{sub}.{ext}'
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() and dst.stat().st_size == src.stat().st_size:
                    skip += 1
                    continue
                shutil.copy2(src, dst)
                ok += 1
    return ok, skip


def main() -> int:
    ap = argparse.ArgumentParser(description='A3 上游：T1w/T2w -> 分区 GWC 与 T1w/T2w')
    ap.add_argument('--raw', required=True, type=Path, help='原始 BIDS 根目录（可读，含 F:）')
    ap.add_argument('--work', required=True, type=Path,
                    help='中间产物目录。**必须在 C: 等固定盘上** —— '
                         'Docker Desktop 无法挂载 F:，容器读写都走这里')
    ap.add_argument('--out', required=True, type=Path,
                    help='衍生物暂存目录。**同样建议在 C:**，容器相关产物先落这里')
    ap.add_argument('--dest', type=Path, default=None,
                    help='最终交付目录（如 F:\\ds005713\\derivatives）。'
                         '指定后会把 a3_gwc/a3_t1t2 的 csv+json 从 --out 拷到此处。'
                         '不指定则只留在 --out')
    ap.add_argument('--atlas', type=Path, default=DEFAULT_ATLAS,
                    help=f'MNI 空间的 Schaefer200 体积图（默认 {DEFAULT_ATLAS}）')
    ap.add_argument('--subjects', nargs='*', default=None, help='只处理这些被试')
    ap.add_argument('--subjects-file', type=Path, default=None,
                    help='从文件读被试清单（每行一个，支持 # 注释）。'
                         '用于批次化推进：把 213 例切成若干批，逐批跑，中断可续')
    ap.add_argument('--stage', choices=['all', 'seg', 'extract'], default='all')
    ap.add_argument('--band-mm', type=float, default=2.0,
                    help='灰白质交界带宽度（mm），默认 2.0，须在论文中写明')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--timeout-h', type=float, default=4.0, help='单例分割超时（小时）')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--timeit', action='store_true', help='打印单例耗时')
    ap.add_argument('--stage-root', type=Path, default=None,
                    help='容器可读的暂存目录（必须在 C: 等固定盘上；'
                         '默认 <项目>/build/docker_staging）。'
                         '原因：Docker Desktop 无法挂载可移动盘 F:，见 run_fastsurfer 文档')
    ap.add_argument('--seg-root', type=Path, default=None,
                    help='分割结果所在目录；默认与 --work 相同。'
                         '用于「分割已在别处完成、只想重跑配准与取值」的场景，'
                         '避免每次调整都重付一次数十分钟的分割代价')
    ap.add_argument('--fs-license', type=Path, default=None,
                    help='FreeSurfer license.txt 路径（学术免费，约 1 分钟取得：'
                         'https://surfer.nmr.mgh.harvard.edu/registration.html）。'
                         '**配准必需** —— 容器内无许可证时 mri_coreg 等命令会'
                         '静默失败（退出码 0 但不产出文件）')
    ap.add_argument('--default-fs-license', action='store_true',
                    help='自动在常见位置查找 license.txt 并使用')
    ap.add_argument('--mni-ref', type=Path,
                    default=ROOT / 'data' / 'atlas' / 'MNI152_T1_1mm_FSL.nii.gz',
                    help='与 Schaefer 图集**同网格**的 MNI152 T1 参考脑（配准的 --ref）。'
                         '默认用 FSL 官方 MNI152_T1_1mm —— 实测其 shape 与 affine '
                         '与 Schaefer2018 FSLMNI152 图集**逐元素相同**，因此不需要任何'
                         '重采样，配准目标就是图集的原生空间，不存在 MNI305/MNI152 落差')
    args = ap.parse_args()

    stage_root = args.stage_root or (ROOT / 'build' / 'docker_staging')
    stage_root.mkdir(parents=True, exist_ok=True)

    # ---- 定位 FreeSurfer 许可证（配准必需）----
    fs_license: Path | None = args.fs_license
    if fs_license is None and args.default_fs_license:
        for cand in (ROOT / 'build' / 'license.txt', ROOT / 'license.txt',
                     Path.home() / 'license.txt',
                     Path('C:/Program Files/FreeSurfer/license.txt'),
                     Path('C:/freesurfer/license.txt')):
            if cand.exists():
                fs_license = cand
                break
    if fs_license is not None and not fs_license.exists():
        print(f'[FATAL] 许可证文件不存在: {fs_license}', file=sys.stderr)
        return 2

    atlas_name = args.atlas.name.replace('.nii.gz', '')
    if not args.atlas.exists():
        print(f'[FATAL] 图集不存在: {args.atlas}\n'
              f'        先运行 scripts/fetch_atlas_volume.py 冻结体积图。', file=sys.stderr)
        return 2

    # 被试清单：--subjects 与 --subjects-file 取并集（任一为空则用之）
    only = list(args.subjects) if args.subjects else None
    if args.subjects_file:
        if not args.subjects_file.exists():
            print(f'[FATAL] 清单文件不存在: {args.subjects_file}', file=sys.stderr)
            return 2
        # utf-8-sig 会剥掉 UTF-8 BOM。用 utf-8 读取时，若清单被记事本等编辑过
        # 而带上 BOM，首行会变成 '\ufeffsub-XXX' —— 一个不存在的"幽灵被试"，
        # 表现为"指定但不存在"的警告且该被试被静默跳过。
        from_file = [
            ln.strip() for ln in
            args.subjects_file.read_text(encoding='utf-8-sig').splitlines()
            if ln.strip() and not ln.strip().startswith('#')
        ]
        # ⚠️ 空清单必须**拒绝运行**。`discover()` 里用的是 `if only and ...`，
        # 所以 `only == []` 等价于"不过滤" —— 一个被意外清空的清单文件会让
        # 整个 200+ 例的多天作业静默重新启动。这里显式拦住。
        if not from_file and not (only or []):
            log(f'[FATAL] 清单文件为空: {args.subjects_file}')
            log('        空清单不等于"全部被试"；请重新生成清单后再运行。')
            return 2
        only = sorted(set((only or []) + from_file))
        log(f'清单文件 {args.subjects_file.name}: {len(from_file)} 个被试')

    subs = discover(args.raw, only)
    if not subs:
        print('[ERR] 未发现任何有 anat 目录的被试', file=sys.stderr)
        return 2

    log(f'A3 上游管线  被试数={len(subs)}  阶段={args.stage}  带={args.band_mm}mm')
    log(f'  raw   : {args.raw}')
    log(f'  work  : {args.work}   (必须固定盘)')
    log(f'  out   : {args.out}    (必须固定盘)')
    if args.dest:
        log(f'  dest  : {args.dest}   (最终交付，产出后拷贝至此)')
    log(f'  atlas : {args.atlas.name}')
    log('')

    # ---- Docker 盘符前置检查：F: 不可用，必须尽早报错而不是跑到一半才失败 ----
    for label, p in (('--work', args.work), ('--out', args.out)):
        drive = str(p.resolve().drive).upper()
        if drive.startswith('F'):
            print(f'[FATAL] {label} 指向 {drive} —— Docker Desktop 无法在 F: 上做挂载'
                  f'（实测容器写入不会传播到宿主）。\n'
                  f'        请把它改到 C:（如 build\\work 与 build\\derivatives），'
                  f'再用 --dest 指定 F: 作为最终交付目录。', file=sys.stderr)
            return 2
    args.work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- 前置清点 ----
    # 除了"文件在不在"，还要验证 gzip 流完好。实测 sub-055 的两个 T1w 在清单里
    # 记录的大小本身就是截断的（270 KB / 14.4 MB，正常约 50 MB），字节数校验
    # 查不出来；交给 FastSurfer 会白等约 30 分钟才失败。
    #
    # 性能注意：完整解压一例的 T1w+T2w 约 360 MB，96 例就是约 35 GB、约 12 分钟。
    # 对**续跑**而言，已完成的被试根本不会被读，给它们做预检纯属浪费
    # （实测续跑 83/96 时白等了 12 分钟）。因此这里先判"整例是否已完整"，
    # 已完整的直接跳过、不做预检。
    plan = []
    n_corrupt = 0
    n_prechecked = 0
    for sub in subs:
        t1, t2 = pick_t1w(args.raw, sub), pick_t2w(args.raw, sub)
        # 整例已完成 -> 既不需要预检，也不需要重算
        if args.stage == 'all' and t1 and t2:
            done, _ = subject_already_extracted(sub, args.out)
            if done:
                plan.append((sub, t1, t2))
                continue
        if t1 and t2:
            for label, f in (('T1w', t1), ('T2w', t2)):
                good, why = gzip_stream_ok(f)
                if not good:
                    log(f'    [corrupt] {sub}: {label} 压缩流损坏（{why}）-> 跳过')
                    log(f'              文件: {f.name}  {f.stat().st_size:,} B')
                    n_corrupt += 1
                    t1 = t2 = None
                    break
            n_prechecked += 1
        plan.append((sub, t1, t2))
    n_ready = sum(1 for _, t1, t2 in plan if t1 and t2)
    log(f'  有可用的 T1w+T2w: {n_ready}/{len(subs)}')
    if n_prechecked:
        log(f'  gzip 完整性预检: {n_prechecked} 例（已完成的被试不做预检以省时）')
    if n_corrupt:
        log(f'  因压缩流损坏而排除: {n_corrupt} 例（见上方 [corrupt] 行）')
    for sub, t1, t2 in plan:
        missing = []
        if t1 is None:
            missing.append('T1w')
        if t2 is None:
            missing.append('T2w')
        if missing:
            log(f'    [skip] {sub}: 缺 {"+".join(missing)}')
    if args.dry_run:
        log('\n(dry-run，未执行)')
        return 0

    ok = fail = 0
    done_subs: list[str] = []
    skipped_subs: list[str] = []
    t_start = time.time()
    for i, (sub, t1, t2) in enumerate(plan, 1):
        if t1 is None or t2 is None:
            continue

        # ---- 整例跳过：契约产物已完整（续跑时的主要提速点）----
        if args.stage == 'all':
            done, why = subject_already_extracted(sub, args.out)
            if done:
                skipped_subs.append(sub)
                done_subs.append(sub)          # 仍需纳入传送核对
                log(f'[{i}/{len(plan)}] {sub}  跳过（产物已完整）')
                continue

        log(f'[{i}/{len(plan)}] {sub}')
        t_sub = time.time()

        # ---- 阶段 1: 分割 ----
        if args.stage in ('all', 'seg'):
            good, why = run_fastsurfer(t1, args.work, sub,
                                       args.threads, args.timeout_h, stage_root)
            log(f'    分割: {"OK" if good else "FAIL"} ({why})')
            if not good:
                fail += 1
                continue
        if args.stage == 'seg':
            ok += 1
            continue

        aseg = (args.seg_root or args.work) / 'fastsurfer' / sub / 'mri' / 'aseg.auto.mgz'
        if not aseg.exists():
            log(f'    [skip] 缺分割结果 {aseg}')
            fail += 1
            continue

        # 注：aseg.auto.mgz 是 FreeSurfer MGZ 格式，nibabel 可直接读，
        # 且与输入 T1w 处于同一原生空间（FastSurfer 不做重采样到 MNI）。
        # ---- 阶段 2: 配准 ----
        if fs_license is None:
            log('    配准: 跳过 —— 未提供 FreeSurfer 许可证（--fs-license）')
            log('          容器内无许可证时 mri_coreg 会静默失败，故不尝试。')
            log('          取得（学术免费，约 1 分钟）:')
            log('          https://surfer.nmr.mgh.harvard.edu/registration.html')
            fail += 1
            continue
        reg_dir = args.work / 'registration'
        atlas_native, how = _atlas_to_native(args.atlas, t1, stage_root, reg_dir, sub,
                                             work=args.work,
                                             license_file=fs_license,
                                             mni_ref=args.mni_ref)
        if atlas_native is None:
            log(f'    配准: FAIL ({how})')
            fail += 1
            continue
        log(f'    配准: OK ({how})')

        # ---- 阶段 3: 取值 ----
        good, why = extract_one(sub, t1, t2, aseg, atlas_native, args.out,
                                args.band_mm, atlas_name)
        log(f'    取值: {"OK" if good else "FAIL"} ({why})')
        if good:
            ok += 1
            done_subs.append(sub)

            # ---- 阶段 4: 每例完成即传送到最终交付目录（F:）----
            # 逐例传送而非批次末尾：中断时已完成的例已在 F:，不必靠 C: 上的暂存。
            if args.dest:
                n_ok, n_skip = transfer_to_dest(args.out, args.dest, [sub])
                log(f'    传送: 拷贝 {n_ok} / 跳过 {n_skip} -> {args.dest}')
        else:
            fail += 1

        if args.timeit:
            log(f'     单例总耗时 {time.time()-t_sub:.1f}s')

    total = time.time() - t_start
    log('')
    log(f'完成: 新算 {ok}，跳过(已完整) {len(skipped_subs)}，失败 {fail}，'
        f'总耗时 {total/60:.1f} min')
    if skipped_subs:
        log(f'  已跳过样例: {", ".join(skipped_subs[:8])}'
            + (' ...' if len(skipped_subs) > 8 else ''))
    if args.timeit and ok:
        per = total / max(ok, 1)
        log(f'单例均值 ≈ {per:.1f}s -> 193 例约 {per*193/3600:.1f} h')

    # 兜底：若某例在传送前中断，这里补一次全量传送（幂等，已存在的按字节数跳过）
    if args.dest and done_subs:
        n_ok, n_skip = transfer_to_dest(args.out, args.dest, done_subs)
        log(f'交付目录核对: {args.dest}（累计拷贝 {n_ok} / 已存在 {n_skip}）')

    return 0 if fail == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
