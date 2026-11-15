#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 A1 / A2 批量处理的被试清单，并给出可处理性诊断。

为什么需要单独一步
------------------
A1 与 A2 都**依赖 A3 的产物**（复用 FastSurfer 分割与原生空间图集），因此
"能跑哪些被试"取决于 A3 跑到哪里了。把这一步显式化，可以：

1. 在 A3 尚未全量完成时就分批启动 A1/A2（A3 做完一批，A1/A2 就能跟上）；
2. 提前发现"缺 DWI / 缺 BOLD / 缺 A3 产物"的被试，避免跑到一半才发现；
3. 产出稳定的清单文件，让 `--subjects-file` 可复现。

判据
----
A1 可处理 = 有 DWI 主序列 + 有反向相位 b0 + 有 A3 的 `aseg.auto.mgz`
            + 有 A3 的原生空间图集 `Schaefer200_in_<sub>.nii.gz`
A2 可处理 = 有 rest BOLD + 有 A3 的 `orig.mgz` / `aseg.auto.mgz` + 原生图集

用法
----
    python scripts/make_a1_a2_lists.py --raw F:\\ds005713\\raw \\
        --a3-work <A3 work 目录> --out-dir build
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import run_a1_pipeline as A1                              # noqa: E402
import run_a2_pipeline as A2                              # noqa: E402


def log(msg: str = '') -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode('ascii'), flush=True)


def a3_ready(a3_work: Path, sub: str) -> dict:
    """检查 A3 为某被试产出了哪些 A1/A2 需要的文件。"""
    fs = a3_work / 'fastsurfer' / sub / 'mri'
    reg = a3_work / 'registration'
    return {
        'aseg': (fs / 'aseg.auto.mgz').exists(),
        'orig': (fs / 'orig.mgz').exists(),
        'atlas': (reg / f'Schaefer200_in_{sub}.nii.gz').exists(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='生成 A1/A2 被试清单')
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--a3-work', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'build')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    subs = sorted(p.name for p in args.raw.iterdir()
                  if p.is_dir() and p.name.startswith('sub-'))

    # 读出时间协议常数（A1 的元数据回退需要）
    protocol = A1.collect_protocol_readout(args.raw)

    a1_ok, a2_ok = [], []
    a1_reasons: Counter = Counter()
    a2_reasons: Counter = Counter()
    detail = {}

    for s in subs:
        r = a3_ready(args.a3_work, s)
        dwi = A1.pick_dwi(args.raw, s, protocol)
        bold = A2.pick_bold(args.raw, s)

        # ---- A1 ----
        if dwi is None:
            a1_reasons['缺 DWI 或相位元数据'] += 1
        elif not dwi.get('rev'):
            a1_reasons['缺反向相位 b0'] += 1
        elif not r['aseg']:
            a1_reasons['A3 未产出 aseg'] += 1
        elif not r['atlas']:
            a1_reasons['A3 未产出原生图集'] += 1
        else:
            a1_ok.append(s)

        # ---- A2 ----
        if bold is None:
            a2_reasons['缺 rest BOLD'] += 1
        elif not r['orig']:
            a2_reasons['A3 未产出 orig.mgz'] += 1
        elif not r['atlas']:
            a2_reasons['A3 未产出原生图集'] += 1
        else:
            a2_ok.append(s)

        detail[s] = {
            'a3': r,
            'dwi': bool(dwi),
            'dwi_pe': dwi['pe'] if dwi else None,
            'dwi_readout_level': dwi.get('readout_level') if dwi else None,
            'dwi_rev': bool(dwi.get('rev')) if dwi else False,
            'bold': bool(bold),
            'bold_tr': bold['tr'] if bold else None,
        }

    def _write(name: str, items: list[str]) -> Path:
        p = args.out_dir / name
        p.write_text('\n'.join(items) + ('\n' if items else ''), encoding='utf-8')
        return p

    p1 = _write('a1_subjects.txt', a1_ok)
    p2 = _write('a2_subjects.txt', a2_ok)

    log(f'=== 清单生成结果（总目录 {len(subs)}）===')
    log(f'  A1 可处理: {len(a1_ok):3d}  -> {p1}')
    for k, v in a1_reasons.most_common():
        log(f'      跳过 {v:3d} 例: {k}')
    log(f'  A2 可处理: {len(a2_ok):3d}  -> {p2}')
    for k, v in a2_reasons.most_common():
        log(f'      跳过 {v:3d} 例: {k}')

    # 细目（便于复核哪些被试因何被排除）
    (args.out_dir / 'a1_a2_availability.json').write_text(
        json.dumps({'n_total': len(subs), 'a1_ok': a1_ok, 'a2_ok': a2_ok,
                    'protocol': protocol, 'detail': detail},
                   ensure_ascii=False, indent=1), encoding='utf-8')
    log(f'\n  细目 -> {args.out_dir / "a1_a2_availability.json"}')

    # 读出时间来源分布（写论文时要在方法/局限里交代）
    if protocol.get('median'):
        lv = protocol['levels']
        log(f'\n=== 读出时间来源分布（A1 元数据回退）===')
        log(f'  协议常数 {protocol["median"]:.7f}s  '
            f'（有值 {protocol["n_with"]}/{protocol["n_total"]}，'
            f'唯一值 {protocol["n_unique"]} 个）')
        log(f'  L1 run-01 自带 : {lv[1]}')
        log(f'  L2 借用反向 b0 : {lv[2]}')
        log(f'  L3 EES 推算    : {lv[3]}')
        log(f'  L4 协议中位数  : {lv[4]}   <- 需在论文局限中说明')

    # 配对检查：A1 与 A2 都可用（做 SC-FC 耦合分析需要同一被试两者都有）
    both = set(a1_ok) & set(a2_ok)
    log(f'\n=== 交叉可用性 ===')
    log(f'  A1 且 A2 都可用: {len(both)} 例（跨模态耦合分析的最小集）')
    base = sorted(s for s in both if not s.endswith('fu'))
    fu = sorted(s for s in both if s.endswith('fu'))
    log(f'    基线 {len(base)}   随访 {len(fu)}')
    # 有随访配对的基线被试
    paired = [s for s in base if f'{s}fu' in both]
    log(f'    有随访配对: {len(paired)} 对')
    _write('a1_a2_both.txt', sorted(both))
    _write('a1_a2_paired_baseline.txt', paired)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
