#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成**符合契约格式的合成派生数据**，用于端到端验证分析流水线。

⚠️ 重要：本脚本产物是**合成数据**，仅用于验证代码路径、文件格式与数值稳定性，
**不得用于任何结果报告或论文数字**。输出目录会写入 ``SYNTHETIC_DO_NOT_REPORT.txt``
标记文件；分析入口若在该目录下运行，会在报告中保留 ``derivatives_dir`` 以便追溯。

用法::

    python scripts/make_synthetic_derivatives.py --out build/synthetic_derivatives \
        --n-patient 60 --n-control 30 --n-parcels 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

MARKER = """本目录由 scripts/make_synthetic_derivatives.py 生成，内容是**合成数据**。
仅供验证分析代码路径与文件格式，不得用于任何结果报告或论文数字。
"""


def _write_vectors(root: Path, kind: str, data: dict[str, np.ndarray]) -> None:
    d = root / kind
    d.mkdir(parents=True, exist_ok=True)
    for sid, vals in data.items():
        pd.DataFrame({'parcel_index': np.arange(1, len(vals) + 1), 'value': vals}) \
            .to_csv(d / f'{sid}.csv', index=False)
        (d / f'{sid}.json').write_text(json.dumps(
            {'produced_by': 'make_synthetic_derivatives.py (SYNTHETIC)',
             'n_parcels': len(vals)}, ensure_ascii=False, indent=2), encoding='utf-8')


def _write_matrices(root: Path, kind: str, data: dict[str, np.ndarray]) -> None:
    d = root / kind
    d.mkdir(parents=True, exist_ok=True)
    for sid, m in data.items():
        np.save(d / f'{sid}.npy', m)
        (d / f'{sid}.json').write_text(json.dumps(
            {'produced_by': 'make_synthetic_derivatives.py (SYNTHETIC)',
             'shape': list(m.shape)}, ensure_ascii=False, indent=2), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='生成合成派生数据（仅用于流水线验证）')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-patient', type=int, default=60)
    ap.add_argument('--n-control', type=int, default=30)
    ap.add_argument('--n-parcels', type=int, default=200)
    ap.add_argument('--n-followup', type=int, default=20)
    ap.add_argument('--effect-parcels', type=int, default=20,
                    help='植入真实组间差异的分区数（用于验证效应检出）')
    ap.add_argument('--effect-size', type=float, default=1.0,
                    help='植入的逐分区效应量（患者组偏移，单位与表型同尺度）')
    ap.add_argument('--seed', type=int, default=20261016)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'SYNTHETIC_DO_NOT_REPORT.txt').write_text(MARKER, encoding='utf-8')

    n_par = args.n_parcels
    patients = [f'sub-{i+1:03d}' for i in range(args.n_patient)]
    controls = [f'sub-{100+i:03d}' for i in range(args.n_control)]
    followup = [f'{s}fu' for s in patients[:args.n_followup]]
    baseline = patients + controls

    # ---- 临床表（模拟 participants_derived.tsv 的结构）----
    def make_clinical(ids: list[str], labels: list[str], seed: int) -> pd.DataFrame:
        """按显式分组标签生成临床行。

        注意：必须用**显式标签列表**而不是单个布尔标志。早期版本把所有基线 ID
        （患者 + 对照拼在一起）统一标为 patient，随后按 participant_id 去重时
        对照行被误判为患者（实测导致组间比较 n_control=0）。
        """
        r = np.random.default_rng(seed)
        n = len(ids)
        is_pat = np.array([lab == 'patient' for lab in labels])
        return pd.DataFrame({
            'participant_id': ids,
            'group_norm': labels,
            'age': np.where(is_pat, r.normal(60, 11, n), r.normal(55, 9, n)).round(1),
            'sex_label': r.choice(['F', 'M'], n),
            'bold_mean_fd': r.uniform(0.06, 0.45, n).round(4),
            'fazekas_grade': np.where(is_pat, r.integers(0, 3, n), r.integers(0, 2, n)).astype(float),
            'pain_side_norm': np.where(is_pat, r.choice(['R', 'L', 'Bilateral'], n, p=[0.58, 0.4, 0.02]), ''),
            'Disease_duration (years)': np.where(is_pat, r.integers(1, 25, n), np.nan).astype(float),
            'Pain_severity (average score)': np.where(is_pat, r.integers(3, 9, n), np.nan).astype(float),
            'Surgery_type': np.where(is_pat, r.choice(['MVD', 'PI', 'Outpatient'], n, p=[0.66, 0.26, 0.08]), ''),
            'pain_reduction_pct': np.where(
                is_pat,
                np.where(r.random(n) < 0.8, 100.0,
                         r.choice([0.0, 30.0, 50.0, 60.0, 70.0, 80.0, 90.0], n)),
                np.nan),
            'complete_relief': np.where(is_pat, np.where(r.random(n) < 0.8, 1.0, 0.0), np.nan),
            'Second follow-up pain NRS (0-10)': np.where(is_pat, r.integers(0, 7, n), np.nan).astype(float),
        })

    clin_base = make_clinical(baseline, ['patient'] * len(patients) + ['control'] * len(controls),
                              args.seed + 1)
    clin_all = clin_base
    if followup:
        clin_all = pd.concat([clin_all, make_clinical(followup, ['patient'] * len(followup),
                                                      args.seed + 3)], ignore_index=True)
    clin_all.to_csv(out / 'clinical_synthetic.tsv', sep='\t', index=False, encoding='utf-8')

    # ---- 植入效应用的空间模式（前 effect_parcels 个分区）----
    #
    # 校准策略：直接按"目标 Hedges' g"设定偏移量，偏移 = g × 该表型的**实测噪声 SD**。
    # 早期版本用固定绝对偏移，导致不同表型的实际效应量差异巨大（实测 A2 仅 0.13、
    # 无法被识别，而 A3 达 0.68），使流水线验证失去意义。
    group_mask = np.array([s in patients for s in baseline])
    target_g = args.effect_size
    fd_vec = clin_all.set_index('participant_id').loc[baseline, 'bold_mean_fd'].to_numpy()
    age_vec = clin_all.set_index('participant_id').loc[baseline, 'age'].to_numpy()

    def _base_values(base: float, spread: float, noise: float, fd_slope: float,
                     age_slope: float, seed: int, clip_a3: bool) -> dict[str, np.ndarray]:
        r = np.random.default_rng(seed)
        out_v: dict[str, np.ndarray] = {}
        for i, sid in enumerate(baseline):
            v = r.normal(base, spread, n_par)
            v = v + noise * r.normal(size=n_par)
            v = v + fd_slope * (fd_vec[i] - fd_vec.mean())
            v = v + age_slope * (age_vec[i] - age_vec.mean())
            if clip_a3:
                v = np.clip(v, 0.05, 0.95)
            out_v[sid] = v
        return out_v

    def make_phenotype(base: float, spread: float, noise: float, fd_slope: float,
                       age_slope: float, seed: int, clip_a3: bool = False,
                       offset: np.ndarray | None = None) -> dict[str, np.ndarray]:
        vals = _base_values(base, spread, noise, fd_slope, age_slope, seed, clip_a3)
        data: dict[str, np.ndarray] = {}
        for i, sid in enumerate(baseline):
            v = vals[sid].copy()
            if group_mask[i] and offset is not None:
                v = v + offset
            if clip_a3:
                v = np.clip(v, 0.05, 0.95)
            data[sid] = np.round(v, 6)
        return data

    def calibrate_offset(base: float, spread: float, noise: float, fd_slope: float,
                         age_slope: float, seed: int, clip_a3: bool = False) -> np.ndarray:
        """按实测噪声 SD 计算达到 target_g 所需的偏移向量。"""
        vals = _base_values(base, spread, noise, fd_slope, age_slope, seed, clip_a3)
        mat = np.vstack([vals[s] for s in baseline])
        sd = float(np.mean(np.nanstd(mat, axis=0, ddof=1)))
        off = np.zeros(n_par)
        off[:args.effect_parcels] = -target_g * sd     # 负向：患者组降低
        return off

    SPECS = {
        'a1_streamlines': dict(base=1.0, spread=0.1, noise=0.05, fd_slope=-0.30,
                               age_slope=-0.001, seed=args.seed + 10, clip_a3=False),
        'a1_fa': dict(base=0.45, spread=0.04, noise=0.02, fd_slope=-0.10,
                      age_slope=-0.0005, seed=args.seed + 11, clip_a3=False),
        'a3_gwc': dict(base=0.40, spread=0.05, noise=0.02, fd_slope=-0.05,
                       age_slope=-0.001, seed=args.seed + 12, clip_a3=True),
        # a3_t1t2 作为"对照表型"：**不植入**分区特异性偏移，仅保留全局均值差
        # （患者组整体偏移 0.9 倍目标 g），用于验证"全脑均值漂移不会被误报为
        # 分区特异性效应"。
        'a3_t1t2': dict(base=1.30, spread=0.08, noise=0.03, fd_slope=-0.08,
                        age_slope=-0.0015, seed=args.seed + 13, clip_a3=False),
    }
    offsets = {k: calibrate_offset(**v) for k, v in SPECS.items()}
    # 对照表型：把分区特异性偏移替换为全局偏移
    offsets['a3_t1t2'] = np.full(n_par, -0.9 * target_g * float(
        np.mean(np.nanstd(np.vstack([_base_values(**SPECS['a3_t1t2'])[s]
                                     for s in baseline]), axis=0, ddof=1))))
    for kind, spec in SPECS.items():
        _write_vectors(out, kind, make_phenotype(**spec, offset=offsets[kind]))

    # ---- 随访（术后）数据 ----
    #
    # 设计：随访只对**患者**生成（真实数据亦然），术后植入效应**部分恢复**
    # （recovery 比例），并叠加随访噪声。这样纵向分析应能检出：
    #   - 配对变化量与 planted 分区重合
    #   - 变化量与"恢复比例"相关的调节效应
    # 早期版本计算了 followup ID 列表却从未生成对应文件，导致纵向链路
    # 始终读到 0 条随访数据（实测 split 后随访矩阵为空）。
    recovery = 0.6
    fu_data: dict[str, dict[str, np.ndarray]] = {k: {} for k in SPECS}
    fu_patients = [s for s in patients[:args.n_followup]]
    for kind, spec in SPECS.items():
        baseline_vals = make_phenotype(**spec, offset=offsets[kind])
        r = np.random.default_rng(spec['seed'] + 5000)
        for sid in fu_patients:
            v = baseline_vals[sid].copy()
            # 术后：植入效应按 (1 - recovery) 残留，即部分恢复
            v = v - (1.0 - recovery) * offsets[kind]
            v = v + spec['noise'] * 0.6 * r.normal(size=n_par)
            if spec['clip_a3']:
                v = np.clip(v, 0.05, 0.95)
            fu_data[kind][f'{sid}fu'] = np.round(v, 6)
        _write_vectors(out, kind, fu_data[kind])

    # ---- 连接矩阵（结构与功能）----
    #
    # 设计要点（合成数据必须忠实于真实数据的生成方向，否则流水线验证会误导）：
    #   1. 结构连接 SC：患者组效应分区连接强度下降 15%
    #   2. 功能连接 FC：以 **患者/对照各自的 SC** 的谐波基生成"个体化"功能连接，
    #      这样 SC 的组间差异会传导到 FC，进而使 A2（结构-功能耦合）也携带组间差异。
    #      若对两组共用同一个 SC 生成 FC，A2 就不会有组间效应（早期版本即如此）。
    #   3. 叠加的噪声量决定 A2 效应量；此处控制为中等水平。
    sc_data, fc_data = {}, {}
    for i, sid in enumerate(baseline):
        r = np.random.default_rng(args.seed + 1000 + i)
        m = r.random((n_par, n_par))
        keep = r.random((n_par, n_par)) < 0.35
        m = m * keep
        m = np.triu(m, 1)
        sc = m + m.T
        if group_mask[i]:
            sc = sc.copy()
            sc[:args.effect_parcels] *= 0.4      # 患者组连接强度显著下降
            sc[:, :args.effect_parcels] *= 0.4
        sc_data[sid] = sc.astype(np.float32)

        deg = sc.sum(axis=1)
        deg[deg == 0] = 1.0
        dinv = 1.0 / np.sqrt(deg)
        L = np.eye(n_par) - (sc * dinv[:, None]) * dinv[None, :]
        ev, U = np.linalg.eigh(L)
        k = max(2, n_par // 8)
        B = U[:, np.argsort(ev)[:k]]
        C = r.normal(size=(n_par, k))
        sig = C @ B.T + r.normal(0, 0.5, (n_par, n_par))
        fc = np.corrcoef(sig)
        np.fill_diagonal(fc, 1.0)
        fc_data[sid] = fc.astype(np.float32)

    # 说明：A2（结构-功能耦合）由 run_phenotypes.py 从 SC/FC **重新计算**，
    # 因此其组间效应必须来自 SC（结构连接）本身的组间差异，不能通过预置 A2 数值注入。
    # 上面把患者组效应分区的 SC 降低到 0.4 倍，即为让 A2 携带可识别的组间效应。
    # 随访的 SC/FC 也一并生成（术后连接强度部分恢复），以便 A2 参与纵向分析。
    for i, sid in enumerate(fu_patients):
        base_sid = sid
        r = np.random.default_rng(args.seed + 3000 + i)
        m = r.random((n_par, n_par))
        keep = r.random((n_par, n_par)) < 0.35
        m = m * keep
        m = np.triu(m, 1)
        sc = m + m.T
        # 术后：结构连接的降低部分恢复（残留 (1-recovery)）
        sc = sc.copy()
        sc[:args.effect_parcels] *= (1.0 - (1.0 - 0.6) * (1.0 - 0.4))
        sc[:, :args.effect_parcels] *= (1.0 - (1.0 - 0.6) * (1.0 - 0.4))
        sc_data[f'{base_sid}fu'] = sc.astype(np.float32)
        deg = sc.sum(axis=1)
        deg[deg == 0] = 1.0
        dinv = 1.0 / np.sqrt(deg)
        L = np.eye(n_par) - (sc * dinv[:, None]) * dinv[None, :]
        ev, U = np.linalg.eigh(L)
        B = U[:, np.argsort(ev)[:max(2, n_par // 8)]]
        C = r.normal(size=(n_par, B.shape[1]))
        sig = C @ B.T + r.normal(0, 0.5, (n_par, n_par))
        fc = np.corrcoef(sig)
        np.fill_diagonal(fc, 1.0)
        fc_data[f'{base_sid}fu'] = fc.astype(np.float32)
    _write_matrices(out, 'a2/sc', sc_data)
    _write_matrices(out, 'a2/fc', fc_data)

    # ---- QC 表 ----
    qc = clin_all[['participant_id']].copy()
    qc['session'] = np.where(qc['participant_id'].str.endswith('fu'), 'followup', 'baseline')
    qc = qc.merge(clin_all[['participant_id', 'bold_mean_fd']], on='participant_id')
    for c in ('has_t1w', 'has_t2w', 'has_dwi', 'has_bold'):
        qc[c] = 1
    qc['bold_pct_fd_gt_0.2'] = (qc['bold_mean_fd'] > 0.2).astype(float) * 0.1
    qc['bold_n_volumes_kept'] = 200
    qc['n_dwi_directions'] = 64
    for c in ('t1_qc_rating', 'dwi_qc_rating', 'bold_qc_rating', 'seg_qc_rating'):
        qc[c] = 1
    qc['exclusion_flag'] = 0
    qc['exclusion_reason'] = ''
    qc['notes'] = 'SYNTHETIC'
    (out / 'qc').mkdir(parents=True, exist_ok=True)
    qc.to_csv(out / 'qc' / 'subjects_qc.csv', index=False)

    print(f'合成派生数据已生成：{out}')
    print(f'  基线被试 {len(baseline)}（患者 {len(patients)}，对照 {len(controls)}）'
          f'，随访 {len(followup)}')
    print(f'  分区数 {n_par}，植入效应分区 {args.effect_parcels} 个'
          f'（目标 Hedges\' g ≈ {target_g}，按各表型实测噪声 SD 校准）')
    print(f'  临床表：{out / "clinical_synthetic.tsv"}')
    print('  ⚠️ 仅供流水线验证，不得报告其数值')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
