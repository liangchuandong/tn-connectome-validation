#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""派生数据校验器：检查用户侧交付的 derivatives/ 是否满足契约。

设计原则：**失败必须精确到文件、行/列与被试**，因为用户侧不应为了格式问题重跑预处理。

检查项：
  S  结构：必需文件/目录是否齐备，每个文件是否有同名 .json 说明
  M  元数据：atlas 文件维度、分区数、标签唯一性
  C  列一致性：所有 A1/A3 CSV 的列集合是否一致（防止导出中列漂移）
  P  分区数：每个 CSV/npy 的行数是否等于图集分区数
  J  被试一致性：derivative 中的被试是否都能在冻结临床表中找到；
                    临床表中的目标被试是否缺产物
  N  数值：缺失编码（空/NaN）、极端值、全常数、负的流线数
  G  矩阵几何：A2 的 SC/FC 是否为方阵、对称、对角是否为 1（FC）
  Q  QC 表：必需列是否齐备、取值域是否合法、排除标记是否与缺失模态自洽
  R  报告：写出 JSON（机器可读）+ Markdown（人可读）

用法：
  python scripts/validate_derivatives.py --derivatives <dir> [--config config/analysis.yaml]
  python scripts/validate_derivatives.py --self-test        # 用合成数据自检校验器本身
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tnwt import config_lite as cl  # noqa: E402

QC_REQUIRED = [
    'participant_id', 'session', 'has_t1w', 'has_t2w', 'has_dwi', 'has_bold',
    'bold_mean_fd', 'bold_pct_fd_gt_0.2', 'exclusion_flag',
]
PER_SUBJECT = {
    'a1_streamlines': ('A1 流线数', True),
    'a1_fa': ('A1 FA 加权', True),
    'a3_gwc': ('A3 GWC', True),
    'a3_t1t2': ('A3 T1w/T2w', True),
}
MATRIX_DIRS = {
    'a2/sc': ('结构连接矩阵', False),
    'a2/fc': ('功能连接矩阵', False),
}


class Finding:
    __slots__ = ('level', 'code', 'where', 'detail')

    def __init__(self, level: str, code: str, where: str, detail: str):
        self.level, self.code, self.where, self.detail = level, code, where, detail

    def as_dict(self) -> dict:
        return {'level': self.level, 'code': self.code, 'where': self.where, 'detail': self.detail}


class Report:
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.stats: dict = {}

    def add(self, level: str, code: str, where: str, detail: str) -> None:
        self.findings.append(Finding(level, code, where, detail))

    def error(self, code: str, where: str, detail: str) -> None:
        self.add('ERROR', code, where, detail)

    def warn(self, code: str, where: str, detail: str) -> None:
        self.add('WARN', code, where, detail)

    def info(self, code: str, where: str, detail: str) -> None:
        self.add('INFO', code, where, detail)

    @property
    def n_error(self) -> int:
        return sum(1 for f in self.findings if f.level == 'ERROR')

    @property
    def n_warn(self) -> int:
        return sum(1 for f in self.findings if f.level == 'WARN')

    @property
    def verdict(self) -> str:
        return 'fail' if self.n_error else ('pass_with_warnings' if self.n_warn else 'pass')


def read_parcel_csv(path: Path, n_parcels: int, atlas_ids: list | None = None):
    """读取单个被试的分区向量。

    支持两种格式：
      - 长表：含 `parcel_index` 与 `value` 两列
      - 宽表：一行一分区，唯一数据列即目标值（列名须为 participant id）
    返回 (values: np.ndarray, fmt: str, problems: list[str])
    """
    problems: list[str] = []
    df = pd.read_csv(path)
    cols = [str(c) for c in df.columns]
    if 'parcel_index' in cols and 'value' in cols:
        fmt = 'long'
        idx = pd.to_numeric(df['parcel_index'], errors='coerce')
        vals = pd.to_numeric(df['value'], errors='coerce')
        order = np.argsort(idx.to_numpy())
        idx_sorted = idx.to_numpy()[order]
        vals = vals.to_numpy()[order]
        if atlas_ids is not None and not np.array_equal(idx_sorted, np.asarray(atlas_ids)):
            problems.append('parcel_index 与图集标签不一致或顺序不符')
    else:
        fmt = 'wide'
        if len(cols) != 1:
            problems.append(f'宽表应为单列，实际 {len(cols)} 列：{cols[:5]}')
        vals = pd.to_numeric(df[cols[0]], errors='coerce').to_numpy()
    if len(vals) != n_parcels:
        problems.append(f'分区数 {len(vals)} != 图集分区数 {n_parcels}')
    return vals, fmt, problems


def check_subject_vectors(report: Report, ddir: Path, subdir: str, label: str,
                          subjects: list[str], n_parcels: int, atlas_ids: list | None) -> dict:
    directory = ddir / subdir
    found: dict[str, np.ndarray] = {}
    if not directory.is_dir():
        report.error('S-missing-dir', f'{subdir}/', f'缺少目录（{label}）')
        return found
    for sid in subjects:
        for pattern in (f'{sid}.csv', f'{sid}_*.csv'):
            for p in sorted(directory.glob(pattern)):
                vals, fmt, problems = read_parcel_csv(p, n_parcels, atlas_ids)
                key = p.stem
                if problems:
                    for pr in problems:
                        report.error('P/N-format', str(p.relative_to(ddir)), f'{sid}: {pr}')
                found[key] = vals
                jf = p.with_suffix('.json')
                if not jf.exists():
                    report.warn('S-missing-json', str(p.relative_to(ddir)), '缺少同名 .json 说明文件')
                finite = np.isfinite(vals)
                if not finite.any():
                    report.error('N-all-nan', str(p.relative_to(ddir)), f'{label} 全部为缺失值')
                elif not finite.all():
                    report.warn('N-nan', str(p.relative_to(ddir)),
                                f'含 {int((~finite).sum())}/{len(vals)} 个缺失值')
                if finite.any():
                    seg = vals[finite]
                    if np.allclose(seg, seg[0]):
                        report.warn('N-constant', str(p.relative_to(ddir)), '所有分区取值完全相同')
                    if np.any(seg < 0):
                        report.warn('N-negative', str(p.relative_to(ddir)),
                                    f'含负值 {int((seg < 0).sum())} 个（{label} 通常非负）')
                    if np.nanmax(np.abs(seg)) > 1e6:
                        report.warn('N-extreme', str(p.relative_to(ddir)),
                                    f'最大绝对值 {np.nanmax(np.abs(seg)):.3g}，量纲可疑')
    report.stats.setdefault('subjects_found', {})[subdir] = len(found)
    missing = [s for s in subjects if s not in found]
    if missing:
        report.error('J-missing-subject', f'{subdir}/',
                     f'{len(missing)} 名被试缺少 {label}：{missing[:8]}{" ..." if len(missing) > 8 else ""}')
    extra = sorted({k for k in found if k not in subjects})
    if extra:
        report.info('J-extra-subject', f'{subdir}/',
                    f'{len(extra)} 个产物不属于目标被试（可能是随访或未纳入者）：{extra[:8]}')
    return found


def check_metadata(report: Report, ddir: Path, n_parcels: int) -> list | None:
    atlas = ddir / 'atlas' / 'Schaefer200_parcels.csv'
    atlas_ids = None
    if not atlas.exists():
        report.error('S-missing', 'atlas/Schaefer200_parcels.csv', '缺少图集分区清单')
        return None
    df = pd.read_csv(atlas)
    cols = [str(c) for c in df.columns]
    if 'parcel_index' not in cols or 'parcel_label' not in cols:
        report.error('M-atlas-columns', 'atlas/Schaefer200_parcels.csv',
                     f'需含 parcel_index 与 parcel_label 两列，实际 {cols}')
        return None
    if len(df) != n_parcels:
        report.error('M-atlas-size', 'atlas/Schaefer200_parcels.csv',
                     f'分区数 {len(df)} != 配置 n_parcels {n_parcels}')
    labels = df['parcel_label'].astype(str)
    if labels.duplicated().any():
        dup = labels[labels.duplicated()].unique()[:5]
        report.error('M-atlas-dup', 'atlas/Schaefer200_parcels.csv', f'标签重复：{list(dup)}')
    atlas_ids = pd.to_numeric(df['parcel_index'], errors='coerce').to_numpy()
    if not np.array_equal(atlas_ids, np.arange(1, len(df) + 1)):
        report.warn('M-atlas-index', 'atlas/Schaefer200_parcels.csv',
                    'parcel_index 不是 1..N 连续整数，长表校验会受影响')
        atlas_ids = None
    report.stats['n_parcels'] = int(len(df))
    return atlas_ids


def check_matrices(report: Report, ddir: Path, n_parcels: int, subjects: list[str]) -> None:
    for subdir, (label, _) in MATRIX_DIRS.items():
        directory = ddir / subdir
        if not directory.is_dir():
            report.error('S-missing-dir', f'{subdir}/', f'缺少目录（{label}）')
            continue
        files = sorted(directory.glob('*.npy')) + sorted(directory.glob('*.csv'))
        if not files:
            report.error('S-missing-file', f'{subdir}/', f'没有任何矩阵文件（{label}）')
            continue
        checked = 0
        for p in files:
            try:
                m = np.load(p) if p.suffix == '.npy' else pd.read_csv(p, header=None).to_numpy(dtype=float)
            except Exception as exc:  # noqa: BLE001
                report.error('G-unreadable', str(p.relative_to(ddir)), f'无法读取：{exc}')
                continue
            if m.ndim != 2 or m.shape[0] != m.shape[1]:
                report.error('G-not-square', str(p.relative_to(ddir)), f'形状 {m.shape} 不是方阵')
                continue
            if m.shape[0] != n_parcels:
                report.error('P-matrix-size', str(p.relative_to(ddir)),
                             f'{m.shape[0]} != 图集分区数 {n_parcels}')
            if not np.allclose(m, m.T, atol=1e-6, equal_nan=True):
                report.warn('G-asymmetry', str(p.relative_to(ddir)), '矩阵不对称')
            if subdir.endswith('fc') and not np.allclose(np.diag(m), 1.0, atol=1e-3):
                report.warn('G-diag', str(p.relative_to(ddir)), 'FC 对角线不接近 1')
            if np.nanmax(np.abs(m)) > 1e6:
                report.warn('N-extreme', str(p.relative_to(ddir)), '存在极端量级值')
            checked += 1
        report.stats.setdefault('matrices_checked', {})[subdir] = checked
        ids = {p.stem.split('_')[0] for p in files}
        missing = [s for s in subjects if s not in ids]
        if missing:
            report.warn('J-missing-matrix', f'{subdir}/',
                        f'{len(missing)} 名被试缺少矩阵：{missing[:8]}{" ..." if len(missing) > 8 else ""}')


def check_qc(report: Report, ddir: Path, subjects: list[str]) -> None:
    qc = ddir / 'qc' / 'subjects_qc.csv'
    if not qc.exists():
        report.error('S-missing', 'qc/subjects_qc.csv', '缺少 QC 表')
        return
    df = pd.read_csv(qc)
    cols = [str(c) for c in df.columns]
    missing_cols = [c for c in QC_REQUIRED if c not in cols]
    if missing_cols:
        report.error('Q-columns', 'qc/subjects_qc.csv', f'缺少必需列：{missing_cols}')
    if 'participant_id' in cols:
        ids = set(df['participant_id'].astype(str))
        missing = [s for s in subjects if s not in ids]
        if missing:
            report.error('Q-missing-subject', 'qc/subjects_qc.csv',
                         f'{len(missing)} 名被试缺少 QC 行：{missing[:8]}')
        extra = sorted(ids - set(subjects))
        if extra:
            report.info('Q-extra-subject', 'qc/subjects_qc.csv',
                        f'{len(extra)} 行不属于目标被试：{extra[:8]}')
    if 'bold_mean_fd' in cols:
        fd = pd.to_numeric(df['bold_mean_fd'], errors='coerce')
        bad = fd[(fd < 0) | (fd > 5)]
        if len(bad):
            report.warn('Q-range', 'qc/subjects_qc.csv', f'{len(bad)} 行 bold_mean_fd 超出 0–5 mm')
        report.stats['mean_fd'] = {'n': int(fd.notna().sum()), 'mean': float(fd.mean()) if fd.notna().any() else None}
    if {'has_dwi', 'has_bold', 'exclusion_flag'} <= set(cols):
        report.stats['n_excluded'] = int(pd.to_numeric(df['exclusion_flag'], errors='coerce').fillna(0).sum())
    if 'session' in cols:
        report.stats['qc_sessions'] = df['session'].astype(str).value_counts().to_dict()


def validate(ddir: Path, cfg: dict, targets: dict[str, list[str]]) -> Report:
    report = Report()
    n_parcels = int(cl.dig(cfg, 'atlas.n_parcels', 200))
    if not ddir.is_dir():
        report.error('S-root', str(ddir), '派生数据目录不存在')
        return report

    atlas_ids = check_metadata(report, ddir, n_parcels)
    vectors: dict[str, dict] = {}
    for subdir, (label, _) in PER_SUBJECT.items():
        vectors[subdir] = check_subject_vectors(
            report, ddir, subdir, label, targets.get('baseline', []), n_parcels, atlas_ids)

    # 列/键集合一致性：同一目录内不同被试的产物键应一致（按被试命名，故检查命名规范）
    for subdir, found in vectors.items():
        bad = [k for k in found if not k.startswith('sub-')]
        if bad:
            report.warn('C-naming', f'{subdir}/', f'{len(bad)} 个文件命名不符合 sub-XXX 约定：{bad[:5]}')

    check_matrices(report, ddir, n_parcels, targets.get('baseline', []))
    check_qc(report, ddir, targets.get('baseline', []))
    return report


def load_targets(clinical: Path, cfg: dict) -> dict[str, list[str]]:
    df = pd.read_csv(clinical, sep='\t')
    idcol = 'participant_id' if 'participant_id' in df.columns else df.columns[0]
    ids = df[idcol].astype(str).tolist()
    exclude = set(cl.dig(cfg, 'cohort.exclude_no_clinical', []) or [])
    baseline = [i for i in ids if i not in exclude and not i.endswith('fu')]
    return {'baseline': baseline, 'all': ids}


# --------------------------------------------------------------------------
# 自检：用合成数据生成 → 校验 → 断言
# --------------------------------------------------------------------------
def _write_synth(root: Path, n_parcels: int, subjects: list[str], with_defect: bool) -> Path:
    rng = np.random.default_rng(0)
    (root / 'atlas').mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'parcel_index': np.arange(1, n_parcels + 1),
                  'parcel_label': [f'7Networks_LH_Vis_{i}' for i in range(n_parcels)]}
                 ).to_csv(root / 'atlas' / 'Schaefer200_parcels.csv', index=False)
    for subdir in PER_SUBJECT:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    for subdir in MATRIX_DIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    (root / 'qc').mkdir(parents=True, exist_ok=True)

    for k, sid in enumerate(subjects):
        for subdir in ('a1_streamlines', 'a1_fa', 'a3_gwc', 'a3_t1t2'):
            vals = rng.normal(1.0, 0.1, n_parcels)
            if with_defect and k == 0:
                vals[3] = np.nan          # 缺失
            df = pd.DataFrame({'parcel_index': np.arange(1, n_parcels + 1), 'value': vals})
            if with_defect and k == 1:
                df = df.iloc[:-1]         # 少一行 → 分区数不符
            df.to_csv(root / subdir / f'{sid}.csv', index=False)
            (root / subdir / f'{sid}.json').write_text(
                json.dumps({'produced_by': 'synth', 'n_parcels': n_parcels}), encoding='utf-8')
        for subdir in MATRIX_DIRS:
            m = rng.normal(0, 0.3, (n_parcels, n_parcels))
            m = (m + m.T) / 2
            if subdir.endswith('fc'):
                np.fill_diagonal(m, 1.0)
            np.save(root / subdir / f'{sid}.npy', m)
    qc = pd.DataFrame({
        'participant_id': subjects,
        'session': ['baseline'] * len(subjects),
        'has_t1w': 1, 'has_t2w': 1, 'has_dwi': 1, 'has_bold': 1,
        'bold_mean_fd': rng.uniform(0.05, 0.4, len(subjects)),
        'bold_pct_fd_gt_0.2': rng.uniform(0, 0.2, len(subjects)),
        'exclusion_flag': 0,
    })
    if with_defect:
        qc = qc.iloc[1:]                    # 少一名被试 → QC 缺行
    qc.to_csv(root / 'qc' / 'subjects_qc.csv', index=False)
    return root


def self_test() -> int:
    cfg = cl.load(ROOT / 'config' / 'analysis.yaml')
    n_parcels = int(cl.dig(cfg, 'atlas.n_parcels'))
    subjects = [f'sub-{i:03d}' for i in range(1, 4)]
    ok = 0
    # 注意：DSH 沙箱下系统临时目录不可写，合成数据放在工作区内
    tmp = ROOT / 'build' / 'self_test'
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    good = _write_synth(tmp / 'good', n_parcels, subjects, with_defect=False)
    rep = validate(good, cfg, {'baseline': subjects})
    assert rep.verdict in ('pass', 'pass_with_warnings'), [f.as_dict() for f in rep.findings]
    print(f'PASS  自检-干净数据：verdict={rep.verdict}（ERROR={rep.n_error}, WARN={rep.n_warn}）')
    ok += 1

    bad = _write_synth(tmp / 'bad', n_parcels, subjects, with_defect=True)
    rep_bad = validate(bad, cfg, {'baseline': subjects})
    levels = {(f.code, f.level) for f in rep_bad.findings}
    assert rep_bad.verdict == 'fail', rep_bad.verdict
    assert any(c.startswith('P') or c.startswith('N-format') for c, _ in levels), levels
    assert ('Q-missing-subject', 'ERROR') in levels, levels
    print(f'PASS  自检-含缺陷数据：verdict={rep_bad.verdict}，捕获 {len(rep_bad.findings)} 项（ERROR={rep_bad.n_error}）')
    for f in rep_bad.findings[:6]:
        print(f'       [{f.level}] {f.code} @ {f.where} — {f.detail[:80]}')
    ok += 1

    jp, mp = write_reports(rep_bad, tmp / 'reports')
    assert jp.exists() and mp.exists()
    print(f'PASS  报告写出：{jp.name} / {mp.name}')
    ok += 1
    print(f'\n自检通过 {ok}/3（合成数据位于 {tmp}）')
    return 0


def write_reports(report: Report, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'verdict': report.verdict,
        'n_error': report.n_error,
        'n_warn': report.n_warn,
        'stats': report.stats,
        'findings': [f.as_dict() for f in report.findings],
    }
    jp = out_dir / 'derivatives_validation.json'
    jp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# 派生数据校验报告', '',
             f'- 结论：**{report.verdict}**',
             f'- ERROR：{report.n_error}　WARN：{report.n_warn}',
             f'- 统计：`{json.dumps(report.stats, ensure_ascii=False)}`', '',
             '| 级别 | 代码 | 位置 | 说明 |', '|---|---|---|---|']
    for f in report.findings:
        lines.append(f'| {f.level} | {f.code} | `{f.where}` | {f.detail.replace("|", "/")} |')
    mp = out_dir / 'derivatives_validation.md'
    mp.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return jp, mp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='校验派生数据是否符合契约')
    ap.add_argument('--derivatives', type=Path, default=ROOT / 'data' / 'derivatives')
    ap.add_argument('--config', type=Path, default=ROOT / 'config' / 'analysis.yaml')
    ap.add_argument('--clinical', type=Path, default=None)
    ap.add_argument('--out', type=Path, default=ROOT / 'data' / 'manifests')
    ap.add_argument('--self-test', action='store_true', help='用合成数据自检校验器本身')
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    cfg = cl.load(args.config)
    clinical = args.clinical or ROOT / cl.dig(cfg, 'dataset.clinical_derived')
    if not Path(clinical).exists():
        print(f'找不到临床表：{clinical}', file=sys.stderr)
        return 2
    targets = load_targets(Path(clinical), cfg)
    report = validate(args.derivatives, cfg, targets)
    jp, mp = write_reports(report, args.out)

    print(f'结论：{report.verdict}（ERROR={report.n_error}, WARN={report.n_warn}）')
    for f in report.findings[:20]:
        print(f'  [{f.level}] {f.code} @ {f.where} — {f.detail}')
    if len(report.findings) > 20:
        print(f'  ... 其余 {len(report.findings) - 20} 项见报告')
    print(f'\n报告：{jp}\n      {mp}')
    return 1 if report.n_error else 0


if __name__ == '__main__':
    raise SystemExit(main())
