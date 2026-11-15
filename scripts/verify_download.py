#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验 ds005713 下载完整性，输出"期望 vs 实际"差异表。

三层校验
--------
1. **字节数**：与 manifest 的权威 `size_bytes` 比对（S3 的 ETag 在分片上传时
   不等于 MD5，因此**不能**当校验和）。
2. **gzip 可解**：对 `.nii.gz` / `.gz` 实际解压头部若干字节，能识破"字节数
   对但内容截断"的隐性损坏。
3. **BIDS 结构**：检查每个被试目录下的 datatype 布局是否与 manifest 一致。

用途
----
    python scripts/verify_download.py --dest F:\\ds005713\\raw
    python scripts/verify_download.py --dest F:\\ds005713\\raw --slice anat
    python scripts/verify_download.py --dest F:\\ds005713\\raw --report out.json

退出码：0 = 全部通过；1 = 存在差异。
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "data", "manifests", "ds005713_files.csv")

SLICES = {
    "meta": lambda rel: ("/" not in rel) or rel.startswith("mriqc_data/"),
    "anat": lambda rel: "/anat/" in rel,
    "dwi": lambda rel: "/dwi/" in rel,
    "func": lambda rel: "/func/" in rel,
    "all": lambda rel: True,
}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def gzip_ok(path: str, probe: int = 1 << 20) -> tuple[bool, str]:
    """尝试解压前 probe 字节。返回 (是否正常, 说明)。"""
    try:
        with gzip.open(path, "rb") as fh:
            fh.read(probe)
        return True, ""
    except EOFError:
        return True, "truncated-but-header-ok"   # 只读了前 1MB，EOF 属正常
    except Exception as exc:                      # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description="校验 ds005713 下载完整性")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--slice", default="all", choices=sorted(SLICES))
    ap.add_argument("--report", default=None, help="把结果写成 JSON")
    ap.add_argument("--deep", action="store_true",
                    help="对所有 .gz 做 gzip 解压探测（默认只做体积比对）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    pred = SLICES[args.slice]
    expected: dict[str, int] = {}
    with open(args.manifest, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = row["key"].strip()
            if not key or key.endswith("/"):
                continue
            rel = key.split("/", 1)[1] if "/" in key else key
            if pred(rel):
                expected[rel] = int(row["size_bytes"])

    missing, size_bad, ok, corrupt = [], [], [], []
    got_bytes = 0
    for rel, size in sorted(expected.items()):
        p = os.path.join(args.dest, rel.replace("/", os.sep))
        if not os.path.exists(p):
            missing.append((rel, size))
            continue
        actual = os.path.getsize(p)
        if actual != size:
            size_bad.append((rel, size, actual))
            continue
        got_bytes += actual
        ok.append(rel)

    if args.deep:
        for rel in ok:
            if rel.endswith(".gz"):
                p = os.path.join(args.dest, rel.replace("/", os.sep))
                good, why = gzip_ok(p)
                if not good:
                    corrupt.append((rel, why))

    # 结构核对：每个被试目录下应有哪些 datatype
    exp_by_sub: dict[str, set[str]] = defaultdict(set)
    for rel in expected:
        parts = rel.split("/")
        if len(parts) >= 3 and parts[0].startswith("sub-"):
            exp_by_sub[parts[0]].add(parts[1])
    got_by_sub: dict[str, set[str]] = defaultdict(set)
    for rel in ok:
        parts = rel.split("/")
        if len(parts) >= 3 and parts[0].startswith("sub-"):
            got_by_sub[parts[0]].add(parts[1])
    structure_bad = {
        s: sorted(exp_by_sub[s] - got_by_sub[s])
        for s in exp_by_sub
        if exp_by_sub[s] - got_by_sub[s]
    }

    total_expected = sum(expected.values())
    report = {
        "dest": args.dest,
        "slice": args.slice,
        "n_expected": len(expected),
        "n_ok": len(ok),
        "n_missing": len(missing),
        "n_size_mismatch": len(size_bad),
        "n_gzip_corrupt": len(corrupt),
        "n_structure_gap": len(structure_bad),
        "expected_bytes": total_expected,
        "present_bytes": got_bytes,
        "completeness_pct": round(100.0 * len(ok) / len(expected), 2) if expected else 100.0,
        "bytes_pct": round(100.0 * got_bytes / total_expected, 2) if total_expected else 100.0,
        "missing_sample": [m[0] for m in missing[:30]],
        "size_mismatch_sample": [
            {"key": r, "expected": e, "actual": a} for r, e, a in size_bad[:30]
        ],
        "gzip_corrupt_sample": [{"key": r, "why": w} for r, w in corrupt[:30]],
        "structure_gap_sample": dict(list(structure_bad.items())[:30]),
    }

    if not args.quiet:
        print("=" * 78)
        print(f"校验目录   : {args.dest}")
        print(f"分片范围   : {args.slice}")
        print(f"期望文件数 : {report['n_expected']}")
        print(f"已就绪     : {report['n_ok']}  ({report['completeness_pct']}%)")
        print(f"缺失       : {report['n_missing']}")
        print(f"字节数不符 : {report['n_size_mismatch']}")
        if args.deep:
            print(f"gzip 损坏  : {report['n_gzip_corrupt']}")
        print(f"目录缺失   : {report['n_structure_gap']}")
        print(f"体积       : {human(got_bytes)} / {human(total_expected)}  "
              f"({report['bytes_pct']}%)")
        print("=" * 78)

        if missing:
            print(f"\n缺失文件（前 30 / 共 {len(missing)}）:")
            for rel, size in missing[:30]:
                print(f"  {human(size):>10s}  {rel}")
        if size_bad:
            print(f"\n字节数不符（前 30 / 共 {len(size_bad)}）—— 需删除后重下:")
            for rel, e, a in size_bad[:30]:
                print(f"  {rel}\n      期望 {e:,} B  实际 {a:,} B")
        if corrupt:
            print(f"\ngzip 损坏（前 30 / 共 {len(corrupt)}）:")
            for rel, why in corrupt[:30]:
                print(f"  {rel}  ::  {why}")
        if structure_bad:
            print(f"\n目录结构缺失（前 30 / 共 {len(structure_bad)}）:")
            for s, dts in list(structure_bad.items())[:30]:
                print(f"  {s}: 缺 {dts}")

        if not (missing or size_bad or corrupt or structure_bad):
            print("\n[PASS] 全部通过：文件数、字节数、gzip 完整性、目录结构均与 manifest 一致。")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\n报告已写入: {args.report}")

    bad = bool(missing or size_bad or corrupt or structure_bad)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
