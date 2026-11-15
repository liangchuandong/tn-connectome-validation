#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按科学优先级分片下载 OpenNeuro ds005713。

设计要点
--------
1. **以 manifest 为唯一真相来源**：不拼接 URL 猜测文件名。本数据集的命名并不
   统一（`sub-009_run-01_T1w.nii.gz` 有 run-01，而 `sub-023_T1w.nii.gz` 没有），
   任何"按规则拼路径"的写法都会 404。文件清单来自
   `data/manifests/ds005713_files.csv`（S3 全量对象枚举，含权威字节数）。
2. **字节数即校验和**：S3 的 ETag 在分片上传时不等于 MD5，不能当校验和用。
   本脚本用 manifest 里的 `size_bytes` 做完整性判据——已存在且字节数完全相等
   则跳过，因此**中断后重跑即断点续传**。
3. **支持 Range 续传**：单个大文件下载中断后，下次用 Range 请求接着写。
4. **S3 直连**：`https://s3.amazonaws.com/openneuro.org`，匿名只读。
   注意：本机 `curl` / `Invoke-WebRequest` 会因 schannel 凭证问题失败，
   必须用 Python `urllib` 或 `wget`。

分片定义（按"最早能产出可用结果"排序）
--------------------------------------
- `meta`  顶层元数据 + mriqc 组表（<1.5 MB）→ 解锁临床冻结与 QC 表
- `anat`  T1w + T2w（26.01 GiB）→ 解锁 A3（GWC / T1w/T2w）全链路
- `dwi`   含反向 b0（6.85 GiB）→ 解锁 A1（topup/eddy/追踪）
- `func`  rs-fMRI（10.58 GiB）→ 解锁 A2（图谐波耦合）
- `all`   全量（43.44 GiB）

分阶段下载（推荐用法）
----------------------
给 **多个** `--slice` 即进入"分阶段顺序下载"模式：按给定顺序逐片下完再进下一片，
每片完成后自动跑一次完整性校验并打印阶段小结。中断后重跑会跳过已完成文件。

    # 推荐：一条命令按科学优先级分阶段下载
    python scripts/fetch_slice.py --dest F:\\ds005713\\raw --workers 8 ^
        --slice meta --slice anat --slice dwi --slice func

    # 只要解锁 A3 就够（下完 anat 自动停止）
    python scripts/fetch_slice.py --dest F:\\ds005713\\raw --slice meta --slice anat

    # 单分片（旧的用法仍然有效）
    python scripts/fetch_slice.py --dest F:\\ds005713\\raw --slice anat

    # 只列清单不下载
    python scripts/fetch_slice.py --dest X --slice anat --dry-run

    # 只看当前完成度，不下载
    python scripts/verify_download.py --dest F:\\ds005713\\raw
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BUCKET = "https://s3.amazonaws.com/openneuro.org"
DATASET = "ds005713"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "data", "manifests", "ds005713_files.csv")

# 分片 -> 判定函数（rel = 去掉 "ds005713/" 前缀后的相对路径）
SLICES = {
    "meta": lambda rel: ("/" not in rel) or rel.startswith("mriqc_data/"),
    "anat": lambda rel: "/anat/" in rel,
    "dwi": lambda rel: "/dwi/" in rel,
    "func": lambda rel: "/func/" in rel,
    "all": lambda rel: True,
}

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def load_manifest(path: str) -> list[tuple[str, int]]:
    """返回 [(key, size_bytes), ...]，key 形如 'ds005713/sub-009/anat/...'。"""
    out: list[tuple[str, int]] = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = row["key"].strip()
            if not key or key.endswith("/"):
                continue
            out.append((key, int(row["size_bytes"])))
    return out


def select(objects, slice_name: str) -> list[tuple[str, int | None]]:
    """按分片筛选，并把 size 归一化为 None（调用方自行处理）。"""
    pred = SLICES[slice_name]
    sel = []
    for key, size in objects:
        rel = key.split("/", 1)[1] if "/" in key else key
        if pred(rel):
            sel.append((key, size))
    return sel


def local_path(dest: str, key: str) -> str:
    rel = key.split("/", 1)[1] if "/" in key else key
    return os.path.join(dest, rel.replace("/", os.sep))


def fetch_one(url: str, dest: str, expect_size: int, retries: int = 5,
              timeout: int = 180, chunk: int = 1 << 20) -> tuple[str, int, str]:
    """下载单文件。返回 (dest, final_size, status)。status ∈ ok/skip/fail。"""
    if os.path.exists(dest) and os.path.getsize(dest) == expect_size:
        return dest, expect_size, "skip"

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last_err = ""
    for attempt in range(retries):
        try:
            have = os.path.getsize(dest) if os.path.exists(dest) else 0
            if have > expect_size:          # 损坏的残留，重下
                os.remove(dest)
                have = 0
            req = urllib.request.Request(url, headers={"User-Agent": "ds005713-fetch/1.0"})
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                # 若服务端不支持续传，会返回 200，此时必须从头写
                mode = "ab" if (have and getattr(resp, "status", 200) == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(dest, mode) as fh:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        fh.write(buf)
            got = os.path.getsize(dest)
            if got == expect_size:
                return dest, got, "ok"
            last_err = f"size mismatch got={got} expect={expect_size}"
        except Exception as exc:                       # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt == retries - 1:
                break
            time.sleep(min(2 ** attempt, 30))
    return dest, (os.path.getsize(dest) if os.path.exists(dest) else 0), f"fail ({last_err})"


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def run_slice(slice_name: str, objects, dest: str, workers: int, retries: int,
              dry_run: bool = False) -> dict:
    """下载单个分片。返回统计 dict。"""
    wanted = select(objects, slice_name)

    todo, total, skipped = [], 0, 0
    for key, size in wanted:
        p = local_path(dest, key)
        if os.path.exists(p) and os.path.getsize(p) == size:
            skipped += 1
            continue
        todo.append((key, size, p))
        total += size

    print("=" * 78)
    print(f"分片        : {slice_name}")
    print(f"清单文件数  : {len(wanted)}")
    print(f"已完成跳过  : {skipped}")
    print(f"待下载      : {len(todo)} 个文件 / {human(total)}")
    print(f"落盘目录    : {dest}")
    print(f"并发        : {workers}")
    print("=" * 78)

    if dry_run:
        for key, size, _ in todo[:50]:
            print(f"  {human(size):>10s}  {key}")
        if len(todo) > 50:
            print(f"  ... 其余 {len(todo) - 50} 个")
        return {"slice": slice_name, "n": len(todo), "bytes": total,
                "failed": 0, "minutes": 0.0, "dry_run": True}

    if not todo:
        print("全部已就绪，无需下载。")
        return {"slice": slice_name, "n": 0, "bytes": 0, "failed": 0, "minutes": 0.0}

    done = {"n": 0, "bytes": 0, "fail": 0}
    t0 = time.time()
    lock = threading.Lock()

    def task(item):
        key, size, p = item
        p, got, status = fetch_one(f"{BUCKET}/{key}", p, size, retries)
        with lock:
            done["n"] += 1
            if status == "fail":
                done["fail"] += 1
                log(f"[{done['n']}/{len(todo)}] FAIL {key} :: {status}")
            else:
                done["bytes"] += got
                el = time.time() - t0
                rate = done["bytes"] / el / 1e6 if el > 0 else 0
                pct = 100.0 * done["bytes"] / total if total else 100.0
                eta = (total - done["bytes"]) / (done["bytes"] / el) if done["bytes"] else 0
                log(f"[{done['n']}/{len(todo)}] {pct:5.1f}%  "
                    f"{rate:6.2f} MB/s  ETA {eta/60:6.1f} min  {key}")
        return status

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(task, todo))

    el = (time.time() - t0) / 60
    avg = done["bytes"] / 1e6 / (el * 60) if el else 0
    print("-" * 78)
    print(f"分片 {slice_name} 完成：成功 {done['n'] - done['fail']}/{len(todo)}，"
          f"失败 {done['fail']}，耗时 {el:.1f} min，平均 {avg:.2f} MB/s")
    return {"slice": slice_name, "n": len(todo), "bytes": total,
            "failed": done["fail"], "minutes": el, "mbps": avg}


def verify_stage(dest: str, slice_name: str) -> dict | None:
    """调 verify_download.py 做一次分片校验（失败不影响下载继续）。"""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_download.py")
    if not os.path.exists(script):
        return None
    import subprocess
    try:
        r = subprocess.run([sys.executable, script, "--dest", dest,
                            "--slice", slice_name, "--quiet"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=3600)
        return {"returncode": r.returncode, "stdout": (r.stdout or "").strip()}
    except Exception as exc:                          # noqa: BLE001
        return {"returncode": -1, "stdout": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description="按分片下载 ds005713（支持分阶段顺序下载）")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--dest", required=True, help="落盘根目录，BIDS 内容直接放在这里")
    ap.add_argument("--slice", required=True, action="append", choices=sorted(SLICES),
                    help="可重复指定；多个即按顺序分阶段下载")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="只列清单与体积，不下载")
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--no-verify", action="store_true",
                    help="分阶段模式下跳过每片完成后的自动校验")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        print(f"[ERR] manifest 不存在: {args.manifest}", file=sys.stderr)
        return 2

    objects = load_manifest(args.manifest)
    stages = args.slice
    staged = len(stages) > 1

    if staged:
        print(f"分阶段下载：{' → '.join(stages)}")
        print(f"合计体积估算：", end="", flush=True)
        grand = 0
        for s in stages:
            grand += sum(sz for _, sz in select(objects, s))
        print(human(grand))
        print()

    os.makedirs(args.dest, exist_ok=True)
    results, failed_slices, t_all = [], [], time.time()

    for i, s in enumerate(stages, 1):
        if staged:
            print(f"\n########## 阶段 {i}/{len(stages)}：{s} ##########")
        res = run_slice(s, objects, args.dest, args.workers, args.retries, args.dry_run)
        results.append(res)

        if args.dry_run:
            continue

        if staged and not args.no_verify and res["n"] > 0:
            print(f"--- 校验分片 {s} ---")
            v = verify_stage(args.dest, s)
            if v:
                print(v["stdout"] or f"(returncode={v['returncode']})")

        if res["failed"]:
            failed_slices.append(s)

    # 阶段小结
    if staged and not args.dry_run:
        total_min = (time.time() - t_all) / 60
        print("\n" + "=" * 78)
        print("分阶段下载小结")
        print("=" * 78)
        print(f"{'阶段':<8}{'文件数':>8}{'体积':>12}{'失败':>8}{'耗时':>10}{'速率':>12}")
        for r in results:
            if r.get("dry_run"):
                print(f"{r['slice']:<8}{r['n']:>8}{human(r['bytes']):>12}{'-':>8}{'-':>10}{'-':>12}")
                continue
            print(f"{r['slice']:<8}{r['n']:>8}{human(r['bytes']):>12}"
                  f"{r['failed']:>8}{r['minutes']:>9.1f}m{r.get('mbps', 0):>10.2f} MB/s")
        print("-" * 78)
        print(f"总耗时 {total_min:.1f} min")
        if failed_slices:
            print(f"[!] 有失败的分片：{failed_slices} —— 直接重跑同一命令即可续传。")
            return 1
        print("[OK] 全部阶段完成。建议再跑一次 verify_download.py --deep 做 gzip 级校验。")
    elif not args.dry_run and failed_slices:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
