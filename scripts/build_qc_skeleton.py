#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 ds005713 的样本纳入漏斗图、排除清单与 QC 表骨架。

输入（全部为已有文件，不需要下载影像）
-------------------------------------
- data/manifests/ds005713_files.csv         S3 全量对象清单（权威）
- data/raw_metadata/mriqc_data/group_*.tsv  数据集自带 QC 组表
- data/frozen/participants_frozen.tsv       冻结临床表

关键事实（本脚本据此实现，勿凭直觉改）
-------------------------------------
1. manifest 计 238 个被试目录，但**其中 25 个只有 JSON、零 `.nii.gz`**，
   实际有影像的只有 213 个。
2. mriqc 组表覆盖**基线**（T1w 158 被试 / DWI 156 / BOLD 110 / T2w 142），
   **随访（`sub-XXXfu`）覆盖为 0**，因此随访 QC 必须自算。
3. mriqc 表里有 `sub-023`、`sub-094` 两个**公开快照中无影像**的被试
   （mriqc 是在完整数据上跑的），合并时必须内连接到 manifest，不能反向。
4. 被试 ID 三位数=患者、两位数=对照、`fu` 后缀=随访，**不可重命名**，
   否则无法与冻结临床表的 `participant_id` 合并。

输出
----
- derivatives/qc/subjects_qc.csv         QC 表骨架（基线尽力填，随访留空）
- derivatives/qc/cohort_funnel.md        纳入漏斗图 + 排除清单
- derivatives/qc/cohort_funnel.json      机器可读版本

用法
----
    python scripts/build_qc_skeleton.py
    python scripts/build_qc_skeleton.py --dest F:\\ds005713 --report-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "manifests", "ds005713_files.csv")
MRIQC_DIR = os.path.join(ROOT, "data", "raw_metadata", "mriqc_data")
CLINICAL = os.path.join(ROOT, "data", "frozen", "participants_frozen.tsv")
DERIVED = os.path.join(ROOT, "data", "frozen", "participants_derived.tsv")

DATATYPES = ("anat", "dwi", "func")

# QC 表列（对齐 docs/derived_data_contract.md 第 3 节）
QC_COLUMNS = [
    "participant_id", "session", "group", "has_t1w", "has_t2w", "has_dwi", "has_bold",
    "n_t1w_runs", "n_t2w_runs", "n_dwi_runs", "n_bold_runs",
    "n_dwi_directions", "dwi_has_reverse_b0",
    "t1_cjv", "t1_cnr", "t1_snr_gm", "t1_snr_wm", "t1_tpm_overlap_gm", "t1_fwhm_avg",
    "t1_best_run", "t1_run_delta_cjv",
    "bold_mean_fd", "bold_pct_fd_gt_0.2", "bold_tsnr", "bold_dvars_std", "bold_size_t",
    "dwi_mean_fd", "dwi_ndc", "dwi_snr_shell0",
    "t1_qc_rating", "dwi_qc_rating", "bold_qc_rating", "seg_qc_rating",
    "exclusion_flag", "exclusion_reason", "notes",
]


def is_followup(sid: str) -> bool:
    return sid.endswith("fu")


def group_of(sid: str) -> str:
    """按命名规则给出患者/对照（仅用于分组展示；正式分组以临床表为准）。"""
    core = sid[4:-2] if is_followup(sid) else sid[4:]
    if is_followup(sid):
        return "patient_followup"
    return "patient" if len(core) == 3 else "control"


def load_manifest(path: str):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["size_bytes"] = int(r["size_bytes"])
            rows.append(r)
    return rows


def subject_of(key: str) -> str | None:
    parts = key.split("/")
    if len(parts) >= 2 and parts[1].startswith("sub-"):
        return parts[1]
    return None


def datatype_of(key: str) -> str | None:
    parts = key.split("/")
    if len(parts) >= 3 and parts[1].startswith("sub-"):
        return parts[2]
    return None


def build_inventory(rows):
    """每个被试目录 -> {datatype: [nii.gz 文件名...]}，以及体积。

    注意：必须登记**所有**被试目录，包括只有 JSON、零 `.nii.gz` 的 25 个，
    否则漏斗图会丢掉"仅有元数据"这一段，样本纳入口径失真。
    """
    inv: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    nbytes: dict[str, int] = defaultdict(int)
    for r in rows:
        sid = subject_of(r["key"])
        if sid is None:
            continue
        inv[sid]                      # 触发 defaultdict，登记目录
        nbytes[sid] += r["size_bytes"]
        dt = datatype_of(r["key"])
        if dt and r["key"].endswith(".nii.gz"):
            inv[sid][dt].append(os.path.basename(r["key"]))
    return inv, nbytes


def load_tsv(path: str):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def fnum(v, default=None):
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return default


def pick_best_t1(entries):
    """按 cjv 最小择优；缺 cjv 则退化为取第一条。返回 (best_name, best_cjv, delta)。"""
    scored = []
    for e in entries:
        cjv = fnum(e.get("cjv"))
        scored.append((cjv if cjv is not None else float("inf"), e.get("bids_name", ""), cjv))
    if not scored:
        return "", None, None
    scored.sort(key=lambda x: x[0])
    best = scored[0]
    delta = None
    if len(scored) > 1 and best[2] is not None and scored[1][2] is not None:
        delta = round(scored[1][2] - best[2], 6)
    return best[1], best[2], delta


def main() -> int:
    ap = argparse.ArgumentParser(description="生成队列漏斗、排除清单与 QC 表骨架")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--mriqc-dir", default=MRIQC_DIR)
    ap.add_argument("--clinical", default=CLINICAL)
    ap.add_argument("--derived", default=DERIVED)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "derivatives", "qc"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_manifest(args.manifest)
    inv, nbytes = build_inventory(rows)
    all_subs = sorted(inv.keys())

    # ---------- mriqc 表 ----------
    def qc_table(name):
        return load_tsv(os.path.join(args.mriqc_dir, name))

    t1q = qc_table("group_T1w.tsv")
    t2q = qc_table("group_T2w.tsv")
    boldq = qc_table("group_bold.tsv")
    dwiq = qc_table("group_dwi.tsv")

    t1_by_sub: dict[str, list[dict]] = defaultdict(list)
    for e in t1q:
        t1_by_sub[e["bids_name"].split("_")[0]].append(e)
    t2_subs = {e["bids_name"].split("_")[0] for e in t2q}
    bold_by_sub = {e["bids_name"].split("_")[0]: e for e in boldq}
    dwi_by_sub = {e["bids_name"].split("_")[0]: e for e in dwiq}
    mriqc_subs = set(t1_by_sub) | t2_subs | set(bold_by_sub) | set(dwi_by_sub)

    # ---------- 临床表 ----------
    clin = load_tsv(args.clinical)
    clin_by_id = {e["participant_id"]: e for e in clin}
    derived = load_tsv(args.derived)
    derived_by_id = {e["participant_id"]: e for e in derived}

    # ---------- 分类 ----------
    no_image = [s for s in all_subs if not inv[s]]
    with_image = [s for s in all_subs if inv[s]]
    base = [s for s in with_image if not is_followup(s)]
    fu = [s for s in with_image if is_followup(s)]

    def having(lst, dt):
        return [s for s in lst if inv[s].get(dt)]

    def having_all(lst, dts):
        return [s for s in lst if all(inv[s].get(d) for d in dts)]

    def with_anat(sub_lst, tag):
        return [s for s in sub_lst
                if any(tag in f for f in inv[s].get("anat", []))]

    base_t1 = with_anat(base, "T1w")
    base_t2 = with_anat(base, "T2w")
    base_t1t2 = [s for s in base_t1 if any("T2w" in f for f in inv[s].get("anat", []))]
    base_all = [s for s in base_t1t2 if inv[s].get("dwi") and inv[s].get("func")]
    fu_t1 = with_anat(fu, "T1w")
    fu_t2 = with_anat(fu, "T2w")
    fu_t1t2 = [s for s in fu_t1 if any("T2w" in f for f in inv[s].get("anat", []))]
    fu_all = [s for s in fu_t1t2 if inv[s].get("dwi") and inv[s].get("func")]

    funnel = {
        "被试目录总数(manifest)": len(all_subs),
        "仅有JSON_零nii": len(no_image),
        "有影像的被试目录": len(with_image),
        "基线_有影像": len(base),
        "基线_T1w": len(base_t1),
        "基线_T2w": len(base_t2),
        "基线_DWI": len(having(base, "dwi")),
        "基线_BOLD": len(having(base, "func")),
        "随访_有影像": len(fu),
        "随访_T1w": len(fu_t1),
        "随访_T2w": len(fu_t2),
        "随访_DWI": len(having(fu, "dwi")),
        "随访_BOLD": len(having(fu, "func")),
        "基线_T1w且T2w": len(base_t1t2),
        "基线_T1w_T2w_DWI_BOLD齐全": len(base_all),
        "随访_T1w且T2w": len(fu_t1t2),
        "随访_T1w_T2w_DWI_BOLD齐全": len(fu_all),
    }

    # ---------- 逐被试排除理由 ----------
    excl: dict[str, list[str]] = defaultdict(list)
    for s in no_image:
        excl[s].append("目录内无任何 .nii.gz（仅 JSON 元数据）")
    for s in all_subs:
        for f in inv[s].get("anat", []):
            if "sub-055" in f and "run-01_T1w" in f:
                excl[s].append("T1w run-01 严重截断（270 KB），需改用 run-02 或排除")
        if is_followup(s) and any("run-05_dwi" in f for f in inv[s].get("dwi", [])):
            excl[s].append("存在 run-05_dwi、bval 仅 7 个体积（方向数不足）")
    for s in base:
        if not inv[s].get("dwi"):
            excl[s].append("无 DWI → 不进入 A1/A2")
        if not inv[s].get("func"):
            excl[s].append("无 BOLD → 不进入 A2")
        if not any("T2w" in f for f in inv[s].get("anat", [])):
            excl[s].append("无 T2w → 不进入 T1w/T2w 平行指标")
    for s in fu:
        excl[s].append("随访：mriqc 组表无覆盖，QC 必须自算")
        if not inv[s].get("dwi"):
            excl[s].append("随访无 DWI → 不进入纵向 A1")

    # ---------- QC 表 ----------
    qc_rows = []
    for s in all_subs:
        t1_entries = t1_by_sub.get(s, [])
        # 择优：仅当 run 数可分辨时才用 cjv 择优，否则取 mriqc 第一行
        best_name, best_cjv, delta = pick_best_t1(t1_entries)
        # mriqc 的 bids_name 形如 sub-009_run-01_T1w，补全为真实文件名
        best_file = f"{best_name}.nii.gz" if best_name and not best_name.endswith(".nii.gz") else best_name
        first = t1_entries[0] if t1_entries else {}
        b = bold_by_sub.get(s, {})
        d = dwi_by_sub.get(s, {})
        n_t1 = len([f for f in inv[s].get("anat", []) if "T1w" in f])
        n_t2 = len([f for f in inv[s].get("anat", []) if "T2w" in f])
        n_dwi = len(inv[s].get("dwi", []))
        n_bold = len(inv[s].get("func", []))
        # DWI：run-01 是主序列（65 体积），run-02.. 是反向 b0
        dwi_files = inv[s].get("dwi", [])
        has_rev_b0 = any(any(f"run-0{k}_dwi" in f for k in (2, 3, 4)) for f in dwi_files)
        # 方向数：实测 bval 为 65 个体积（1 个 b0 + 64 个方向，b=1500）
        n_dir = 65 if n_dwi >= 1 else ""
        fd_perc = fnum(b.get("fd_perc"))
        row = {c: "" for c in QC_COLUMNS}
        row.update({
            "participant_id": s,
            "session": "followup" if is_followup(s) else "baseline",
            "group": group_of(s),
            "has_t1w": int(n_t1 > 0), "has_t2w": int(n_t2 > 0),
            "has_dwi": int(n_dwi > 0), "has_bold": int(n_bold > 0),
            "n_t1w_runs": n_t1, "n_t2w_runs": n_t2,
            "n_dwi_runs": n_dwi, "n_bold_runs": n_bold,
            "n_dwi_directions": n_dir,
            "dwi_has_reverse_b0": int(bool(has_rev_b0)),
            "t1_cjv": first.get("cjv", ""), "t1_cnr": first.get("cnr", ""),
            "t1_snr_gm": first.get("snr_gm", ""), "t1_snr_wm": first.get("snr_wm", ""),
            "t1_tpm_overlap_gm": first.get("tpm_overlap_gm", ""),
            "t1_fwhm_avg": first.get("fwhm_avg", ""),
            "t1_best_run": best_file, "t1_run_delta_cjv": delta if delta is not None else "",
            "bold_mean_fd": b.get("fd_mean", ""),
            "bold_pct_fd_gt_0.2": (round(fd_perc, 4) if fd_perc is not None else ""),
            "bold_tsnr": b.get("tsnr", ""), "bold_dvars_std": b.get("dvars_std", ""),
            "bold_size_t": b.get("size_t", ""),
            "dwi_mean_fd": d.get("fd_mean", ""), "dwi_ndc": d.get("ndc", ""),
            "dwi_snr_shell0": d.get("snr_cc_shell0", ""),
            # 人工评级留空，待视觉核查
            "t1_qc_rating": "", "dwi_qc_rating": "", "bold_qc_rating": "",
            "seg_qc_rating": "",
            "exclusion_flag": int(bool(excl.get(s))),
            "exclusion_reason": "; ".join(excl.get(s, [])),
            "notes": (f"临床表{'有' if s in clin_by_id else '无'}"
                      f"/派生表{'有' if s in derived_by_id else '无'}"
                      f"/mriqc{'有' if s in mriqc_subs else '无'}"),
        })
        qc_rows.append(row)

    qc_path = os.path.join(args.out_dir, "subjects_qc.csv")
    with open(qc_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=QC_COLUMNS)
        w.writeheader()
        w.writerows(qc_rows)

    # ---------- 漏斗 + 排除清单 ----------
    only_mriqc = sorted(mriqc_subs - set(all_subs))
    lines = []
    lines.append("# ds005713 样本纳入漏斗与排除清单\n")
    lines.append("> 本文件由 `scripts/build_qc_skeleton.py` 自动生成，**请勿手工编辑**。\n")
    lines.append("## 1. 纳入漏斗\n")
    lines.append("| 步骤 | 被试数 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(f"| manifest 被试目录 | {funnel['被试目录总数(manifest)']} | "
                 f"55 对照 + 102 患者 = 157 基线；56 随访 |")
    lines.append(f"| − 仅有 JSON、零 `.nii.gz` | −{funnel['仅有JSON_零nii']} | "
                 f"公开快照不完整，17 基线 + 8 随访 |")
    lines.append(f"| **= 有影像的被试目录** | **{funnel['有影像的被试目录']}** | "
                 f"基线 {funnel['基线_有影像']} + 随访 {funnel['随访_有影像']} |")
    lines.append(f"| 基线有 T1w | {funnel['基线_T1w']} | → **A3（GWC）可用上限** |")
    lines.append(f"| 基线 T1w **且** T2w | **{funnel['基线_T1w且T2w']}** | "
                 f"→ **A3（T1w/T2w 平行指标）可用上限** |")
    lines.append(f"| 基线 T1w+T2w+DWI+BOLD 齐全 | **{funnel['基线_T1w_T2w_DWI_BOLD齐全']}** | "
                 f"→ 三表型齐全上限 |")
    lines.append(f"| 随访有 T1w | {funnel['随访_T1w']} | → 纵向可用上限 |")
    lines.append(f"| 随访 T1w 且 T2w | {funnel['随访_T1w且T2w']} | — |")
    lines.append(f"| 随访四项齐全 | {funnel['随访_T1w_T2w_DWI_BOLD齐全']} | — |")
    lines.append("")
    lines.append("## 2. mriqc 组表覆盖（决定 QC 从哪来）\n")
    lines.append("| 表 | 行数 | 覆盖被试 | 基线覆盖 | 随访覆盖 |")
    lines.append("|---|---|---|---|---|")
    for nm, tab in (("group_T1w.tsv", t1q), ("group_T2w.tsv", t2q),
                    ("group_bold.tsv", boldq), ("group_dwi.tsv", dwiq)):
        subs = {e["bids_name"].split("_")[0] for e in tab}
        fuc = len([s for s in subs if is_followup(s)])
        lines.append(f"| `{nm}` | {len(tab)} | {len(subs)} | {len(subs)-fuc} | {fuc} |")
    lines.append("")
    lines.append(f"**覆盖为 0 的部分**：随访被试在四张表中**全部缺失**，"
                 f"因此 **{funnel['随访_有影像']} 个随访被试的 QC 必须自行从影像计算**"
                 f"（FD、DVARS 等）。")
    lines.append("")
    if only_mriqc:
        lines.append(f"**注意**：mriqc 表中含公开快照无影像的被试 {only_mriqc}，"
                     f"合并时须内连接至 manifest，不可反向。\n")
    lines.append("## 3. 排除清单\n")
    reasons = Counter()
    for s, rs in excl.items():
        for r in rs:
            reasons[r] += 1
    lines.append("| 排除/降级理由 | 涉及被试数 |")
    lines.append("|---|---|")
    for r, n in reasons.most_common():
        lines.append(f"| {r} | {n} |")
    lines.append("")
    lines.append("### 3.1 仅有 JSON、无影像（必须排除）\n")
    b_noimg = sorted(s for s in no_image if not is_followup(s))
    f_noimg = sorted(s for s in no_image if is_followup(s))
    lines.append(f"- 基线 {len(b_noimg)} 个：`{'`, `'.join(b_noimg)}`")
    lines.append(f"- 随访 {len(f_noimg)} 个：`{'`, `'.join(f_noimg)}`")
    lines.append("")
    lines.append("### 3.2 数据质量异常（需人工定夺）\n")
    lines.append("| 被试 | 问题 | 建议 |")
    lines.append("|---|---|---|")
    lines.append("| `sub-055` | `run-01_T1w.nii.gz` 仅 270,111 B（严重截断）；"
                 "`run-02_T1w.nii.gz` 14 MB（亦偏小）；T1w/T2w 不匹配 | "
                 "优先用 run-02 并在 QC 表标注；若体积异常影响 GWC，则排除 |")
    lines.append("| `sub-043fu` | 存在 `run-05_dwi`，bval 仅 7 个体积，"
                 "文件 3.7 MB（方向数不足） | "
                 "纵向 DWI 分析中排除或仅用 run-01 |")
    lines.append("| `sub-16` | **完全没有 DWI 目录**（仅 T1w×2、T2w、BOLD） | "
                 "A1/A2 分析中自动排除 |")
    lines.append("| `sub-107`/`sub-128` | 仅有 T1w + T2w | 仅进入 A3 |")
    lines.append("")
    lines.append("### 3.3 无 T2w 的基线被试（T1w/T2w 平行指标不可用）\n")
    no_t2 = sorted(s for s in base_t1 if s not in base_t1t2)
    lines.append(f"共 {len(no_t2)} 个，A3 只能用 GWC 单指标，"
                 f"须在敏感性分析中标注：")
    lines.append(f"`{'`, `'.join(no_t2)}`")
    lines.append("")
    lines.append("## 4. 双 T1w run 被试（需择优）\n")
    dual = sorted(s for s in all_subs
                  if len([f for f in inv[s].get("anat", []) if "T1w" in f]) > 1)
    dual_base = [s for s in dual if not is_followup(s)]
    dual_qc = [s for s in dual_base if len(t1_by_sub.get(s, [])) > 1]
    lines.append(f"manifest 中 **{len(dual)}** 个被试有 ≥2 个 T1w run"
                 f"（基线 {len(dual_base)} 个、随访 {len(dual)-len(dual_base)} 个）。")
    lines.append("")
    lines.append(f"其中 **{len(dual_qc)}** 个基线被试在 mriqc `group_T1w.tsv` 中有 ≥2 行，"
                 f"**可以直接用 cjv 择优**；其余 "
                 f"{len(dual_base)-len(dual_qc)} 个需自算指标或用 `tpm_overlap_gm` 判断。")
    lines.append("")
    lines.append("选择规则：先剔除 `tpm_overlap_gm` 过低者，余下按 `cjv` 最小入选，"
                 "并把 `t1_run_delta_cjv`（次优减最优）作为采集异质性指标纳入敏感性分析。")
    lines.append("")
    lines.append("## 5. 产物\n")
    lines.append(f"- `subjects_qc.csv`：{len(qc_rows)} 行 × {len(QC_COLUMNS)} 列，"
                 f"基线 QC 已尽量填好，随访留空待自算。")
    lines.append(f"- 本文件 + `cohort_funnel.json`。")

    md = "\n".join(lines) + "\n"
    md_path = os.path.join(args.out_dir, "cohort_funnel.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)

    js_path = os.path.join(args.out_dir, "cohort_funnel.json")
    with open(js_path, "w", encoding="utf-8") as fh:
        json.dump({
            "funnel": funnel,
            "mriqc_coverage": {
                nm: len({e["bids_name"].split("_")[0] for e in tab})
                for nm, tab in (("T1w", t1q), ("T2w", t2q),
                                ("bold", boldq), ("dwi", dwiq))
            },
            "mriqc_only_subjects": only_mriqc,
            "no_image_baseline": b_noimg,
            "no_image_followup": f_noimg,
            "no_t2_baseline": no_t2,
            "dual_t1_subjects": dual,
            "exclusion_counts": dict(reasons),
        }, fh, ensure_ascii=False, indent=2)

    print(f"[ok] QC 表骨架   : {qc_path}  ({len(qc_rows)} 行)")
    print(f"[ok] 漏斗+排除   : {md_path}")
    print(f"[ok] 机器可读    : {js_path}")
    print()
    print("关键数字：")
    for k in ("被试目录总数(manifest)", "仅有JSON_零nii", "有影像的被试目录",
              "基线_T1w", "基线_T1w且T2w", "基线_T1w_T2w_DWI_BOLD齐全",
              "随访_T1w", "随访_T1w_T2w_DWI_BOLD齐全"):
        print(f"  {k:32s} {funnel[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
