# -*- coding: utf-8 -*-
"""对 A1 所有阶段的容器脚本做 `bash -n` 语法校验（不实际执行）。

为什么需要
----------
A1 的每个阶段都是"生成 bash 脚本 → 丢进容器执行"。若脚本有语法错误，
要到真正跑那一步才会暴露，而像 `eddy` 这种阶段一跑就是几十分钟，
失败代价高。`bash -n` 只解析不执行，能在零算力代价下提前拦截
引号/换行/续行错误（本项目就在 CRLF 上栽过一次）。

做法
----
把 `run_a1_pipeline._run_script` 换成"只做 bash -n"的桩，然后依次调用
各阶段函数。阶段函数内部有"产物已存在则跳过"的早退，因此这里先把
工作目录指向一个空的临时目录，确保脚本一定会被渲染出来。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import run_a1_pipeline as A1                              # noqa: E402

TMP_WORK = ROOT / 'build' / 'a1_syntax_check'
STAGE = ROOT / 'build' / 'a1_syntax_check_stage'

CHECKED: list[tuple[str, bool, str]] = []


def fake_run_script(work, stage_root, name, body, timeout_h):
    """替换真正的容器执行：只把脚本写盘（LF）并跑 bash -n。"""
    scripts = stage_root / 'scripts'
    scripts.mkdir(parents=True, exist_ok=True)
    sp = scripts / f'{name}.sh'
    with open(sp, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(body)
    # 关键：必须确认文件里没有 CR
    raw = sp.read_bytes()
    has_cr = b'\r' in raw
    r = subprocess.run(['docker', 'run', '--rm', '-u', '0:0',
                        '-v', f'{stage_root}:/data:ro',
                        '--entrypoint', 'bash', A1.IMAGE,
                        '-n', f'/data/scripts/{name}.sh'],
                       capture_output=True, text=True)
    ok = (r.returncode == 0) and not has_cr
    detail = 'ok' if ok else (
        ('CRLF! ' if has_cr else '') +
        ((r.stderr or r.stdout or '').strip().splitlines() or ['?'])[-1][:200])
    CHECKED.append((name, ok, detail))
    return ok, detail


def main() -> int:
    for d in (TMP_WORK, STAGE):
        if d.exists():
            import shutil
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    A1._run_script = fake_run_script

    dwi = {
        'main_rel': 'in/x_main_dwi.nii.gz',
        'bval_rel': 'in/x_main_dwi.bval',
        'bvec_rel': 'in/x_main_dwi.bvec',
        'pe': 'j',
        'readout': 0.0417004,
        'rev': {'nii_rel': 'in/x_rev_dwi.nii.gz',
                'bvec_rel': 'in/x_rev_dwi.bvec',
                'bval_rel': 'in/x_rev_dwi.bval',
                'run': 'run-02'},
    }
    sub = 'sub-CHK'
    args = [
        ('prep', lambda: A1.stage_prep(sub, dwi, TMP_WORK, STAGE, 4, 1)),
        ('preproc', lambda: A1.stage_preproc(sub, dwi, TMP_WORK, STAGE, 4, 1)),
        ('fod', lambda: A1.stage_fod(sub, TMP_WORK, STAGE, 4, 1)),
        ('5tt', lambda: A1.stage_5tt(sub, TMP_WORK, STAGE,
                                    'in/x_aseg.mgz', 'in/x_orig.mgz', 1)),
        ('tract', lambda: A1.stage_tract(sub, TMP_WORK, STAGE, 4, 10_000_000, 1)),
        ('connectome', lambda: A1.stage_connectome(sub, TMP_WORK, STAGE,
                                                  'in/x_atlas.nii.gz', 4, 1)),
    ]
    for tag, fn in args:
        fn()

    print('=' * 62)
    bad = 0
    for name, ok, detail in CHECKED:
        print(f'  {"PASS" if ok else "FAIL"}  {name:24s} {detail}')
        if not ok:
            bad += 1
    print('=' * 62)
    print(f'  {len(CHECKED) - bad}/{len(CHECKED)} 个阶段脚本语法正确')
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
