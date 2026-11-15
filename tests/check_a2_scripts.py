# -*- coding: utf-8 -*-
"""A2 容器脚本的 `bash -n` 语法校验（与实际 A1 的检查同构）。

A2 只有一个容器阶段（FSL：mcflirt + flirt + 重采样），但其中包含
多行续行（反斜杠）与 `$?` 检查，是最容易因 CRLF 或引号出错的地方。
本脚本把它渲染出来只做语法解析，零算力代价。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import run_a2_pipeline as A2                              # noqa: E402

TMP_WORK = ROOT / 'build' / 'a2_syntax_check'
STAGE = ROOT / 'build' / 'a2_syntax_check_stage'


def main() -> int:
    for d in (TMP_WORK, STAGE):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    captured = {}

    def fake_run_script(work, stage_root, name, body, timeout_h):
        scripts = stage_root / 'scripts'
        scripts.mkdir(parents=True, exist_ok=True)
        sp = scripts / f'{name}.sh'
        with open(sp, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(body)
        captured[name] = sp
        return True, 'rendered'

    A2._run_script = fake_run_script

    sub = 'sub-CHK'
    A2.stage_fsl(sub, TMP_WORK, STAGE, 3.0, 1.0)

    bad = 0
    print('=' * 62)
    for name, sp in captured.items():
        raw = sp.read_bytes()
        has_cr = b'\r' in raw
        r = subprocess.run(
            ['docker', 'run', '--rm', '-u', '0:0', '-v', f'{STAGE}:/data:ro',
             '--entrypoint', 'bash', A2.IMAGE, '-n', f'/data/scripts/{name}.sh'],
            capture_output=True, text=True)
        ok = (r.returncode == 0) and not has_cr
        detail = 'ok' if ok else (
            ('CRLF! ' if has_cr else '') +
            ((r.stderr or r.stdout or '').strip().splitlines() or ['?'])[-1][:200])
        print(f'  {"PASS" if ok else "FAIL"}  {name:24s} {detail}')
        if not ok:
            bad += 1
    print('=' * 62)
    print(f'  {len(captured)-bad}/{len(captured)} 个阶段脚本语法正确')

    shutil.rmtree(TMP_WORK, ignore_errors=True)
    shutil.rmtree(STAGE, ignore_errors=True)
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
