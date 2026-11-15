#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""探测公开基因集与细胞类型参考资源是否可获取。

只做元数据探测（HTTP 状态与体积），不下载大文件。
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.request

PROBES = [
    ('GO basic (go-basic.obo, ~30MB)',
     'https://purl.obolibrary.org/obo/go/go-basic.obo'),
    ('GO human annotations via AmiGO (gaf, 大)',
     'https://current.geneontology.org/annotations/goa_human.gaf.gz'),
    ('GO human annotations (gpad)',
     'https://current.geneontology.org/annotations/goa_human.gpad.gz'),
    ('Reactome NCBI2Reactome (~1MB)',
     'https://reactome.org/download/current/NCBI2Reactome_All_Levels.txt'),
    ('Reactome pathway names',
     'https://reactome.org/download/current/ReactomePathways.txt'),
    ('MSigDB hallmark (需登录，预期失败)',
     'https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2023.2.Hs/h.all.v2023.2.Hs.symbols.gmt'),
    ('Brain RNA-Seq 论文补充表（Zhang 2016 / 或 Karolinska）',
     'https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03705-x/MediaObjects/41586_2021_3705_MOESM4_ESM.xlsx'),
    ('Allen Brain Cell Atlas 文档入口',
     'https://brain-map.org/abc'),
]


def probe(name: str, url: str) -> dict:
    req = urllib.request.Request(url, headers={'User-Agent': 'tnwt-probe/1.0'}, method='GET')
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            status = r.status
            clen = r.headers.get('Content-Length')
            # 只读前 512 字节确认可读
            head = r.read(512)
            return {'name': name, 'url': url, 'status': status,
                    'content_length': int(clen) if clen else None,
                    'readable': len(head) > 0}
    except urllib.error.HTTPError as e:
        return {'name': name, 'url': url, 'status': e.code, 'error': f'HTTP {e.code}'}
    except Exception as e:  # noqa: BLE001
        return {'name': name, 'url': url, 'status': None, 'error': f'{type(e).__name__}: {e}'}


def main() -> int:
    ok = 0
    for name, url in PROBES:
        r = probe(name, url)
        if r.get('status') == 200:
            ok += 1
            size = r.get('content_length')
            size_s = f'{size/1024/1024:.1f} MB' if size else '未知大小'
            print(f'[OK]   {name}\n       {url}\n       status=200, {size_s}')
        else:
            print(f'[FAIL] {name}\n       {url}\n       {r.get("error") or r.get("status")}')
    print(f'\n可用资源 {ok}/{len(PROBES)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
