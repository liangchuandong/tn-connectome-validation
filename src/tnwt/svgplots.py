"""出版级矢量图生成（零依赖，输出 SVG）。

存在理由：本环境无 matplotlib，而 SVG 是纯文本格式，可直接生成且完全可复现。
所有图件为**矢量**，可直接投稿或后续用 Illustrator/Inkscape 微调。

设计约定
--------
- 单位用 pt（1pt = 1/72 inch）；默认单栏宽 90 mm ≈ 255pt、双栏 183 mm ≈ 519pt。
- 字体使用通用族（Helvetica/Arial），中文由 ``CJK_FALLBACK`` 兜底。
- 颜色使用色盲友好调色板（Okabe-Ito）。
- 每个绘图函数返回 SVG 字符串；:func:`write_svg` 负责落盘并记录元数据。
- **不做统计计算**：所有数据由调用方提供，绘图函数只负责呈现。

坐标边距约定
------------
绘图区左边界统一取 ``AXIS_LEFT``（36pt）或 ``AXIS_LEFT_FOREST``（76pt，容纳长行标签），
y 轴标签为旋转 -90° 文字、基线在 ``x - YLABEL_PAD``。这样可保证较长轴标签
（含中文）不被左边界裁剪。修改边距前请先跑 ``verify_figures.py``。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PALETTE = {
    'black': '#000000',
    'orange': '#E69F00',
    'sky': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'purple': '#CC79A7',
    'grey': '#999999',
    'lightgrey': '#DDDDDD',
    'darkgrey': '#666666',
}
FONT = 'Helvetica, Arial, sans-serif'
CJK_FALLBACK = "'Microsoft YaHei', 'SimHei', sans-serif"

SINGLE_COL = 255.0
DOUBLE_COL = 519.0

AXIS_LEFT = 36.0
AXIS_LEFT_FOREST = 76.0
YLABEL_PAD = 20.0


def _esc(text: str) -> str:
    return (str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


@dataclass
class SVG:
    """极简 SVG 画布。"""
    width: float
    height: float
    background: str = '#FFFFFF'
    elements: list[str] = field(default_factory=list)
    title: str = ''

    def add(self, elem: str) -> None:
        self.elements.append(elem)

    def rect(self, x, y, w, h, fill='none', stroke='none', sw=0.5, opacity=1.0,
             rx=0.0) -> None:
        self.add(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                 f'rx="{rx:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.2f}" '
                 f'opacity="{opacity:.3f}"/>')

    def line(self, x1, y1, x2, y2, stroke=PALETTE['black'], sw=0.7,
             dash: str | None = None, opacity=1.0) -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ''
        self.add(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                 f'stroke="{stroke}" stroke-width="{sw:.2f}" opacity="{opacity:.3f}"{d}/>')

    def path(self, d: str, stroke=PALETTE['black'], fill='none', sw=0.7,
             opacity=1.0) -> None:
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{sw:.2f}" opacity="{opacity:.3f}"/>')

    def circle(self, cx, cy, r, fill=PALETTE['blue'], stroke='none', sw=0.5,
               opacity=1.0) -> None:
        self.add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw:.2f}" opacity="{opacity:.3f}"/>')

    def text(self, x, y, s, size=6.5, fill=PALETTE['black'], anchor='start',
             weight='normal', style='normal', rotate: float | None = None) -> None:
        if s is None or str(s) == '':
            return
        tr = f' transform="rotate({rotate:.1f} {x:.2f} {y:.2f})"' if rotate else ''
        self.add(f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" '
                 f'font-size="{size:.1f}" fill="{fill}" text-anchor="{anchor}" '
                 f'font-weight="{weight}" font-style="{style}"{tr}>{_esc(s)}</text>')

    def panel_label(self, x, y, label, size=8.5) -> None:
        self.text(x, y, label, size=size, weight='bold')

    def axis_frame(self, x, y, w, h, xlabel='', ylabel='', title='',
                   size=6.5, xlabel_pad=13, ylabel_pad=YLABEL_PAD) -> None:
        """画坐标框并放置轴标签（y 轴标签旋转 -90°）。"""
        self.rect(x, y, w, h, fill='none', stroke=PALETTE['black'], sw=0.7)
        if xlabel:
            self.text(x + w / 2, y + h + xlabel_pad, xlabel, size=size, anchor='middle')
        if ylabel:
            self.text(x - ylabel_pad, y + h / 2, ylabel, size=size, anchor='middle',
                      rotate=-90)
        if title:
            self.text(x + w / 2, y - 4, title, size=size + 0.5, anchor='middle')

    def render(self) -> str:
        body = '\n  '.join(self.elements)
        t = f'<title>{_esc(self.title)}</title>\n  ' if self.title else ''
        return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{self.width:.0f}pt" height="{self.height:.0f}pt" '
                f'viewBox="0 0 {self.width:.0f} {self.height:.0f}">\n  {t}'
                f'<rect width="100%" height="100%" fill="{self.background}"/>\n  '
                f'{body}\n</svg>\n')


def _scale(vals, lo, hi, a, b):
    v = np.asarray(vals, dtype=float)
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return np.full_like(v, (a + b) / 2.0)
    return a + (np.clip(v, lo, hi) - lo) / (hi - lo) * (b - a)


def _nice_ticks(lo: float, hi: float, n: int = 5) -> np.ndarray:
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        return np.array([lo])
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / max(n, 1)))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    start = math.floor(lo / step) * step
    ticks = np.arange(start, hi + step * 0.5, step)
    return ticks[(ticks >= lo - 1e-9) & (ticks <= hi + 1e-9)]


# ==========================================================================
# 图类型
# ==========================================================================
def bar_chart(width: float, height: float, values, labels, *, ylabel='', title='',
              colors=None, errors=None, highlight=None, note='',
              ylim: tuple | None = None, panel_label='') -> str:
    """柱状图。``labels`` 中的空串会被跳过（避免输出空 <text> 元素）。"""
    svg = SVG(width, height, title=title)
    x0, y0, w, h = AXIS_LEFT, 18, width - AXIS_LEFT - 8, height - 44
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if ylim:
        lo, hi = ylim
    else:
        lo = 0.0
        hi = float(np.nanmax(vals) * 1.15) if finite.any() else 1.0
        if errors is not None and np.isfinite(errors).any():
            hi = float(np.nanmax(vals + np.asarray(errors))) * 1.15
    hi = hi if hi > lo else lo + 1
    svg.axis_frame(x0, y0, w, h, ylabel=ylabel)
    for t in _nice_ticks(lo, hi, 5):
        yy = y0 + h - _scale([t], lo, hi, 0, h)[0]
        svg.line(x0, yy, x0 + w, yy, stroke=PALETTE['lightgrey'], sw=0.5)
        svg.text(x0 - 2, yy + 2, f'{t:g}', size=5.5, anchor='end')
    n = max(len(vals), 1)
    bw = w / n * 0.62
    for i, v in enumerate(vals):
        cx = x0 + w * (i + 0.5) / n
        if not np.isfinite(v):
            svg.text(cx, y0 + h - 3, 'NA', size=5.5, anchor='middle',
                     fill=PALETTE['darkgrey'])
            continue
        col = (colors[i] if colors else
               (PALETTE['vermillion'] if (highlight is not None and highlight[i])
                else PALETTE['blue']))
        top = y0 + h - _scale([v], lo, hi, 0, h)[0]
        svg.rect(cx - bw / 2, top, bw, y0 + h - top, fill=col, stroke=PALETTE['black'],
                 sw=0.4)
        if errors is not None and np.isfinite(errors[i]):
            e = _scale([errors[i]], 0, hi - lo, 0, h)[0]
            svg.line(cx, top - e, cx, top + e, stroke=PALETTE['black'], sw=0.7)
            svg.line(cx - 2, top - e, cx + 2, top - e, stroke=PALETTE['black'], sw=0.7)
            svg.line(cx - 2, top + e, cx + 2, top + e, stroke=PALETTE['black'], sw=0.7)
        lab = labels[i] if i < len(labels) else ''
        if lab is not None and str(lab) != '':
            svg.text(cx, y0 + h + 9, str(lab), size=5.5, anchor='middle',
                     rotate=(-30 if len(str(lab)) > 6 and n > 4 else None))
    if panel_label:
        svg.panel_label(4, 10, panel_label)
    if note:
        svg.text(x0, height - 4, note, size=5.5, fill=PALETTE['darkgrey'])
    return svg.render()


def scatter_chart(width: float, height: float, x, y, *, xlabel='', ylabel='',
                  title='', fit=True, color=PALETTE['blue'], annotate=None,
                  panel_label='', point_size=1.8, equal_axes=False) -> str:
    """散点图（含可选最小二乘拟合线）。"""
    svg = SVG(width, height, title=title)
    x0, y0, w, h = AXIS_LEFT, 18, width - AXIS_LEFT - 8, height - 42
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    m = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[m], yv[m]
    if xv.size == 0:
        svg.axis_frame(x0, y0, w, h, xlabel=xlabel, ylabel=ylabel)
        svg.text(x0 + w / 2, y0 + h / 2, '无有效数据', size=6, anchor='middle')
        return svg.render()
    lo_x, hi_x = float(xv.min()), float(xv.max())
    lo_y, hi_y = float(yv.min()), float(yv.max())
    if equal_axes:
        lo, hi = min(lo_x, lo_y), max(hi_x, hi_y)
        lo_x, hi_x = lo_y, hi_y = lo, hi
    pad_x = (hi_x - lo_x) * 0.05 or 1.0
    pad_y = (hi_y - lo_y) * 0.05 or 1.0
    lo_x, hi_x = lo_x - pad_x, hi_x + pad_x
    lo_y, hi_y = lo_y - pad_y, hi_y + pad_y
    svg.axis_frame(x0, y0, w, h, xlabel=xlabel, ylabel=ylabel)
    for t in _nice_ticks(lo_x, hi_x, 5):
        xx = x0 + _scale([t], lo_x, hi_x, 0, w)[0]
        svg.line(xx, y0, xx, y0 + h, stroke=PALETTE['lightgrey'], sw=0.5)
        svg.text(xx, y0 + h + 9, f'{t:g}', size=5.5, anchor='middle')
    for t in _nice_ticks(lo_y, hi_y, 5):
        yy = y0 + h - _scale([t], lo_y, hi_y, 0, h)[0]
        svg.line(x0, yy, x0 + w, yy, stroke=PALETTE['lightgrey'], sw=0.5)
        svg.text(x0 - 2, yy + 2, f'{t:g}', size=5.5, anchor='end')
    px = x0 + _scale(xv, lo_x, hi_x, 0, w)
    py = y0 + h - _scale(yv, lo_y, hi_y, 0, h)
    for a, b in zip(px, py):
        svg.circle(a, b, point_size, fill=color, opacity=0.75)
    if fit and xv.size > 2 and xv.std() > 0:
        beta = np.polyfit(xv, yv, 1)
        xs = np.array([lo_x, hi_x])
        ys = np.polyval(beta, xs)
        svg.line(x0, y0 + h - _scale([ys[0]], lo_y, hi_y, 0, h)[0],
                 x0 + w, y0 + h - _scale([ys[1]], lo_y, hi_y, 0, h)[0],
                 stroke=PALETTE['vermillion'], sw=0.9)
        r = float(np.corrcoef(xv, yv)[0, 1])
        svg.text(x0 + w - 3, y0 + 9, f'r = {r:+.3f}  n = {xv.size}', size=6,
                 anchor='end')
    if annotate:
        svg.text(x0 + 3, y0 + 9, annotate, size=5.5, fill=PALETTE['darkgrey'])
    if panel_label:
        svg.panel_label(4, 10, panel_label)
    return svg.render()


def histogram_chart(width: float, height: float, values, *, bins=40, xlabel='',
                    ylabel='基因数', title='', threshold=None, threshold_label='',
                    panel_label='', color=PALETTE['sky'], annotate='') -> str:
    """直方图（如 PLS1 基因权重 Z 分布）。"""
    svg = SVG(width, height, title=title)
    x0, y0, w, h = AXIS_LEFT, 18, width - AXIS_LEFT - 8, height - 42
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        svg.axis_frame(x0, y0, w, h, xlabel=xlabel, ylabel=ylabel)
        svg.text(x0 + w / 2, y0 + h / 2, '无有效数据', size=6, anchor='middle')
        return svg.render()
    counts, edges = np.histogram(v, bins=bins)
    lo, hi = float(edges[0]), float(edges[-1])
    ymax = float(counts.max()) * 1.12 or 1.0
    svg.axis_frame(x0, y0, w, h, xlabel=xlabel, ylabel=ylabel)
    for t in _nice_ticks(0, ymax, 4):
        yy = y0 + h - _scale([t], 0, ymax, 0, h)[0]
        svg.line(x0, yy, x0 + w, yy, stroke=PALETTE['lightgrey'], sw=0.5)
        svg.text(x0 - 2, yy + 2, f'{t:g}', size=5.5, anchor='end')
    for i, c in enumerate(counts):
        a = x0 + _scale([edges[i]], lo, hi, 0, w)[0]
        b = x0 + _scale([edges[i + 1]], lo, hi, 0, w)[0]
        top = y0 + h - _scale([c], 0, ymax, 0, h)[0]
        svg.rect(a, top, max(b - a, 0.4), y0 + h - top, fill=color,
                 stroke=PALETTE['darkgrey'], sw=0.2)
    for t in _nice_ticks(lo, hi, 5):
        xx = x0 + _scale([t], lo, hi, 0, w)[0]
        svg.text(xx, y0 + h + 9, f'{t:g}', size=5.5, anchor='middle')
    if threshold is not None:
        for sgn in (1, -1):
            tv = sgn * threshold
            if lo <= tv <= hi:
                xx = x0 + _scale([tv], lo, hi, 0, w)[0]
                svg.line(xx, y0, xx, y0 + h, stroke=PALETTE['vermillion'], sw=0.8,
                         dash='2,1.5')
                if threshold_label:
                    svg.text(xx + 1.5, y0 + 8, threshold_label, size=5.5,
                             fill=PALETTE['vermillion'])
    if annotate:
        svg.text(x0 + 3, y0 + h - 3, annotate, size=5.5, fill=PALETTE['darkgrey'])
    if panel_label:
        svg.panel_label(4, 10, panel_label)
    return svg.render()


def forest_chart(width: float, height: float, labels, estimates, *, ci_low=None,
                 ci_high=None, xlabel='效应量', title='', panel_label='',
                 null_value=0.0, color=PALETTE['blue'], annotate=None) -> str:
    """森林图（如各组 AUC 或富集条目 -log10(p)）。"""
    svg = SVG(width, height, title=title)
    x0, y0, w, h = AXIS_LEFT_FOREST, 14, width - AXIS_LEFT_FOREST - 14, height - 30
    est = np.asarray(estimates, dtype=float)
    lows = np.asarray(ci_low, dtype=float) if ci_low is not None else est
    highs = np.asarray(ci_high, dtype=float) if ci_high is not None else est
    m = np.isfinite(est)
    if not m.any():
        svg.text(width / 2, height / 2, '无有效数据', size=6, anchor='middle')
        return svg.render()
    allv = np.concatenate([est[m], lows[np.isfinite(lows)], highs[np.isfinite(highs)],
                           [null_value]])
    lo, hi = float(allv.min()), float(allv.max())
    pad = (hi - lo) * 0.08 or 0.5
    lo, hi = lo - pad, hi + pad
    n = len(est)
    row_h = h / max(n, 1)
    for t in _nice_ticks(lo, hi, 5):
        xx = x0 + _scale([t], lo, hi, 0, w)[0]
        svg.line(xx, y0, xx, y0 + h, stroke=PALETTE['lightgrey'], sw=0.5)
        svg.text(xx, y0 + h + 9, f'{t:g}', size=5.5, anchor='middle')
    nv = x0 + _scale([null_value], lo, hi, 0, w)[0]
    svg.line(nv, y0, nv, y0 + h, stroke=PALETTE['darkgrey'], sw=0.8, dash='2,1.5')
    for i, lab in enumerate(labels):
        cy = y0 + row_h * (i + 0.5)
        svg.text(x0 - 4, cy + 2, str(lab), size=6, anchor='end')
        if np.isfinite(est[i]):
            if ci_low is not None and np.isfinite(lows[i]) and np.isfinite(highs[i]):
                a = x0 + _scale([lows[i]], lo, hi, 0, w)[0]
                b = x0 + _scale([highs[i]], lo, hi, 0, w)[0]
                svg.line(a, cy, b, cy, stroke=color, sw=0.8)
                svg.line(a, cy - 2, a, cy + 2, stroke=color, sw=0.8)
                svg.line(b, cy - 2, b, cy + 2, stroke=color, sw=0.8)
            cx = x0 + _scale([est[i]], lo, hi, 0, w)[0]
            svg.circle(cx, cy, 2.0, fill=color)
    svg.text(x0 + w / 2, y0 + h + 22, xlabel, size=6.5, anchor='middle')
    if annotate:
        svg.text(x0, y0 - 4, annotate, size=5.5, fill=PALETTE['darkgrey'])
    if panel_label:
        svg.panel_label(4, 10, panel_label)
    return svg.render()


def heatmap_chart(width: float, height: float, matrix, row_labels, col_labels, *,
                  title='', cbar_label='', diverging=True, panel_label='',
                  value_range: tuple | None = None) -> str:
    """热图（如 A1/A2/A3 共线性矩阵）。"""
    svg = SVG(width, height, title=title)
    M = np.asarray(matrix, dtype=float)
    nr, nc = M.shape
    left, top = 62, 20
    cbar_w = 8
    gw = width - left - 34 - cbar_w
    gh = height - top - 30
    if value_range:
        vlo, vhi = value_range
    else:
        vlo, vhi = float(np.nanmin(M)), float(np.nanmax(M))
    if diverging:
        lim = max(abs(vlo), abs(vhi)) or 1.0
        vlo, vhi = -lim, lim
    if vhi <= vlo:
        vhi = vlo + 1.0
    cw, ch = gw / max(nc, 1), gh / max(nr, 1)

    def _color(v):
        if not np.isfinite(v):
            return PALETTE['lightgrey']
        t = (v - vlo) / (vhi - vlo)
        if diverging:
            if t < 0.5:
                f = t / 0.5
                a, b = (0, 114, 178), (255, 255, 255)
            else:
                f = (t - 0.5) / 0.5
                a, b = (255, 255, 255), (213, 94, 0)
        else:
            f = t
            a, b = (255, 255, 255), (0, 114, 178)
        rgb = tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(3))
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

    for i in range(nr):
        for j in range(nc):
            svg.rect(left + j * cw, top + i * ch, cw, ch, fill=_color(M[i, j]),
                     stroke='#FFFFFF', sw=0.3)
    for j, cl in enumerate(col_labels):
        if nc <= 12 or j % max(1, nc // 10) == 0:
            svg.text(left + cw * (j + 0.5), top + gh + 8, str(cl), size=5.5,
                     anchor='middle', rotate=(-40 if nc > 8 else None))
    for i, rl in enumerate(row_labels):
        if nr <= 20 or i % max(1, nr // 15) == 0:
            svg.text(left - 3, top + ch * (i + 0.5) + 2, str(rl), size=5.5,
                     anchor='end')
    cbx = left + gw + 14
    steps = 40
    for s in range(steps):
        v = vlo + (vhi - vlo) * s / (steps - 1)
        svg.rect(cbx, top + gh * (1 - (s + 1) / steps), cbar_w, gh / steps,
                 fill=_color(v), stroke='none')
    svg.rect(cbx, top, cbar_w, gh, fill='none', stroke=PALETTE['black'], sw=0.5)
    svg.text(cbx + cbar_w + 2, top + 4, f'{vhi:.2f}', size=5.5)
    svg.text(cbx + cbar_w + 2, top + gh, f'{vlo:.2f}', size=5.5)
    if cbar_label:
        svg.text(cbx + cbar_w / 2, top - 5, cbar_label, size=5.5, anchor='middle')
    if panel_label:
        svg.panel_label(4, 10, panel_label)
    return svg.render()


def line_chart(width: float, height: float, series: dict, *, xlabel='', ylabel='',
               title='', panel_label='', xvalues=None, colors=None,
               marker='circle', annotate='') -> str:
    """折线图（如稳定性随阈值变化、LODO 覆盖曲线）。"""
    svg = SVG(width, height, title=title)
    x0, y0, w, h = AXIS_LEFT, 18, width - AXIS_LEFT - 8, height - 42
    finite_series = {k: np.asarray(v, dtype=float) for k, v in series.items()}
    if not finite_series:
        svg.axis_frame(x0, y0, w, h, xlabel=xlabel, ylabel=ylabel)
        return svg.render()
    n = max(len(v) for v in finite_series.values())
    xs = (np.asarray(xvalues, dtype=float) if xvalues is not None
          else np.arange(n, dtype=float))
    allv = np.concatenate([v[np.isfinite(v)] for v in finite_series.values()])
    lo_x, hi_x = float(np.nanmin(xs)), float(np.nanmax(xs))
    lo_y, hi_y = float(allv.min()), float(allv.max())
    if hi_y <= lo_y:
        hi_y = lo_y + 1.0
    if hi_x <= lo_x:
        hi_x = lo_x + 1.0
    pad = (hi_y - lo_y) * 0.08
    lo_y, hi_y = lo_y - pad, hi_y + pad
    svg.axis_frame(x0, y0, w, h, xlabel=xlabel, ylabel=ylabel)
    for t in _nice_ticks(lo_y, hi_y, 5):
        yy = y0 + h - _scale([t], lo_y, hi_y, 0, h)[0]
        svg.line(x0, yy, x0 + w, yy, stroke=PALETTE['lightgrey'], sw=0.5)
        svg.text(x0 - 2, yy + 2, f'{t:g}', size=5.5, anchor='end')
    for t in _nice_ticks(lo_x, hi_x, 5):
        xx = x0 + _scale([t], lo_x, hi_x, 0, w)[0]
        svg.text(xx, y0 + h + 9, f'{t:g}', size=5.5, anchor='middle')
    for k, v in finite_series.items():
        col = (colors or {}).get(k, PALETTE['blue'])
        px = x0 + _scale(xs[: len(v)], lo_x, hi_x, 0, w)
        py = y0 + h - _scale(v, lo_y, hi_y, 0, h)
        m = np.isfinite(v)
        if m.sum() > 1:
            d = 'M ' + ' L '.join(f'{a:.2f} {b:.2f}' for a, b in zip(px[m], py[m]))
            svg.path(d, stroke=col, sw=1.0)
        if marker == 'circle':
            for a, b in zip(px[m], py[m]):
                svg.circle(a, b, 1.5, fill=col)
    ly = y0 + 4
    for k in finite_series:
        col = (colors or {}).get(k, PALETTE['blue'])
        svg.line(x0 + w - 60, ly, x0 + w - 52, ly, stroke=col, sw=1.0)
        svg.circle(x0 + w - 56, ly, 1.5, fill=col)
        svg.text(x0 + w - 50, ly + 2, str(k), size=5.5)
        ly += 8
    if annotate:
        svg.text(x0, y0 - 4, annotate, size=5.5, fill=PALETTE['darkgrey'])
    if panel_label:
        svg.panel_label(4, 10, panel_label)
    return svg.render()


def flow_diagram(width: float, height: float, stages: list[dict], title='',
                 panel_label='') -> str:
    """受试者纳入流程图（CONSORT 式）。

    ``stages`` 每项：``{'label': str, 'n': int, 'note': str,
    'excluded': [(reason, n), ...]}``。
    """
    svg = SVG(width, height, title=title)
    bw, bh = width * 0.52, 26
    x = (width - bw) / 2
    y = 16
    for i, st in enumerate(stages):
        svg.rect(x, y, bw, bh, fill='#F7F9FC', stroke=PALETTE['black'], sw=0.7)
        svg.text(x + bw / 2, y + bh / 2 - 1, st['label'], size=6.5, anchor='middle',
                 weight='bold')
        svg.text(x + bw / 2, y + bh / 2 + 9, f"n = {st['n']}", size=6.5,
                 anchor='middle')
        if st.get('note'):
            svg.text(x + bw / 2, y + bh + 8, st['note'], size=5.5, anchor='middle',
                     fill=PALETTE['darkgrey'])
        if i < len(stages) - 1:
            svg.line(x + bw / 2, y + bh + (12 if st.get('note') else 0),
                     x + bw / 2, y + bh + 22, stroke=PALETTE['black'], sw=0.8)
        ex = st.get('excluded') or []
        if ex:
            ex_x = x + bw + 12
            svg.line(x + bw, y + bh / 2, ex_x, y + bh / 2, stroke=PALETTE['black'],
                     sw=0.6)
            for k, (reason, cnt) in enumerate(ex):
                yy = y + 8 + k * 10
                svg.text(ex_x + 2, yy, f'排除 {cnt}：{reason}', size=5.5)
        y += bh + 22 + (8 if st.get('note') else 0)
    if panel_label:
        svg.panel_label(4, 10, panel_label)
    return svg.render()


# ==========================================================================
# 落盘与组合
# ==========================================================================
def write_svg(path: str | Path, content: str, meta: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding='utf-8')
    if meta is not None:
        p.with_suffix('.json').write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=float),
            encoding='utf-8')
    return p


def composite(width: float, height: float, panels: list[tuple[float, float, str]],
              title='') -> str:
    """把多个已渲染的 SVG 片段拼成一张多面板图。

    ``panels`` 每项为 ``(x, y, svg_string)``；片段内容通过 ``<g transform>``
    定位（片段内的绝对坐标会被平移），并移除片段自身的白色背景以免遮挡。
    """
    svg = SVG(width, height, title=title)
    for x, y, frag in panels:
        if '<svg' not in frag:
            continue
        body = frag.split('<svg', 1)[1]
        inner = body.split('>', 1)[1].rsplit('</svg>', 1)[0]
        inner = inner.replace('<rect width="100%" height="100%" fill="#FFFFFF"/>', '')
        svg.add(f'<g transform="translate({x:.2f},{y:.2f})">{inner}</g>')
    return svg.render()


def figure_manifest(figures: dict[str, dict], path: str | Path) -> Path:
    """写出图件清单（供论文与补充材料引用）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({'figures': figures}, ensure_ascii=False, indent=2,
                            default=float), encoding='utf-8')
    return p
