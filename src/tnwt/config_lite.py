"""零依赖配置解析（YAML 安全子集）与访问工具。

存在理由：分析环境不含 PyYAML，但 `config/analysis.yaml` 是唯一参数入口，
必须能被读取且**不可执行任意内容**。

支持的子集：
- 缩进嵌套映射
- 键值：``key: value``
- 行内列表：``[a, b]``；行内映射：``{a: 1, b: 2}``
- 块列表：``- item``（标量、行内映射、或 ``- key: value`` 起头的映射）
- 标量：int / float / bool / null / 引号或裸字符串；整行与行尾 ``#`` 注释
- 文档分隔符 ``---``、注释行、空行

明确不支持（遇到即报错，避免静默误读）：
锚点/别名（``&``/``*``）、标签（``!``）、多行标量（``|``/``>``）、
复杂键（``?``）、块内重复键、tab 缩进。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """配置解析错误。"""


_INLINE_MAP = re.compile(r'^\{(.*)\}$')
_INLINE_LIST = re.compile(r'^\[(.*)\]$')


def _strip_comment(line: str) -> str:
    out, quote = [], None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
            out.append(ch)
            continue
        if ch in '"\'':
            quote = ch
            out.append(ch)
            continue
        if ch == '#' and (i == 0 or line[i - 1] in ' \t'):
            break
        out.append(ch)
    return ''.join(out).rstrip()


def _split_top(text: str, sep: str = ',') -> list[str]:
    parts, depth, quote, buf = [], 0, None, []
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in '"\'':
            quote = ch
            buf.append(ch)
            continue
        if ch in '[{':
            depth += 1
        elif ch in ']}':
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(''.join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    tail = ''.join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _key_value(text: str) -> tuple[str, str]:
    depth, quote = 0, None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in '"\'':
            quote = ch
        elif ch in '[{':
            depth += 1
        elif ch in ']}':
            depth -= 1
        elif ch == ':' and depth == 0:
            return text[:i].strip(), text[i + 1:].strip()
    raise ConfigError(f'不是合法的 key: value —— {text!r}')


def parse_scalar(text: str) -> Any:
    t = text.strip()
    if t == '':
        return None
    if t[0] in '"\'':  # 引号字符串优先，保证 "1.8.0" 不被当作数字
        if len(t) < 2 or t[-1] != t[0]:
            raise ConfigError(f'引号未闭合：{t!r}')
        return t[1:-1]
    low = t.lower()
    if low in ('null', '~'):
        return None
    if low == 'true':
        return True
    if low == 'false':
        return False
    m = _INLINE_MAP.match(t)
    if m:
        body = m.group(1).strip()
        if not body:
            return {}
        result: dict[str, Any] = {}
        for item in _split_top(body):
            k, v = _key_value(item)
            k = k.strip('"\'')
            if k in result:
                raise ConfigError(f'行内映射存在重复键：{k!r}')
            result[k] = parse_scalar(v)
        return result
    m = _INLINE_LIST.match(t)
    if m:
        body = m.group(1).strip()
        return [] if not body else [parse_scalar(x) for x in _split_top(body)]
    if t.startswith(('&', '*', '!')):
        raise ConfigError(f'不支持锚点/别名/标签：{t!r}')
    if re.fullmatch(r'[+-]?\d+', t):
        return int(t)
    if re.fullmatch(r'[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?', t):
        return float(t)
    return t


class _Frame:
    __slots__ = ('indent', 'container')

    def __init__(self, indent: int, container: Any):
        self.indent = indent
        self.container = container


def parse(text: str) -> Any:
    """把 YAML 子集解析为嵌套 dict / list。"""
    lines: list[tuple[int, str, int]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        if '\t' in raw[: len(raw) - len(raw.lstrip())]:
            raise ConfigError(f'第 {lineno} 行使用了 tab 缩进，YAML 不允许')
        content = _strip_comment(raw)
        if not content.strip() or content.strip().startswith('---'):
            continue
        lines.append((len(content) - len(content.lstrip(' ')), content, lineno))

    if not lines:
        return {}
    # 顶层必须是映射或列表
    def build(index: int, indent: int) -> tuple[Any, int]:
        first_indent, first_text, lineno = lines[index]
        if first_indent != indent:
            raise ConfigError(f'第 {lineno} 行缩进异常：期望 {indent}，实际 {first_indent}')
        is_list = first_text.lstrip().startswith('- ')
        container: Any = [] if is_list else {}
        while index < len(lines):
            cur_indent, cur_text, cur_lineno = lines[index]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise ConfigError(f'第 {cur_lineno} 行缩进跳跃：{cur_indent} > {indent}')
            body = cur_text.strip()
            if is_list:
                if not body.startswith('- '):
                    break
                item = body[2:].strip()
                if not item:
                    nxt = lines[index + 1] if index + 1 < len(lines) else None
                    if nxt and nxt[0] > indent:
                        sub, index = build(index + 1, nxt[0])
                        container.append(sub)
                        continue
                    container.append(None)
                    index += 1
                    continue
                try:
                    k, v = _key_value(item)
                except ConfigError:
                    container.append(parse_scalar(item))
                    index += 1
                    continue
                node: dict[str, Any] = {}
                k = k.strip('"\'')
                node[k] = parse_scalar(v) if v else None
                index += 1
                if index < len(lines) and lines[index][0] > indent:
                    sub, index = build(index, lines[index][0])
                    if not isinstance(sub, dict):
                        raise ConfigError(f'第 {cur_lineno} 行的列表项映射无法合并子层级')
                    if k in node and node[k] is None:
                        node[k] = sub
                    else:
                        node.update(sub)
                container.append(node)
                continue
            k, v = _key_value(body)
            k = k.strip('"\'')
            if k in container:
                raise ConfigError(f'第 {cur_lineno} 行重复键：{k!r}')
            index += 1
            if v:
                container[k] = parse_scalar(v)
            elif index < len(lines) and lines[index][0] > indent:
                sub, index = build(index, lines[index][0])
                container[k] = sub
            else:
                container[k] = None
        return container, index

    value, end = build(0, lines[0][0])
    if end != len(lines):
        raise ConfigError(f'第 {lines[end][2]} 行未能解析，存在无法归属的层级')
    return value


def load(path: str | Path) -> dict[str, Any]:
    data = parse(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ConfigError(f'顶层必须是映射：{path}')
    return data


def dig(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """按 ``a.b.c`` 取值，缺失时返回 default。"""
    node: Any = cfg
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
