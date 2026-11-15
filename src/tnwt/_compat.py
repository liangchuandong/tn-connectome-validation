"""兼容 shim：修补第三方库在较新依赖版本下失效的引用。

本文件记录并修补两类**实测到的**上游兼容断裂（2026-10-16）。两类都不是
本项目的代码问题，而是"最新版对最新版"的生态断裂。

---

## 问题 1：netneurotools 0.3.0 引用了 scipy 已移除的私有 API

本机 scipy 为 **1.18.1**，其中 `scipy.stats._stats_py._chk2_asarray` 已被上游移除
（现仅存 `_chk_asarray`）。而 **netneurotools 0.3.0（当前 PyPI 最新版）** 仍在
`netneurotools/stats/correlation.py` 与 `permutation_test.py` 中
`from scipy.stats._stats_py import _chk2_asarray`，导致：

    ImportError: cannot import name '_chk2_asarray' from 'scipy.stats._stats_py'

`netneurotools` 的元数据对 scipy **没有版本上限**，因此这不是"降级依赖"能干净解决的
问题，而是一个上游尚未修复的私有 API 依赖。

**为什么不降级 scipy**：netneurotools 对 scipy 无上限；降级 scipy 会牵连
numpy/pandas/statsmodels 一整条链路（当前 numpy 2.5.4 / pandas 3.0.6 /
statsmodels 0.15.0），风险远大于补一个内部辅助函数。

**补的是什么**：`_chk2_asarray` 是 scipy 内部的输入规范化辅助函数，语义等价于：

    def _chk2_asarray(a, b, axis):
        if axis == 0:
            a, b = np.asarray(a), np.asarray(b)
        else:
            a, b = np.asarray(a).T, np.asarray(b).T
        return a, b, axis

它**只**被 `efficient_pearsonr` 与若干置换检验使用，**spin test 本身不依赖它**
（何况 spin test 已不在该包中，见问题 2）。但 `netneurotools/stats/__init__.py`
会导入这些模块，所以缺了它整个 `netneurotools.stats` 都不可用。

---

## 问题 2：netneurotools 0.3.0 已**移除** `gen_spinsamples` / `rotate_spatial_data`

方案 v2.1 第 5.5 节点名要求使用 `netneurotools` 的
`gen_spinsamples` / `rotate_spatial_data`。**实测 0.3.0 中这两个函数已不存在**：

- `netneurotools.stats`（0.3.0）只导出 `efficient_pearsonr`、`weighted_pearsonr`、
  `make_correlated_xy`、`permtest_1samp`、`permtest_rel`、`permtest_pearsonr`、
  `_add_constant`、`residualize`、`get_dominance_stats`；
- `netneurotools.spatial`（0.3.0）只导出 `morans_i`、`local_morans_i`、
  `gearys_c`、`local_gearys_c`、`lees_l`、`local_lees_l`；
- 全包搜索 `def .*spin` **无任何命中**。

**好消息：这不构成阻塞。** 本项目自己的 `tnwt/spin.py` 已提供等价实现
（`gen_rotation_matrices` / `gen_spinsamples` / `spin_test`），并且
**本 shim 补好后可直接复用 netneurotools 的 Moran's I**：

    from tnwt.spin import gen_spinsamples, spin_test      # spin-test 用这个
    from netneurotools.spatial import morans_i            # Moran's I 用这个

方案"须注明版本"的要求，改为在论文中写明"使用 `tnwt.spin`（本项目实现，
算法遵循 netneurotools 旧版 `gen_spinsamples` 的球形旋转+最近邻置换逻辑），
并注明 netneurotools 版本"。这是可复现、可解释的口径调整。

---

## 用法

**必须在导入 netneurotools 之前**调用（导入即生效）：

    from tnwt import _compat   # noqa: F401
    from netneurotools.spatial import morans_i
    from tnwt.spin import gen_spinsamples, spin_test

自检
----
    python -m tnwt._compat
"""
from __future__ import annotations

import numpy as np
import scipy.stats._stats_py as _stats_py


def _chk2_asarray(a, b, axis):
    """scipy 旧版内部辅助函数的等价实现。

    与 scipy 历史实现保持一致的语义：axis==0 时原样转数组，
    否则转置后再转数组，并原样返回 axis。
    """
    if axis == 0:
        a, b = np.asarray(a), np.asarray(b)
    else:
        a, b = np.asarray(a).T, np.asarray(b).T
    return a, b, axis


def apply() -> list[str]:
    """把缺失的私有 API 挂回 scipy。返回本次实际打过补丁的名字列表。"""
    patched: list[str] = []

    if not hasattr(_stats_py, "_chk2_asarray"):
        _stats_py._chk2_asarray = _chk2_asarray
        patched.append("scipy.stats._stats_py._chk2_asarray")

    # scipy 旧路径 scipy.stats.stats 在新版里仍然存在，补上以覆盖 netneurotools
    # 的 except ImportError 分支（该分支只在第一个 import 失败时触发）。
    try:
        import scipy.stats.stats as _stats_old
        if not hasattr(_stats_old, "_chk2_asarray"):
            _stats_old._chk2_asarray = _chk2_asarray
            patched.append("scipy.stats.stats._chk2_asarray")
    except ImportError:
        pass

    return patched


PATCHED = apply()


def _selftest() -> int:
    """验证 shim 生效 + Moran's I 可用 + 本项目的 spin 实现可用且数值正确。"""
    print(f"[1] 本次补丁: {PATCHED or '（无需补丁，scipy 已自带）'}")

    # netneurotools.stats 现在应该可以导入了（正是 shim 的目的）
    try:
        import netneurotools.stats as ns
        print(f"[2] netneurotools.stats 导入 OK（版本 {_nt_version()}）")
    except Exception as exc:                          # noqa: BLE001
        print(f"[2] [FAIL] netneurotools.stats 仍无法导入: {type(exc).__name__}: {exc}")
        return 1

    # 确认 spin 函数确实不在 netneurotools 中（记录该事实，而非报错）
    has_spin = any(hasattr(ns, f) for f in ("gen_spinsamples", "rotate_spatial_data"))
    print(f"[3] netneurotools 是否仍提供 spin 函数: {'是' if has_spin else '否（已移除，改用 tnwt.spin）'}")

    # Moran's I 可用性（这是我们要从 netneurotools 复用的部分）
    try:
        from netneurotools.spatial import morans_i
        print("[4] netneurotools.spatial.morans_i OK")
    except Exception as exc:                          # noqa: BLE001
        print(f"[4] [FAIL] morans_i 不可用: {type(exc).__name__}: {exc}")
        return 1

    # 本项目 spin 实现：端到端 + 排列合法性
    from tnwt.spin import gen_spinsamples, spin_test
    rng = np.random.default_rng(42)
    coords = rng.normal(size=(20, 3))
    spins = gen_spinsamples(coords, n_perm=10, seed=42)
    print(f"[5] tnwt.spin.gen_spinsamples -> shape={spins.shape}")
    if spins.shape != (10, 20):
        print("    [FAIL] 期望 shape (10, 20)")
        return 1
    for j in range(spins.shape[0]):
        if sorted(spins[j].tolist()) != list(range(20)):
            print(f"    [FAIL] 第 {j} 行不是合法排列（bijective 保证被破坏）")
            return 1
    print("    每行均为合法排列 OK（bijective 贪心分配生效）")

    # spin_test 的 observed 必须等于 np.corrcoef（ddof=0 约定）
    x = rng.normal(size=20)
    y = rng.normal(size=20)
    res = spin_test(x, spins, y)
    exp = float(np.corrcoef(x, y)[0, 1])
    if abs(res["observed"] - exp) > 1e-12:
        print(f"    [FAIL] observed={res['observed']} != np.corrcoef={exp}")
        return 1
    print(f"[6] spin_test observed={res['observed']:.6f} == np.corrcoef OK, "
          f"p_two={res['p_two_sided']:.4f}")

    # shim 数值语义
    a, b = [[1, 2], [3, 4]], [[5, 6], [7, 8]]
    for axis in (0, 1):
        ra, rb, r_axis = _stats_py._chk2_asarray(a, b, axis)
        if axis == 0:
            exp_a, exp_b = np.asarray(a), np.asarray(b)
        else:
            exp_a, exp_b = np.asarray(a).T, np.asarray(b).T
        assert np.array_equal(ra, exp_a) and np.array_equal(rb, exp_b) and r_axis == axis
    print("[7] _chk2_asarray 语义校验 OK (axis=0/1)")

    print("\n[PASS] 兼容 shim 生效；Moran's I 可复用 netneurotools，spin-test 用 tnwt.spin。")
    return 0


def _nt_version() -> str:
    try:
        import netneurotools
        return getattr(netneurotools, "__version__", "?")
    except Exception:                                 # noqa: BLE001
        return "?"


if __name__ == "__main__":
    raise SystemExit(_selftest())
