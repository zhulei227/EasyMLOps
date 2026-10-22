# -*- coding: utf-8 -*-
"""
TS 全量测试门禁: 串联全部套件, 任一失败则进程退出码为 1。
运行方式: python tests/test_ts_all.py
"""
import os
import subprocess
import sys
import time

SUITES = [
    # (套件, 是否需要 torch)
    ("test_ts_ops.py", False),          # P0 基础特征/可逆变换
    ("test_ts_ops_p1.py", False),       # P1 重采样/扩展特征/平稳性
    ("test_ts_ops_p2.py", False),       # P2 变化量/幂变换/STL/熵
    ("test_ts_ops_p3.py", False),       # P3 统计模型(ARIMA 家族)
    ("test_ts_ops_p4.py", True),        # P4 深度模型修复验收
    ("test_ts_ops_p5.py", True),        # P5 GroupForecast 分组预测包装器
    ("test_ts_statistical.py", False),  # 统计模型(April 升级版)
    ("test_ts_deep_learning.py", True),
    ("test_ts_transformer.py", True),
    ("test_ts_state_space.py", True),
    ("test_ts_generative.py", True),
]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    has_torch = torch_available()
    results = []
    print("=" * 60)
    print("TS 全量测试门禁 (torch={})".format("available" if has_torch else "NOT available"))
    print("=" * 60)

    for suite, need_torch in SUITES:
        if need_torch and not has_torch:
            print("\n>>> [SKIP] {} (需要 PyTorch)".format(suite))
            results.append((suite, "SKIP"))
            continue
        print("\n>>> [RUN ] {}".format(suite))
        start = time.time()
        proc = subprocess.run([sys.executable, os.path.join("tests", suite)],
                              cwd=root)
        elapsed = time.time() - start
        status = "PASS" if proc.returncode == 0 else "FAIL"
        results.append((suite, status))
        print("<<< [{}] {} ({:.1f}s)".format(status, suite, elapsed))

    print("\n" + "=" * 60)
    print("汇总:")
    failed = 0
    for suite, status in results:
        print("  [{:4}] {}".format(status, suite))
        if status == "FAIL":
            failed += 1
    skipped = sum(1 for _, s in results if s == "SKIP")
    print("共 {} 套件: PASS {}, FAIL {}, SKIP {}".format(
        len(results), len(results) - failed - skipped, failed, skipped))
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
