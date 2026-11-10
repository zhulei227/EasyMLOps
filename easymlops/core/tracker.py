# -*- coding: utf-8 -*-
"""
训练监控 Tracker（v0.2.13 新增）

轻量适配层: 对接 swanlab(https://swanlab.cn) 监控模型训练过程, 并做本地 JSONL 落盘
以便离线分析。设计原则:

1. swanlab 为纯可选依赖: 未安装时 Tracker 自动降级为 no-op + 本地 JSONL 记录,
   不影响任何现有训练路径;
2. 通过 fit 的 **kwargs 透传: pipeline.fit(x, tracker=Tracker(...)) 一次传入,
   覆盖整条流水线所有训练点(训练循环内消费后 pop 掉, 不污染下游 pipe);
3. mode="offline" 使用 swanlab 离线模式(不联网, 可用 `swanlab watch` 看板回放),
   适合内网/CI 环境; mode="disabled" 完全禁用云端上报。

用法:
    from easymlops.core import Tracker

    tracker = Tracker(project="my-project", config={"lr": 0.01})
    # 单模型
    pipe.fit(x, tracker=tracker)
    # 或整条流水线
    pipeline.fit(x, tracker=tracker)
    tracker.finish()

    # 训练后分析本地记录
    Tracker.from_jsonl(tracker.history_path)  # 读取历史
"""

import json
import os
import time

__all__ = ["Tracker", "pop_tracker"]


def pop_tracker(kwargs: dict):
    """从 kwargs 中取出 tracker(不存在则返回 None), 供 udf_fit 内使用。

    注意: 直接修改传入的 dict, 因此应在 udf_fit 内最先调用。
    """
    return kwargs.pop("tracker", None)


class Tracker(object):
    """训练过程监控器, 适配 swanlab, 未安装时降级为本地 JSONL 记录。"""

    def __init__(self, project="easymlops", experiment_name=None, config=None,
                 mode="offline", log_dir=None, log_every=1):
        """
        Args:
            project: swanlab 项目名
            experiment_name: 本次运行名; None 时自动生成
            config: 超参数字典, 随 run 记录(如 {"lr": 0.01, "epochs": 50})
            mode: "offline"(默认, swanlab 离线模式)/"cloud"/"disabled";
                  swanlab 未安装时自动降级为仅本地 JSONL 记录
            log_dir: 本地 JSONL 记录目录; None 时为 ./tracker_logs
            log_every: 每 N 次上报一次 swanlab(本地 JSONL 仍逐条记录)
        """
        assert mode in ("cloud", "offline", "disabled"), \
            "mode must be one of cloud/offline/disabled, got '{}'".format(mode)
        self.project = project
        self.experiment_name = experiment_name or "{}-{}".format(
            project, time.strftime("%Y%m%d-%H%M%S"))
        self.config = dict(config) if config else {}
        self.mode = mode
        self.log_dir = log_dir or os.path.join(os.getcwd(), "tracker_logs")
        self.log_every = max(1, int(log_every))
        self._run = None          # swanlab run 对象(惰性初始化)
        self._swanlab = None      # swanlab 模块(惰性导入)
        self._swanlab_available = False
        self._step = 0
        self._finished = False
        os.makedirs(self.log_dir, exist_ok=True)
        self.history_path = os.path.join(
            self.log_dir, "{}.jsonl".format(self.experiment_name))

    # ---- 内部 ----
    def _ensure_init(self):
        if self._finished:
            return
        if self._swanlab is None:
            try:
                import swanlab
                self._swanlab = swanlab
                self._swanlab_available = True
            except ImportError:
                self._swanlab = False  # 标记: 尝试过且失败
        if (self._run is None and self._swanlab_available
                and self.mode != "disabled"):
            try:
                self._run = self._swanlab.init(
                    project=self.project,
                    experiment_name=self.experiment_name,
                    config=self.config or None,
                    mode=self.mode)
            except TypeError:
                # 兼容不支持 mode/experiment_name 的旧版本
                self._run = self._swanlab.init(
                    project=self.project, config=self.config or None)

    def _write_history(self, metrics, step):
        record = {"step": step, "time": time.time(), "metrics": metrics}
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---- 公开接口 ----
    def log(self, metrics: dict, step=None):
        """记录一批指标(loss/acc/val_loss 等)。

        Args:
            metrics: {"loss": 0.32, "acc": 0.85} 形式的指标字典
            step: 可选步数; None 时自增
        """
        self._step = int(step) if step is not None else self._step + 1
        metrics = {k: float(v) for k, v in metrics.items()
                   if isinstance(v, (int, float))}
        # 本地 JSONL 永远逐条记录(离线分析用)
        self._write_history(metrics, self._step)
        # swanlab 按 log_every 采样上报, 避免高频日志
        if self._step % self.log_every == 0:
            self._ensure_init()
            if self._run is not None:
                try:
                    self._swanlab.log(metrics, step=self._step)
                except Exception as e:  # noqa: 上报失败不影响训练
                    print("[tracker] swanlab log failed: {}".format(e))

    def log_params(self, params: dict):
        """补充记录超参数(合并进 config)。"""
        self.config.update(params)
        if self._run is not None:
            try:
                self._swanlab.config.update(params)
            except Exception:
                pass

    def read_history(self):
        """读取本地 JSONL 历史, 返回 [{"step", "time", "metrics"}], 供分析用。"""
        records = []
        if os.path.exists(self.history_path):
            with open(self.history_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    def summary(self):
        """训练结果摘要: 每个指标的首值/末值/最小值, 便于 fit 后快速分析。"""
        records = self.read_history()
        summary = {}
        for record in records:
            for k, v in record["metrics"].items():
                item = summary.setdefault(
                    k, {"first": v, "last": v, "min": v, "max": v})
                item["last"] = v
                item["min"] = min(item["min"], v)
                item["max"] = max(item["max"], v)
        return summary

    def finish(self):
        """结束本次监控(幂等)。"""
        if self._finished:
            return
        self._finished = True
        if self._run is not None:
            try:
                self._swanlab.finish()
            except Exception:
                pass
            self._run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def __repr__(self):
        return "Tracker(project={}, run={}, mode={}, logs={})".format(
            self.project, self.experiment_name, self.mode, self.history_path)
