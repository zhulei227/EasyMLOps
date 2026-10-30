# -*- coding: utf-8 -*-
"""
统一模型文件存储层（v0.2.8+ 新增）：为全部 save/load 提供本地与远程(HDFS)透明读写。

约定:
    - 本地路径（含 Windows 盘符）行为与原 open(path, mode) 完全一致
    - 远程路径通过 fsspec 解析：
        webhdfs://[user@]host:port/path   -> WebHDFS REST（仅需 requests，无 JVM）
        hdfs://[user@]namenode:port/path  -> pyarrow HadoopFileSystem（需 JVM/libhdfs）
        memory://path                     -> fsspec 内存文件系统（单测用）
    - WebHDFS 亦可经环境变量提供默认连接信息：
        WEBHDFS_HOST / WEBHDFS_PORT / WEBHDFS_USER
      使 "webhdfs://path" 这类省写也可用
"""
import os

__all__ = ["open_model_file", "save_model_bytes", "load_model_bytes", "is_remote_path"]

_REMOTE_SCHEMES = ("webhdfs://", "hdfs://", "viewfs://", "memory://", "s3://", "gs://", "az://")


def is_remote_path(path) -> bool:
    """判断路径是否为 fsspec 支持的远程/虚拟路径（本地 Windows 盘符不受影响）。"""
    if not isinstance(path, str):
        return False
    return any(path.startswith(s) for s in _REMOTE_SCHEMES)


def _resolve_webhdfs_url(path: str) -> str:
    """webhdfs://path 省写时用 WEBHDFS_HOST/PORT/USER 环境变量补全连接信息。"""
    if path.startswith("webhdfs://") and "webhdfs://@" not in path:
        rest = path[len("webhdfs://"):]
        host_in_url = rest.split("/", 1)[0]
        if host_in_url and ("@" in host_in_url or ":" in host_in_url or
                            host_in_url.replace(".", "").isdigit()):
            return path
        host = os.environ.get("WEBHDFS_HOST")
        if host:
            port = os.environ.get("WEBHDFS_PORT", "9870")
            user = os.environ.get("WEBHDFS_USER", "")
            prefix = "{}@{}:{}".format(user, host, port) if user else "{}:{}".format(host, port)
            return "webhdfs://" + prefix + ("/" + rest if rest else "/")
    return path


def open_model_file(path, mode: str = "rb"):
    """
    打开模型文件：本地路径用内建 open，远程路径经 fsspec。

    Args:
        path: 本地路径或 fsspec 支持的远程 URI（webhdfs:// 等）
        mode: "rb"/"wb"/"ab" 等二进制模式

    Returns:
        可上下文管理的二进制文件对象
    """
    if not is_remote_path(path):
        return open(path, mode)
    import fsspec
    if str(path).startswith("webhdfs://"):
        path = _resolve_webhdfs_url(path)
    return fsspec.open(path, mode).open()


def save_model_bytes(data: bytes, path) -> None:
    """把模型字节流写到 path（本地或远程）。"""
    with open_model_file(path, "wb") as f:
        f.write(data)


def load_model_bytes(path) -> bytes:
    """从 path（本地或远程）读取模型字节流。"""
    with open_model_file(path, "rb") as f:
        return f.read()
