"""Local rotating logs with secret and prompt content kept out of log messages."""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from agent_test.rag.config import settings

# 主要作用：1 控制台和日志文件输出，2 日志文件按大小轮转，3 日志格式化，4 日志级别设置为 INFO
#          2 使用根日志记录器，所有模块的日志都会被捕获并写入日志文件和控制台

def configure_logging() -> None:
    log_path = os.path.abspath(settings.log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if any(getattr(handler, "baseFilename", None) == log_path for handler in root.handlers):
        return
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(message)s"
    )
    file_handler = RotatingFileHandler(
        log_path, maxBytes=settings.log_max_bytes,
        backupCount=settings.log_backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
