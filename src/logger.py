"""结构化日志：JSON 格式，支持 trace_id 追踪

用法:
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("文件加载完成", file="sample.txt", chunks=3)
"""
import logging
import json
import sys
import os
import uuid
from datetime import datetime

_LOGGING_KWARGS = {"exc_info", "stack_info", "stacklevel", "extra"}
_STANDARD_RECORD_ATTRS = set(logging.makeLogRecord({}).__dict__)


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """Allow log.info("event", key=value) and map custom kwargs to extra."""

    def process(self, msg, kwargs):
        extra = dict(kwargs.pop("extra", {}) or {})
        custom = {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
            if key not in _LOGGING_KWARGS
        }
        for key, value in custom.items():
            if key in _STANDARD_RECORD_ATTRS:
                key = f"field_{key}"
            extra[key] = value
        kwargs["extra"] = extra
        return msg, kwargs


class StructuredFormatter(logging.Formatter):
    """JSON 格式输出"""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": self.formatTime(record, "%H:%M:%S"),
            "level": record.levelname,
            "module": record.name,
            "event": record.getMessage(),
        }
        # 额外字段
        for key in ("file", "chunks", "query", "elapsed_ms", "trace_id",
                    "count", "doc", "method", "error"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        # 异常信息
        for key, val in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in entry:
                entry[key] = val
        if record.exc_info and record.exc_info[1]:
            entry["error"] = str(record.exc_info[1])
        return json.dumps(entry, ensure_ascii=False)


class SimpleFormatter(logging.Formatter):
    """简洁格式（兼容 print 习惯）"""

    def format(self, record: logging.LogRecord) -> str:
        prefix = {
            "DEBUG": "  ",
            "INFO": "  ",
            "WARN": "⚠ ",
            "ERROR": "✗ ",
        }.get(record.levelname, "  ")
        return f"{self.formatTime(record, '%H:%M:%S')} {prefix}{record.getMessage()}"


def setup_logging(level: str = "INFO",
                  log_file: str = "",
                  fmt: str = "structured"):
    """初始化全局日志"""
    root = logging.getLogger("rag")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # 控制台
    ch = logging.StreamHandler(sys.stdout)
    if fmt == "structured":
        ch.setFormatter(StructuredFormatter())
    else:
        ch.setFormatter(SimpleFormatter())
    root.addHandler(ch)

    # 文件
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(StructuredFormatter())
        root.addHandler(fh)

    return root


def get_logger(name: str) -> logging.LoggerAdapter:
    """获取带结构化能力的 logger"""
    logger = logging.getLogger(f"rag.{name}")
    return StructuredLoggerAdapter(logger, {})
