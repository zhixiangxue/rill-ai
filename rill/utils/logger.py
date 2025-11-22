"""统一的日志模块

基于 loguru 的简单易用的日志系统，可以在任何地方直接 import 使用。

使用方式：
    from rill.utils.logger import logger
    
    logger.info("这是一条信息")
    logger.debug("调试信息")
    logger.warning("警告信息")
    logger.error("错误信息")

配置方式：
    通过环境变量配置：
    - LOG_LEVEL: 日志级别（DEBUG/INFO/WARNING/ERROR）默认 INFO
    - LOG_TO_FILE: 是否输出到文件（true/false）默认 false
    - LOG_FILE: 日志文件路径，默认 logs/rill.log
"""

import os
import sys
from pathlib import Path

from loguru import logger as _logger

# 移除 loguru 的默认 handler
_logger.remove()

# ===== 配置参数（从环境变量读取） =====
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
LOG_FILE = os.getenv("LOG_FILE", "logs/rill.log")

# ===== 日志格式 =====
# 简洁格式：时间 | 级别 | 消息
SIMPLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

# 详细格式：时间 | 级别 | 位置 | 消息
DETAILED_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# 根据日志级别选择格式（DEBUG 使用详细格式，其他使用简洁格式）
LOG_FORMAT = DETAILED_FORMAT if LOG_LEVEL == "DEBUG" else SIMPLE_FORMAT


# ===== 配置 logger =====
_logger_initialized = False

def _setup_logger():
    """配置 logger（延迟初始化，第一次使用时自动执行）"""
    global _logger_initialized
    if _logger_initialized:
        return
    
    # 读取环境变量
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_to_file = os.getenv("LOG_TO_FILE", "false").lower() == "true"
    log_file = os.getenv("LOG_FILE", "logs/rill.log")
    log_format = DETAILED_FORMAT if log_level == "DEBUG" else SIMPLE_FORMAT
    
    # 1. 控制台输出（始终开启）
    _logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 2. 文件输出（可选）
    if log_to_file:
        # 确保日志目录存在
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        _logger.add(
            log_file,
            format=DETAILED_FORMAT,  # 文件始终使用详细格式
            level=log_level,
            rotation="10 MB",  # 单文件最大 10MB
            retention="30 days",  # 保留 30 天
            compression="zip",  # 压缩旧日志
            backtrace=True,
            diagnose=True,
            encoding="utf-8"
        )
        _logger.info(f"日志文件输出已启用: {log_file}")
    
    _logger_initialized = True


# 包装 logger，实现延迟初始化
class _LazyLogger:
    """延迟初始化的 logger 包装器"""
    
    def __getattr__(self, name):
        # 第一次访问任何方法时，先初始化 logger
        _setup_logger()
        # 然后返回真正的 logger 的属性
        return getattr(_logger, name)


# 导出 logger（用户直接使用这个）
logger = _LazyLogger()

def set_level(level: str):
    """运行时动态设置日志级别
    
    Args:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
    """
    level = level.upper()
    _logger.remove()  # 移除所有 handler
    # 重新添加控制台输出
    _logger.add(
        sys.stdout,
        format=DETAILED_FORMAT if level == "DEBUG" else SIMPLE_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )

__all__ = ["logger", "set_level"]
