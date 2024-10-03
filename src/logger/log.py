from logging import (
    getLogger,
    basicConfig,
    StreamHandler,
    Formatter,
    DEBUG,
    INFO,
    WARNING,
    ERROR
)
from logging.handlers import TimedRotatingFileHandler
from os import path
from pathlib import Path

from src.logger.log_config import LoggerConfig

(CUSTOM_LOG_FOLDER,
 DEBUG_LOG_LEVEL,
 ERROR_LOG_LEVEL,
 LOG_HISTORY_DAYS_LIMIT) = (LoggerConfig.LOG_FOLDER, LoggerConfig.DEBUG_LOG_LEVEL, LoggerConfig.LOG_HISTORY_DAYS_LIMIT,
                            LoggerConfig.ERROR_LOG_LEVEL)

LOG_INDENTATION_UNIT = "  "
LOG_FOLDER = CUSTOM_LOG_FOLDER
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)


class CustomFormatter(Formatter):
    def format(self, record):
        # Add your custom argument to the log record
        record.indentation_space = LOG_INDENTATION_UNIT * int(getattr(record, "indentation_level", 0))
        record.file_info = f"{record.filename}:{record.funcName}:{record.lineno}"
        return super().format(record)


basicConfig(level=DEBUG)

logger = getLogger(__name__)
logger.setLevel(DEBUG)

if len(logger.handlers) == 0:
    formatter = CustomFormatter("%(asctime)s - %(levelname)-8s - %(file_info)-40s - %(indentation_space)s%(message)s")

    syslog = StreamHandler()
    syslog.setFormatter(formatter)
    logger.setLevel(DEBUG)
    logger.addHandler(syslog)

    optional_handlers = []
    if DEBUG_LOG_LEVEL:
        optional_handlers.append((DEBUG, "DEBUG.log", "midnight"))
    if ERROR_LOG_LEVEL:
        optional_handlers.append((ERROR, "ERROR.log", "midnight"))
    for log_level, filename, when in [
                                         (INFO, "INFO.log", "midnight"),
                                         (WARNING, "WARNING.log", "W0"),
                                     ] + optional_handlers:
        fh = TimedRotatingFileHandler(
            path.join(LOG_FOLDER, filename),
            when=when,
            interval=1,
            backupCount=LOG_HISTORY_DAYS_LIMIT,
            utc=True,
        )
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
