import logging
import logging.config
import os
import re
import sys
from datetime import datetime


class LoggerSetUp(object):
    def __init__(self, **kwargs):
        self.is_test = True
        try:
            self.str_today = datetime.today().strftime("%Y%m%d")
            str_env = os.environ.get("ENV_RUN", "development")

            self.str_logger_path = kwargs["logger_path"][str_env]
            self.flt_threshold = float(kwargs.get("logger_threshold", "365"))
            logger_conf = kwargs["logger_conf"]

            self.str_filename = (
                f"{logger_conf['handlers']['file']['filename']}_{self.str_today}.txt"
            )
            logger_conf["handlers"]["file"]["filename"] = (
                    self.str_logger_path + self.str_filename
            )

            if "file" not in logger_conf["loggers"][str_env]["handlers"]:
                del logger_conf["handlers"]["file"]

            logging.config.dictConfig(logger_conf)
            self.logger = logging.getLogger(str_env)
            self.str_id = "iderror"  # noqa
        except Exception as e:
            logging.error(e)
            sys.exit(1)

    def debug(self, str_msg):
        self.logger.debug(str_msg)

    def info(self, str_msg):
        self.logger.info(str_msg)

    def warning(self, str_msg):
        self.logger.warning(str_msg)

    def error(self, str_msg):
        self.logger.error(str_msg)

    def sys_exit(self, int_exit, str_exception_msg):
        if not self.is_test:
            self.logger.error(
                f"{self.str_id} - exit code: {int_exit} - {str_exception_msg}",
                exc_info=sys.exc_info(),
            )
        sys.exit(int_exit)

    @staticmethod
    def __get_datetime_logs(str_x, str_logname):
        return re.findall(f"{str_logname}_(.*?)\.txt", str_x)[0]  # noqa

    @staticmethod
    def __get_ndays(str_1, str_2):
        datetime_delta = datetime.strptime(str_1, "%Y%m%d") - datetime.strptime(
            str_2, "%Y%m%d"
        )
        return datetime_delta.days
