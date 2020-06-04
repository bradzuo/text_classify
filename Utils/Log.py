# coding:utf-8
import logging
import re
import sys
import time
import os
# 日志默认配置
# 默认等级

DEFAULT_LOG_LEVEL = logging.DEBUG
# 默认日志格式
DEFAULT_LOG_FMT = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s'
# 默认时间格式
DEFUALT_LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'
# 日期时间
rq = time.strftime('%Y%m%d', time.localtime(time.time()))
# 默认Python日志存放地址
DEFAULT_LOG_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
print(DEFAULT_LOG_DIR)
# print(os.path.dirname(os.getcwd()))

class Logger(object):

    def __init__(self):

        # 项目日志名称构造
        self.DEFAULT_LOG_FILENAME = DEFAULT_LOG_DIR + "/Log/" + "{}".format(rq) + ".log"
        # print(self.DEFAULT_LOG_FILENAME)

        self._logger = logging.getLogger()
        self.formatter = logging.Formatter(fmt=DEFAULT_LOG_FMT, datefmt=DEFUALT_LOG_DATEFMT)
        if not self._logger.handlers:
            self._logger.addHandler(self._get_file_handler(self.DEFAULT_LOG_FILENAME))
        # self._logger.addHandler(self._get_console_handler())
            self._logger.setLevel(DEFAULT_LOG_LEVEL)
            self.base_dir = os.path.dirname(os.getcwd())

    def _get_file_handler(self, filename):
        try:
            filehandler = logging.FileHandler(filename=filename, encoding="utf-8")
        except FileNotFoundError:
            os.mkdir(DEFAULT_LOG_DIR + "/Logs")
            filehandler = logging.FileHandler(filename=filename, encoding="utf-8")

        filehandler.setFormatter(self.formatter)
        return filehandler

    def _get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler

    @property
    def logger(self):
        return self._logger


if __name__ == '__main__':
    logger = Logger().logger
    print('ok')
    logger.debug('this is a logger debug message')
    logger.info('this is a logger info message')
    logger.warning('this is a logger warning message')
    logger.error('this is a logger error message')
    logger.critical('this is a logger critical message')
    # logger.handlers.clear()
