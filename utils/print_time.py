import time

from utils.timehelper import time_left, time_str


class TimeLeft(object):
    def __init__(self, args) -> None:
        self.args = args
        self.last_test_T = -1000
        self.last_time = -1000
        self.start_time = time.time()

    def print_time_left(self, logger, t_env):
        t_left = "Estimated time left: {}. Time passed: {}".format(
            time_left(self.last_time, self.last_test_T, t_env, self.args.t_max), time_str(time.time() - self.start_time))
        self.last_test_T = t_env
        self.last_time = time.time()
        logger.console_logger.info(f"t_left: {t_left}")
