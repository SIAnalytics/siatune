# Copyright (c) SI-Analytics. All rights reserved.
import os
import re
import threading
import time
from glob import glob
from typing import Callable, List

import psutil
from ray.air import session


class _BaseBystander:
    refresh_delay_secs: float = 1

    def __init__(
        self,
        callbacks: List[Callable] = [],
    ):
        self._cached_stamp: float = 0
        self._exit: bool = False
        assert callbacks
        self._callbacks = callbacks
        self._thr = threading.Thread(target=self.run)

    def _detect(self) -> bool:
        raise NotImplementedError

    def shutdown(self):
        self._exit = True

    def start(self):
        self._thr.start()

    def run(self):
        while not self._exit:
            time.sleep(self.refresh_delay_secs)
            if not self._detect():
                continue
            for f in self._callbacks:
                f(self)

    def __del__(self):
        self.shutdown()
        self._thr.join()


class FileBystander(_BaseBystander):

    def __init__(self, file_name: str, *args, **kwargs):
        self._file_name = file_name
        assert os.path.exists(self._file_name)
        super().__init__(*args, **kwargs)

    def _detect(self) -> bool:
        stamp = os.stat(self._file_name).st_size
        if stamp != self._cached_stamp:
            self._cached_stamp = stamp
            return True
        return False


def reporter_factory(file_name: str, metric: str, pid: int):

    class Reporter(FileBystander):

        def __init__(self, metric: str, *args, **kwargs):
            self._metric = metric
            super().__init__(*args, **kwargs)

    def report_callback(self: Reporter) -> None:

        def get_last_line(file_name: str) -> str:
            with open(file_name, 'r') as f:
                last_line = f.readlines()[-1]
            return last_line

        file_name = self._file_name
        metric = self._metric
        dict_reg = r'\w+: [\d.e-]+'
        last_line = get_last_line(file_name)
        result = dict()
        for kv_pair in re.findall(dict_reg, last_line):
            kv_pair = re.split(':', kv_pair)
            if len(kv_pair) != 2:
                continue
            key, value = kv_pair
            result[key] = eval(value)
        if metric in result:
            psutil.suspend(pid)
            session.report(result)
            psutil.resume(pid)

    return Reporter(metric, file_name, [report_callback])


class CkptBystander(_BaseBystander):

    ckpt_suffix: str = '.pth'

    def __init__(self, dir_name: str, *args, **kwargs):
        self.dir_name = dir_name
        assert os.path.isdir(self.dir_name)
        super().__init__(*args, **kwargs)

    def _count_ckpt(self, dir_name) -> int:
        files = glob(os.path.join(dir_name, '*' + self.ckpt_suffix))
        return len(files)

    def _detect(self) -> bool:
        stamp = self._count_ckpt(self.dir_name)
        if stamp != self._cached_stamp:
            self._cached_stamp = stamp
            return True
        return False


# TODO: CKPT bystander
# def ckpt_link_factory(dir_name: str):
