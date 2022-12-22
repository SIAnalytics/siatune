# Copyright (c) SI-Analytics. All rights reserved.
import multiprocessing
import os
import re
import time
from glob import glob
from typing import Callable, List

from ray import tune


class _BaseBystander(multiprocessing.Process):
    refresh_delay_secs: float = 1

    def __init__(self,
                 change_callbacks: List[Callable] = [],
                 stop_callbaks: List[Callable] = []):
        self._cached_stamp: float = 0
        self._exit = multiprocessing.Event()
        assert change_callbacks or stop_callbaks
        self._change_callbacks = change_callbacks
        self._stop_callbacks = stop_callbaks

    def _detect(self) -> bool:
        raise NotImplementedError

    def shutdown(self):
        self._exit.set()

    def run(self):
        while not self._exit.is_set():
            time.sleep(self.refresh_delay_secs)
            if not self._detect():
                continue
            for f in self._change_callbacks:
                f(self)
        for f in self._stop_callback:
            f(self)


class FileBystander(_BaseBystander):

    def __init__(self, file_name: str, *args, **kwargs):
        self._file_name = file_name
        assert os.path.exists(self._file_name)
        super().__init__(*args, **kwargs)

    def _detect(self) -> False:
        stamp = os.stat(self._file_name).st_size
        if not stamp != self._cached_stamp:
            self._cached_stamp = stamp
            return True
        return False


def reporter_factory(file_name: str, metric: str):

    class Reporter(FileBystander):

        def __init__(self, metric: str, *args, **kwargs):
            self._metric = metric
            super().__init__(*args, **kwargs)

    def ch_callback(self: Reporter) -> None:

        def get_last_line(file_name: str) -> str:
            with open(file_name, 'r') as f:
                last_line = f.readlines()[-1]
            return last_line

        file_name = self._file_name
        metric = self._metric
        dict_reg = r'^\{(.*)\}$'
        last_line = get_last_line(file_name)
        searched = re.search(dict_reg, last_line)
        result = dict()
        if searched:
            # TODO
            result.update(eval(searched.group(1)))
        if metric not in result:
            return
        tune.report(result)

    def st_callback(self: Reporter):
        tune.report(done=True)

    return Reporter(metric, file_name, [ch_callback], [st_callback])


class CkptBystander(_BaseBystander):

    ckpt_suffix: str = '.pth'

    def __init__(self, dir_name: str, *args, **kwargs):
        self.dir_name = dir_name
        assert os.path.isdir(self.dir_name)
        super().__init__(*args, **kwargs)

    def _count_ckpt(self, dir_name) -> int:
        files = glob(os.path.join(dir_name, '*' + self.ckpt_suffix))
        return len(files)

    def _detect(self) -> False:
        stamp = self._count_ckpt(self.dir_name)
        if not stamp != self._cached_stamp:
            self._cached_stamp = stamp
            return True
        return False


# TODO: CKPT bystander
# def ckpt_link_factory(dir_name: str):
