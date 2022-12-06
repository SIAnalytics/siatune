# Copyright (c) SI-Analytics. All rights reserved.
import multiprocessing
import os
import re
import time
from typing import Callable, Optional

from ray import tune


class _Bystander(multiprocessing.Process):
    refresh_delay_secs: float = 1

    def __init__(self,
                 file_name: str,
                 change_callback: Callable,
                 stop_callbak: Optional[Callable] = None,
                 *args,
                 **kwargs):
        self._cached_stamp: float = 0
        self._exit = multiprocessing.Event()
        self._change_callback = change_callback
        self._stop_callback = stop_callbak

        self._file_name = file_name
        self._args = args
        self._kwargs = kwargs

    def _look(self):
        stamp = os.stat(self._file_name).st_mtime
        if not stamp != self._cached_stamp:
            return
        self._cached_stamp = stamp
        self._change_callback(self._file_name, *self._args, **self._kwargs)

    def shutdown(self):
        self._exit.set()

    def run(self):
        while not self._exit.is_set():
            try:
                time.sleep(self.refresh_delay_secs)
                self._look()
            except FileNotFoundError:
                pass
        if self._stop_callback is None:
            return
        self._stop_callback(self._file_name, *self._args, **self._kwargs)


def reporter_factory(file_name: str, metric: str):

    def ch_callback(file_name: str, metric: str, *argd, **kwargs) -> None:

        def get_last_line(file_name: str):
            with open(file_name, 'r') as f:
                last_line = f.readlines()[-1]
            return last_line

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

    def st_callback(*args, **kwargs):
        tune.report(done=True)

    return _Bystander(file_name, metric, ch_callback, st_callback)
