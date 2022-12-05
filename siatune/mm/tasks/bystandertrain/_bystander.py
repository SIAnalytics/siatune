import os
from typing import Callable

class _Bystander:
    refresh_delay_secs: float = 1
    def __init__(self, file_name: str, change_callback:Callable, *args, **kwargs):
        self._cached_stamp: float = 0
        self._running :bool = True

        self.file_name = file_name
        self.change_callback = change_callback
        self.args = args
        self.kwargs = kwargs

    def _look(self):
        stamp = os.stat(self.file_name).st_mtime
        if not stamp != self._cached_stamp:
            return
        self._cached_stamp = stamp
        self.change_callback(*self.args, **self.kwargs)
    
    def stop(self):
        self._running = False

    def start(self):
        while self._running:
            try:
                time.sleep(self.refresh_delay_secs)
                self._look()
            except FileNotFoundError:
                pass