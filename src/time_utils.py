from collections import deque
import time
from typing import Deque


def now() -> float:
    return time.monotonic()


class FPSMeter:
    def __init__(self, window_size: int = 30) -> None:
        self.window_size = max(2, int(window_size))
        self._timestamps: Deque[float] = deque(maxlen=self.window_size)

    def update(self, t_now: float) -> float:
        self._timestamps.append(float(t_now))
        if len(self._timestamps) < 2:
            return 0.0

        t_first = self._timestamps[0]
        t_last = self._timestamps[-1]
        dt = t_last - t_first
        if dt <= 0.0:
            return 0.0

        return (len(self._timestamps) - 1) / dt
