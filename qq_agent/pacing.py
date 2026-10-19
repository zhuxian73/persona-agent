"""按阅读量和打字量估计发送节奏，随机波动仅用于避免机械的固定间隔。"""

from __future__ import annotations

import random
from typing import Callable


class ReplyPacing:
    """短回应约 1 到 2 秒，普通回复约 3 到 6 秒，长内容最多约 10 秒。"""

    def __init__(self, *, uniform: Callable[[float, float], float] = random.uniform) -> None:
        self._uniform = uniform

    def first_delay(self, incoming: str, reply: str, *, elapsed: float) -> float:
        """从最近一次输入计时；收集、思考和模型请求已经花掉的时间不重复等待。"""
        incoming_size, reply_size = len(incoming.strip()), len(reply.strip())
        if incoming_size <= 8 and reply_size <= 8:
            target = 1.5
        else:
            target = 2.5 + incoming_size / 45 + reply_size / 18
        target = min(10.0, max(1.0, target * self._uniform(0.85, 1.15)))
        return max(0.0, target - elapsed)

    def next_delay(self, reply: str) -> float:
        """后续气泡只估计当前这条的打字时间，不重复计算阅读输入的时间。"""
        target = 0.7 + len(reply.strip()) / 14
        return min(5.0, max(0.5, target * self._uniform(0.85, 1.15)))
