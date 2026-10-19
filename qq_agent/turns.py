"""按用户收集输入、调度等待决策，并中断尚未发送的过时回复。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from qq_agent.agent import AgentResponse, HumanlikeAgent
from qq_agent.pacing import ReplyPacing


@dataclass
class PendingTurn:
    """由一个处理协程拥有；后续事件只追加输入并设置 changed。"""

    started: float
    last_received: float
    collection_delay: float
    messages: list[str] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)


class TurnCoordinator:
    """每个用户一个处理循环，不额外创建脱离连接生命周期的后台任务。"""

    def __init__(
        self,
        agent: HumanlikeAgent,
        send: Callable[[int, str], Awaitable[None]],
        *,
        collection_delay: float = 0.2,
        max_collection_delay: float = 0.8,
        max_wait: float = 10.0,
        pacing: ReplyPacing | None = None,
    ) -> None:
        self._agent = agent
        self._send = send
        self._collection_delay = collection_delay
        self._max_collection_delay = max_collection_delay
        self._max_wait = max_wait
        self._pacing = pacing if pacing is not None else ReplyPacing()
        self._pending: dict[int, PendingTurn] = {}

    async def handle_message(self, user_id: int, text: str) -> None:
        """首条事件负责处理循环，连发事件合并后立即返回。"""
        now = asyncio.get_running_loop().time()
        pending = self._pending.get(user_id)
        if pending is not None:
            # 用当前连发间隔估计节奏；收集窗口随节奏变化，不固定等两秒。
            gap = now - pending.last_received
            pending.collection_delay = min(
                self._max_collection_delay, max(self._collection_delay, gap * 1.3)
            )
            pending.last_received = now
            pending.messages.append(text)
            pending.changed.set()
            return

        pending = PendingTurn(now, now, self._collection_delay, [text])
        self._pending[user_id] = pending
        try:
            await self._run(user_id, pending)
        finally:
            # 连接退出时取消拥有者即可清理整个用户循环。
            self._pending.pop(user_id, None)

    async def _wait_for_input(self, pending: PendingTurn, seconds: float) -> bool:
        """等待补充或超时；返回 True 表示有新输入，不是必须等满时长。"""
        try:
            await asyncio.wait_for(pending.changed.wait(), timeout=max(0, seconds))
            return True
        except TimeoutError:
            return pending.changed.is_set()

    async def _decide(
        self, user_id: int, pending: PendingTurn, text: str, allow_wait: bool
    ) -> AgentResponse | None:
        """新输入优先于旧模型结果；无论退出原因如何，都回收子任务。"""
        now = asyncio.get_running_loop().time()
        context = (
            f"本轮 {len(pending.messages)} 条消息，"
            f"距离最近输入 {now - pending.last_received:.1f} 秒，"
            f"本轮已过 {now - pending.started:.1f} 秒。"
        )
        generation = asyncio.create_task(self._agent.decide(
            user_id, text, allow_wait=allow_wait, input_context=context
        ))
        incoming = asyncio.create_task(pending.changed.wait())
        try:
            await asyncio.wait({generation, incoming}, return_when=asyncio.FIRST_COMPLETED)
            if pending.changed.is_set():
                return None
            return await generation
        finally:
            generation.cancel()
            incoming.cancel()
            await asyncio.gather(generation, incoming, return_exceptions=True)

    async def _run(self, user_id: int, pending: PendingTurn) -> None:
        """合并 -> 决策 -> 继续听或发送；等待预算不包含模型网络超时。"""
        loop = asyncio.get_running_loop()
        deadline = pending.started + self._max_wait
        waits = 0
        while pending.messages:
            pending.changed.clear()
            remaining = deadline - loop.time()
            if remaining > 0 and await self._wait_for_input(
                pending, min(pending.collection_delay, remaining)
            ):
                continue

            text = "\n".join(pending.messages)
            count = len(pending.messages)
            allow_wait = loop.time() < deadline and waits < 3
            response = await self._decide(user_id, pending, text, allow_wait)
            if response is None:
                continue

            if response.wait_seconds is not None and allow_wait:
                waits += 1
                delay = min(max(response.wait_seconds, 0.01), 3.0, max(0, deadline - loop.time()))
                await self._wait_for_input(pending, delay)
                continue

            parts = list(response.messages) or ([response.text] if response.text else [])
            sent: list[str] = []
            try:
                if response.reply:
                    for part in parts:
                        # 首条扣除已耗时；后续气泡按各自长度模拟打字。
                        delay = (
                            self._pacing.next_delay(part) if sent else
                            self._pacing.first_delay(
                                text, part, elapsed=loop.time() - pending.last_received
                            )
                        )
                        if delay > 0 and await self._wait_for_input(pending, delay):
                            break
                        if pending.changed.is_set():
                            break
                        # 发送本身不强制取消；已写入连接的气泡无法撤回。
                        await self._send(user_id, part)
                        sent.append(part)
            finally:
                if sent or not pending.changed.is_set():
                    self._agent.remember(user_id, text, response, sent)
                    del pending.messages[:count]
            if not pending.messages:
                return
            # 已发送部分回复后，后续输入是新一轮；未发送时则保留原文合并。
            if sent:
                pending.started = loop.time()
                deadline = loop.time() + self._max_wait
                waits = 0
