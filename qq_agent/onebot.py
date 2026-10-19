"""OneBot 私聊适配：筛选事件、提取文本，并将 Agent 回复发回原用户。"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Protocol


class Agent(Protocol):
    """传输层所需的最小接口；返回 None 表示本次保持沉默。"""

    async def reply(self, user_id: int, text: str) -> str | list[str] | None: ...


SendAction = Callable[[dict], Awaitable[None]]


def private_text_from_event(event: dict, self_id: int) -> tuple[int, str] | None:
    """提取非自身发送的私聊文本；其他事件或空文本返回 None。"""
    if event.get("post_type") != "message" or event.get("message_type") != "private":
        return None
    try:
        user_id = int(event.get("user_id"))
    except (TypeError, ValueError):
        return None
    if user_id == self_id:
        return None

    message = event.get("message")
    if isinstance(message, list):
        # 分段消息只拼接文字，图片、语音等不会进入模型上下文。
        text = "".join(
            str(segment.get("data", {}).get("text", ""))
            for segment in message
            if isinstance(segment, dict) and segment.get("type") == "text"
        )
    else:
        text = str(event.get("raw_message", ""))
    text = text.strip()
    return (user_id, text) if text else None


class OneBotPrivateBot:
    """连接事件与会话逻辑，通过注入发送函数与具体 WebSocket 解耦。"""

    def __init__(
        self,
        agent: Agent,
        *,
        self_id: int,
        send_action: SendAction,
        allowed_users: frozenset[int] = frozenset(),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        handle_message: Callable[[int, str], Awaitable[None]] | None = None,
    ) -> None:
        self._agent = agent
        self._self_id = self_id
        self._send_action = send_action
        self._allowed_users = allowed_users
        self._sleep = sleep
        self._handle_message = handle_message
        self._user_locks: dict[int, asyncio.Lock] = {}

    async def handle_event(self, event: dict) -> None:
        """过滤事件与白名单，按用户串行完成生成和整组消息发送。"""
        incoming = private_text_from_event(event, self._self_id)
        if incoming is None:
            return
        user_id, text = incoming
        if self._allowed_users and user_id not in self._allowed_users:
            return
        # 运行入口可注入输入调度；过滤始终先于收集和模型调用。
        if self._handle_message is not None:
            await self._handle_message(user_id, text)
            return
        # 锁覆盖生成与发送，防止同一用户的下一组回复插入当前分段中。
        lock = self._user_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            reply = await self._agent.reply(user_id, text)
            if not reply:
                return
            messages = [reply] if isinstance(reply, str) else reply
            for index, message in enumerate(messages):
                # 首条立即发送，仅在两条消息之间停顿，不阻塞其他用户。
                if index:
                    await self._sleep(0.8)
                await self._send_action(
                    {
                        "action": "send_private_msg",
                        "params": {"user_id": user_id, "message": message},
                    }
                )
