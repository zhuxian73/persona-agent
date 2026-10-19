"""组装模型与 QQ 连接，管理异步消息任务和断线重连。"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from typing import Awaitable, Callable

import websockets
from openai import AsyncOpenAI

from qq_agent.agent import HumanlikeAgent
from qq_agent.config import Settings
from qq_agent.onebot import OneBotPrivateBot

logger = logging.getLogger(__name__)


async def reconnect_forever(
    connect_once: Callable[[], Awaitable[None]],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """连接结束或报错后按 1 到 30 秒指数退避重连；取消信号直接向上传递。"""
    delay = 1
    while True:
        try:
            await connect_once()
            logger.warning("OneBot 连接已关闭；%s 秒后重连", delay)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("OneBot 连接中断: %s；%s 秒后重连", exc, delay)
        await sleep(delay)
        delay = min(delay * 2, 30)


class QQAgentRuntime:
    """复用模型客户端与 Agent；重建 WebSocket 时保留内存会话。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=60.0,
            max_retries=2,
        )

        self.agent = HumanlikeAgent(self._generate, persona=settings.persona)

    async def _generate(self, messages: list[dict[str, str]]) -> str:
        """调用兼容接口并读取首个候选回复；网络重试由 SDK 负责。"""
        response = await self._client.chat.completions.create(
            model=self.settings.model_name,
            messages=messages,
            temperature=0.1,
            stream=False,
        )
        return response.choices[0].message.content or ""

    async def run_forever(self) -> None:
        """持续维护 OneBot 连接，直到上层取消运行。"""
        await reconnect_forever(self._run_connection)

    async def _run_connection(self) -> None:
        """建立一次连接，分发收到的事件，并在退出时回收该连接的任务。"""
        headers = {}
        if self.settings.onebot_access_token:
            headers["Authorization"] = f"Bearer {self.settings.onebot_access_token}"
        async with websockets.connect(
            self.settings.onebot_ws_url,
            additional_headers=headers or None,
            ping_interval=20,
            ping_timeout=20,
        ) as websocket:
            logger.info("已连接 OneBot: %s", self.settings.onebot_ws_url)
            send_lock = asyncio.Lock()

            async def send_action(action: dict) -> None:
                """串行写入当前连接，避免不同用户的回复同时发送。"""
                async with send_lock:
                    await websocket.send(json.dumps(action, ensure_ascii=False))

            bot = OneBotPrivateBot(
                self.agent,
                self_id=self.settings.qq_self_id,
                send_action=send_action,
                allowed_users=self.settings.allowed_qq_users,
            )
            tasks: set[asyncio.Task] = set()
            try:
                async for raw in websocket:
                    try:
                        event = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("忽略无法解析的 OneBot 消息")
                        continue
                    if not isinstance(event, dict):
                        continue
                    task = asyncio.create_task(self._handle_safely(bot, event))
                    # 独立任务避免慢模型请求阻塞接收；强引用保留到任务完成。
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
            finally:
                # 旧连接不再可用，取消尚未完成的请求并等待其退出。
                for task in tasks:
                    task.cancel()
                with suppress(asyncio.CancelledError):
                    await asyncio.gather(*tasks)

    @staticmethod
    async def _handle_safely(bot: OneBotPrivateBot, event: dict) -> None:
        """记录单条消息处理异常，避免它中断其他消息或整个连接。"""
        try:
            await bot.handle_event(event)
        except Exception:
            logger.exception("处理 QQ 私聊消息失败")
