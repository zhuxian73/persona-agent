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
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=60.0,
            max_retries=2,
        )

        async def generate(messages: list[dict[str, str]]) -> str:
            response = await client.chat.completions.create(
                model=settings.model_name,
                messages=messages,
                temperature=0.1,
                stream=False,
            )
            return response.choices[0].message.content or ""

        self.agent = HumanlikeAgent(generate, persona=settings.persona)

    async def run_forever(self) -> None:
        await reconnect_forever(self._run_connection)

    async def _run_connection(self) -> None:
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
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
            finally:
                for task in tasks:
                    task.cancel()
                with suppress(asyncio.CancelledError):
                    await asyncio.gather(*tasks)

    @staticmethod
    async def _handle_safely(bot: OneBotPrivateBot, event: dict) -> None:
        try:
            await bot.handle_event(event)
        except Exception:
            logger.exception("处理 QQ 私聊消息失败")
