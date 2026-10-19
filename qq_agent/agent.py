from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable


Generate = Callable[[list[dict[str, str]]], Awaitable[str]]


@dataclass(frozen=True)
class AgentResponse:
    reply: bool
    text: str | None = None
    mood: str = ""
    positive: float | None = None
    excitement: float | None = None
    daze_time: float | None = None


@dataclass
class SessionState:
    positive: float = 0.0
    excitement: float = 0.0
    history: list[str] = field(default_factory=list)


def _clamp(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(-1.0, min(1.0, number))


def _json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("模型没有返回 JSON 对象") from None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("模型返回的 JSON 无法解析") from exc
    if not isinstance(value, dict):
        raise ValueError("模型返回值不是 JSON 对象")
    return value


def parse_ai_response(response_text: str) -> AgentResponse:
    try:
        data = _json_object(response_text)
    except ValueError:
        return _parse_legacy_response(response_text)
    reply = data.get("reply") is True
    if not reply:
        try:
            daze_time = float(data["daze_time"]) if "daze_time" in data else None
        except (TypeError, ValueError):
            daze_time = None
        return AgentResponse(reply=False, daze_time=daze_time)

    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("reply=true 时必须提供非空 text")
    return AgentResponse(
        reply=True,
        text=text.strip(),
        mood=str(data.get("mood", "")),
        positive=_clamp(data.get("positive")),
        excitement=_clamp(data.get("excitement")),
    )


def _parse_legacy_response(response_text: str) -> AgentResponse:
    if not re.search(r"reply:\s*true", response_text, re.IGNORECASE):
        daze = re.search(r"\[dazeTime:\s*([^]]+)]", response_text)
        try:
            daze_time = float(daze.group(1).strip()) if daze else None
        except ValueError:
            daze_time = None
        return AgentResponse(reply=False, daze_time=daze_time)

    mood_text = re.search(
        r"\[mood:\s*'([^']*)'\s*;\s*text:\s*'([^']*)']",
        response_text,
        re.DOTALL,
    )
    if not mood_text or not mood_text.group(2).strip():
        raise ValueError("旧格式缺少有效回复文本")
    emotion = re.search(
        r"积极\[-1,1]=\s*([-\d.]+)\s*/\s*激动\[-1,1]=\s*([-\d.]+)",
        response_text,
    )
    return AgentResponse(
        reply=True,
        mood=mood_text.group(1),
        text=mood_text.group(2).strip(),
        positive=_clamp(emotion.group(1)) if emotion else None,
        excitement=_clamp(emotion.group(2)) if emotion else None,
    )


class HumanlikeAgent:
    def __init__(
        self,
        generate: Generate,
        *,
        persona: str = "你是一个正在使用 QQ 私聊的朋友。说话自然、简短、有个性，不使用 Markdown。",
        max_history_items: int = 16,
    ) -> None:
        self._generate = generate
        self._persona = persona
        self._max_history_items = max_history_items
        self._sessions: dict[int, SessionState] = {}
        self._session_locks: dict[int, asyncio.Lock] = {}

    async def reply(self, user_id: int, user_message: str) -> str | None:
        lock = self._session_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            return await self._reply_locked(user_id, user_message)

    async def _reply_locked(self, user_id: int, user_message: str) -> str | None:
        state = self._sessions.setdefault(user_id, SessionState())
        now = datetime.now().strftime("%Y.%m.%d.%H:%M:%S")
        history = "/".join(state.history) if state.history else "没有聊天记录"
        content = (
            f"当前时间：{now}\n"
            f"当前心情：积极[-1,1]={state.positive:.1f}/激动[-1,1]={state.excitement:.1f}\n"
            f"之前的记录：{history}\n"
            f"当前私聊消息：{user_message}\n"
            "结合时间、最近对话、人设和心情决定是否回复。只输出约定的 JSON。"
        )
        messages = [
            {
                "role": "system",
                "content": (
                    f"{self._persona}\n"
                    "输出 JSON："
                    '{"reply":true,"mood":"心情","text":"回复",'
                    '"positive":0.0,"excitement":0.0}。'
                    '如果不回复，输出 {"reply":false,"daze_time":秒数}。'
                    "positive 和 excitement 必须在 -1 到 1 之间。不要输出分析或代码块。"
                ),
            },
            {"role": "user", "content": content},
        ]
        parsed = parse_ai_response(await self._generate(messages))

        state.history.append(f"{now}[用户]'{user_message}'")
        if parsed.reply and parsed.text:
            reply_time = datetime.now().strftime("%Y.%m.%d.%H:%M:%S")
            state.history.append(f"{reply_time}[我]'{parsed.text}'")
            if parsed.positive is not None:
                state.positive = parsed.positive
            if parsed.excitement is not None:
                state.excitement = parsed.excitement
        state.history = state.history[-self._max_history_items :]
        return parsed.text if parsed.reply else None
