"""拟人会话逻辑：构建提示词、解析模型输出，并维护各用户的短期状态。"""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable


Generate = Callable[[list[dict[str, str]]], Awaitable[str]]
DEFAULT_PERSONA = "你是一个正在使用 QQ 私聊的朋友。说话自然、简短、有个性，不使用 Markdown。"


@dataclass(frozen=True)
class AgentResponse:
    """统一回复与等待决策；wait_seconds 触发重评估，daze_time 仅兼容旧格式。"""

    reply: bool
    text: str | None = None
    mood: str = ""
    positive: float | None = None
    excitement: float | None = None
    daze_time: float | None = None
    messages: tuple[str, ...] = ()
    wait_seconds: float | None = None


@dataclass
class SessionState:
    """单个用户的内存状态；history 按消息条数计数，重启后不保留。"""

    positive: float = 0.0
    excitement: float = 0.0
    history: list[str] = field(default_factory=list)


def _optional_float(value: object) -> float | None:
    """容忍模型返回数值字符串；缺失或无法转换时返回 None。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: object) -> float | None:
    """将可解析的情绪值限制在 [-1, 1]，保留缺失值语义。"""
    number = _optional_float(value)
    if number is None:
        return None
    return max(-1.0, min(1.0, number))


def _json_object(text: str) -> dict:
    """读取 JSON 对象，兼容代码围栏及对象前后的额外说明。"""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        # 部分模型未遵守纯 JSON 约定，尝试提取其中的对象后再解析。
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
    """优先解析 JSON，无法读取对象时回退到旧模板。"""
    try:
        data = _json_object(response_text)
    except ValueError:
        return _parse_legacy_response(response_text)
    if data.get("reply") is not True:
        if data.get("action") == "wait":
            delay = _optional_float(data.get("wait_seconds"))
            if delay is None or not math.isfinite(delay) or delay <= 0:
                raise ValueError("wait_seconds 必须是有限的正数")
            return AgentResponse(reply=False, wait_seconds=delay)
        # 仅接受 JSON 布尔值 true，避免把字符串 "false" 当成需要回复。
        return AgentResponse(reply=False, daze_time=_optional_float(data.get("daze_time")))

    parts: list[str] = []
    if "messages" in data:
        messages = data["messages"]
        if not isinstance(messages, list) or not all(isinstance(item, str) for item in messages):
            raise ValueError("messages 必须是字符串数组")
        parts = [item.strip() for item in messages if item.strip()]
        if not parts:
            raise ValueError("messages 必须包含非空文本")
        # 超出的分段并入最后一条，限制气泡数但不丢失回复内容。
        if len(parts) > 4:
            parts = parts[:3] + ["\n".join(parts[3:])]
        text = "\n".join(parts)
    else:
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("reply=true 时必须提供非空 text")
    return AgentResponse(
        reply=True,
        text=text.strip(),
        messages=tuple(parts),
        mood=str(data.get("mood", "")),
        positive=_clamp(data.get("positive")),
        excitement=_clamp(data.get("excitement")),
    )


def _parse_legacy_response(response_text: str) -> AgentResponse:
    """兼容旧脚本的 reply、mood/text 和中文情绪值模板。"""
    if not re.search(r"reply:\s*true", response_text, re.IGNORECASE):
        daze = re.search(r"\[dazeTime:\s*([^]]+)]", response_text)
        daze_time = _optional_float(daze.group(1)) if daze else None
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
    """按用户隔离会话；注入异步生成函数，不直接依赖模型 SDK 或 QQ。"""

    def __init__(
        self,
        generate: Generate,
        *,
        persona: str = DEFAULT_PERSONA,
        max_history_items: int = 16,
    ) -> None:
        self._generate = generate
        self._persona = persona
        self._max_history_items = max_history_items
        self._sessions: dict[int, SessionState] = {}
        self._session_locks: dict[int, asyncio.Lock] = {}

    async def reply(self, user_id: int, user_message: str) -> str | list[str] | None:
        """返回单条文本、分段列表或 None；同一用户串行生成，不阻塞其他用户。"""
        lock = self._session_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            return await self._reply_locked(user_id, user_message)

    async def _reply_locked(self, user_id: int, user_message: str) -> str | list[str] | None:
        """在持有用户锁时生成回复，并更新该用户的历史和情绪。"""
        state = self._sessions.setdefault(user_id, SessionState())
        now = datetime.now().strftime("%Y.%m.%d.%H:%M:%S")
        messages = self._build_messages(state, now, user_message)
        response = parse_ai_response(await self._generate(messages))
        if response.wait_seconds is None:
            parts = list(response.messages) or ([response.text] if response.text else [])
            self.remember(user_id, user_message, response, parts)
        if not response.reply:
            return None
        return list(response.messages) if response.messages else response.text

    async def decide(
        self, user_id: int, user_message: str, *, allow_wait: bool = True,
        input_context: str = "",
    ) -> AgentResponse:
        """只生成决策，不修改历史；过时结果可以安全丢弃。"""
        lock = self._session_locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            state = self._sessions.setdefault(user_id, SessionState())
            now = datetime.now().strftime("%Y.%m.%d.%H:%M:%S")
            messages = self._build_messages(state, now, user_message)
            if input_context:
                messages[-1]["content"] += f"\n输入节奏：{input_context}"
            if not allow_wait:
                messages[0]["content"] += (
                    "本轮等待预算已用尽，请根据已有内容回复或保持沉默，不要再输出 wait。"
                )
            return parse_ai_response(await self._generate(messages))

    def remember(
        self, user_id: int, user_message: str, response: AgentResponse, sent: list[str]
    ) -> None:
        """记录已处理的输入和实际发出的气泡，未发送的候选内容不进入历史。"""
        state = self._sessions.setdefault(user_id, SessionState())
        now = datetime.now().strftime("%Y.%m.%d.%H:%M:%S")
        state.history.append(f"{now}[用户]'{user_message}'")
        if sent:
            text = "\n".join(sent)
            state.history.append(f"{now}[我]'{text}'")
            if response.positive is not None:
                state.positive = response.positive
            if response.excitement is not None:
                state.excitement = response.excitement
        state.history = state.history[-self._max_history_items :]

    def _build_messages(
        self, state: SessionState, now: str, user_message: str
    ) -> list[dict[str, str]]:
        """将人设放在 system 消息，将时间、情绪和对话上下文放在 user 消息。"""
        history = "/".join(state.history) if state.history else "没有聊天记录"
        content = (
            f"当前时间：{now}\n"
            f"当前心情：积极[-1,1]={state.positive:.1f}/激动[-1,1]={state.excitement:.1f}\n"
            f"之前的记录：{history}\n"
            f"当前私聊消息：{user_message}\n"
            "结合时间、最近对话、人设和心情决定是否回复。只输出约定的 JSON。"
        )
        return [
            {
                "role": "system",
                "content": (
                    f"{self._persona}\n"
                    "输出 JSON："
                    '{"reply":true,"mood":"心情","messages":["第一条消息","第二条消息"],'
                    '"positive":0.0,"excitement":0.0}。'
                    "messages 是按语义自然分段的 1 到 4 条 QQ 消息。"
                    "需要展开时可以多说几句，分成几个气泡；简短回应只发一条，不要凑数。"
                    "不要机械地逐标点拆句，每条消息可以包含多个句子。"
                    "用户消息可能由连续几个气泡合并而成。结合语义和上下文判断是否说完，"
                    "不要仅凭标点判断。明确提问、结束语或简短寒暄应及时回应；"
                    "明显的铺垫、未完的话或承接词可以继续听。"
                    '若需要等用户补充，输出 {"action":"wait","wait_seconds":1.5}，'
                    "按语境选择 0.5 到 3 秒，不要对每条消息都等待。"
                    '如果不回复，输出 {"reply":false,"daze_time":秒数}。'
                    "positive 和 excitement 必须在 -1 到 1 之间。不要输出分析或代码块。"
                ),
            },
            {"role": "user", "content": content},
        ]
