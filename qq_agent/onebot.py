from __future__ import annotations

from typing import Awaitable, Callable, Protocol


class Agent(Protocol):
    async def reply(self, user_id: int, text: str) -> str | None: ...


SendAction = Callable[[dict], Awaitable[None]]


def private_text_from_event(event: dict, self_id: int) -> tuple[int, str] | None:
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
    def __init__(
        self,
        agent: Agent,
        *,
        self_id: int,
        send_action: SendAction,
        allowed_users: frozenset[int] = frozenset(),
    ) -> None:
        self._agent = agent
        self._self_id = self_id
        self._send_action = send_action
        self._allowed_users = allowed_users

    async def handle_event(self, event: dict) -> None:
        incoming = private_text_from_event(event, self._self_id)
        if incoming is None:
            return
        user_id, text = incoming
        if self._allowed_users and user_id not in self._allowed_users:
            return
        reply = await self._agent.reply(user_id, text)
        if reply:
            await self._send_action(
                {
                    "action": "send_private_msg",
                    "params": {"user_id": user_id, "message": reply},
                }
            )
