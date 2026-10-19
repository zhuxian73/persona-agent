import pytest

from qq_agent.onebot import OneBotPrivateBot, private_text_from_event


def test_accepts_private_text_message():
    event = {
        "post_type": "message",
        "message_type": "private",
        "user_id": 123,
        "self_id": 999,
        "raw_message": "你好",
    }

    assert private_text_from_event(event, self_id=999) == (123, "你好")


@pytest.mark.parametrize(
    "event",
    [
        {"post_type": "message", "message_type": "group", "user_id": 123, "raw_message": "群消息"},
        {"post_type": "notice", "message_type": "private", "user_id": 123},
        {"post_type": "message", "message_type": "private", "user_id": 999, "raw_message": "自身消息"},
        {"post_type": "message", "message_type": "private", "user_id": 123, "raw_message": "   "},
    ],
)
def test_rejects_non_private_or_self_or_empty_messages(event):
    assert private_text_from_event(event, self_id=999) is None


def test_extracts_only_text_segments():
    event = {
        "post_type": "message",
        "message_type": "private",
        "user_id": 123,
        "message": [
            {"type": "text", "data": {"text": "看这里 "}},
            {"type": "image", "data": {"file": "x.jpg"}},
            {"type": "text", "data": {"text": "好吗"}},
        ],
    }

    assert private_text_from_event(event, self_id=999) == (123, "看这里 好吗")


@pytest.mark.asyncio
async def test_bot_sends_agent_reply_to_original_user():
    sent = []

    class FakeAgent:
        async def reply(self, user_id, text):
            assert (user_id, text) == (123, "你好")
            return "你好呀"

    async def fake_send(action):
        sent.append(action)

    bot = OneBotPrivateBot(FakeAgent(), self_id=999, send_action=fake_send)
    await bot.handle_event(
        {
            "post_type": "message",
            "message_type": "private",
            "user_id": 123,
            "raw_message": "你好",
        }
    )

    assert sent == [
        {"action": "send_private_msg", "params": {"user_id": 123, "message": "你好呀"}}
    ]


@pytest.mark.asyncio
async def test_bot_ignores_users_outside_allowlist():
    class UnexpectedAgent:
        async def reply(self, user_id, text):
            pytest.fail("Blocked users must not reach the agent")

    async def unexpected_send(action):
        pytest.fail("Blocked users must not receive a reply")

    bot = OneBotPrivateBot(
        UnexpectedAgent(), self_id=999, send_action=unexpected_send,
        allowed_users=frozenset({456}),
    )
    await bot.handle_event({
        "post_type": "message", "message_type": "private",
        "user_id": 123, "raw_message": "hello",
    })
