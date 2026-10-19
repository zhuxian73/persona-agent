import asyncio

import pytest

from qq_agent.agent import HumanlikeAgent
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply, expected",
    [(None, []), ("one. Two!", ["one. Two!"]), (["one", "two", "three"], ["one", "two", "three"])],
)
async def test_bot_sends_parts_with_delays_only_between_messages(reply, expected):
    events = []

    class FakeAgent:
        async def reply(self, user_id, text):
            return reply

    async def send(action):
        assert action["params"]["user_id"] == 123
        events.append(action["params"]["message"])

    async def sleep(delay):
        events.append(delay)

    bot = OneBotPrivateBot(FakeAgent(), self_id=999, send_action=send, sleep=sleep)
    await bot.handle_event({
        "post_type": "message", "message_type": "private",
        "user_id": 123, "raw_message": "hello",
    })

    assert [event for event in events if isinstance(event, str)] == expected
    if len(expected) > 1:
        assert events == ["one", 0.8, "two", 0.8, "three"]
    else:
        assert events == expected


@pytest.mark.asyncio
async def test_reply_groups_do_not_interleave_or_block_other_users():
    paused = asyncio.Event()
    resume = asyncio.Event()
    sent = []
    generated = []

    class FakeAgent:
        async def reply(self, user_id, text):
            generated.append(text)
            return [f"{text}-one", f"{text}-two"]

    async def send(action):
        sent.append(action["params"]["message"])

    async def sleep(delay):
        if not paused.is_set():
            paused.set()
            await resume.wait()

    def event(user_id, text):
        return {"post_type": "message", "message_type": "private",
                "user_id": user_id, "raw_message": text}

    bot = OneBotPrivateBot(FakeAgent(), self_id=999, send_action=send, sleep=sleep)
    first = asyncio.create_task(bot.handle_event(event(123, "first")))
    second = None
    try:
        await asyncio.wait_for(paused.wait(), timeout=1)
        second = asyncio.create_task(bot.handle_event(event(123, "second")))
        await asyncio.sleep(0)
        await asyncio.wait_for(bot.handle_event(event(456, "other")), timeout=1)
        assert generated == ["first", "other"]
        assert sent == ["first-one", "other-one", "other-two"]
    finally:
        resume.set()
        await asyncio.gather(first, *([second] if second else []))
    assert sent == ["first-one", "other-one", "other-two", "first-two", "second-one", "second-two"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output, expected",
    [
        ('{"reply":true,"messages":["one","two","three","four","five"]}',
         ["one", "two", "three", "four\nfive"]),
        ('{"reply":false,"messages":["must not send"]}', []),
        ("reply: true\n[mood: 'calm'; text: 'legacy reply']", ["legacy reply"]),
    ],
)
async def test_model_output_reaches_qq_as_individual_messages(output, expected):
    sent = []

    async def generate(messages):
        return output

    async def send(action):
        assert action["action"] == "send_private_msg"
        sent.append(action["params"]["message"])

    async def sleep(delay):
        pass

    bot = OneBotPrivateBot(
        HumanlikeAgent(generate), self_id=999, send_action=send, sleep=sleep,
    )
    await bot.handle_event({
        "post_type": "message", "message_type": "private",
        "user_id": 123, "raw_message": "hello",
    })
    assert sent == expected
