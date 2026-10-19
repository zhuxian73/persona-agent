import asyncio
import json

import pytest

from qq_agent.agent import HumanlikeAgent, parse_ai_response


def test_parse_json_reply_and_clamp_mood():
    parsed = parse_ai_response(
        json.dumps(
            {
                "reply": True,
                "mood": "开心",
                "text": "晚上好",
                "positive": 1.8,
                "excitement": -2,
            },
            ensure_ascii=False,
        )
    )

    assert parsed.text == "晚上好"
    assert parsed.positive == 1.0
    assert parsed.excitement == -1.0


def test_parse_no_reply_returns_silence():
    parsed = parse_ai_response('{"reply": false, "daze_time": 3}')

    assert parsed.reply is False
    assert parsed.text is None


def test_parse_legacy_template_for_easy_migration():
    parsed = parse_ai_response(
        "reply: true\n[mood: '有点困'; text: '你怎么还没睡']\n"
        "情绪值：积极[-1,1]=-0.2/激动[-1,1]=0.3"
    )

    assert parsed.reply is True
    assert parsed.mood == "有点困"
    assert parsed.text == "你怎么还没睡"
    assert parsed.positive == -0.2
    assert parsed.excitement == 0.3


@pytest.mark.asyncio
async def test_agent_keeps_sessions_separate():
    prompts = []

    async def fake_generate(messages):
        prompts.append(messages[-1]["content"])
        return '{"reply": true, "mood": "平静", "text": "收到", "positive": 0.2, "excitement": 0.1}'

    agent = HumanlikeAgent(fake_generate)

    assert await agent.reply(10001, "甲的第一句话") == "收到"
    assert await agent.reply(10002, "乙的第一句话") == "收到"

    assert "甲的第一句话" in prompts[0]
    assert "甲的第一句话" not in prompts[1]
    assert "乙的第一句话" in prompts[1]


@pytest.mark.asyncio
async def test_agent_does_not_retry_an_intentional_silence():
    calls = 0

    async def fake_generate(_messages):
        nonlocal calls
        calls += 1
        return '{"reply": false, "daze_time": 5}'

    agent = HumanlikeAgent(fake_generate)

    assert await agent.reply(10001, "在吗") is None
    assert calls == 1


@pytest.mark.asyncio
async def test_same_user_messages_are_processed_in_order():
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    prompts = []

    async def fake_generate(messages):
        prompts.append(messages[-1]["content"])
        if len(prompts) == 1:
            first_started.set()
            await release_first.wait()
        return '{"reply": true, "text": "收到", "positive": 0, "excitement": 0}'

    agent = HumanlikeAgent(fake_generate)
    first = asyncio.create_task(agent.reply(10001, "第一条"))
    await first_started.wait()
    second = asyncio.create_task(agent.reply(10001, "第二条"))
    await asyncio.sleep(0)

    assert len(prompts) == 1
    release_first.set()
    await asyncio.gather(first, second)

    assert "第一条" in prompts[1]
    assert "第二条" in prompts[1]


@pytest.mark.parametrize(
    "response",
    [
        '```json\n{"reply": true, "text": " hello "}\n```',
        'Result: {"reply": true, "text": " hello "}',
    ],
)
def test_parse_json_with_wrapping_text(response):
    assert parse_ai_response(response).text == "hello"


@pytest.mark.parametrize("text", [None, "", "   ", 123])
def test_reply_requires_nonempty_text(text):
    with pytest.raises(ValueError, match="text"):
        parse_ai_response(json.dumps({"reply": True, "text": text}))


@pytest.mark.parametrize(
    "response, expected",
    [
        ('{"reply": false, "daze_time": "2.5"}', 2.5),
        ('{"reply": false, "daze_time": "invalid"}', None),
        ('{"reply": false}', None),
        ('reply: false\n[dazeTime: 2.5]', 2.5),
        ('reply: false\n[dazeTime: invalid]', None),
        ('reply: false', None),
    ],
)
def test_parse_silence_duration(response, expected):
    parsed = parse_ai_response(response)
    assert parsed.reply is False
    assert parsed.daze_time == expected


@pytest.mark.asyncio
async def test_agent_preserves_mood_and_limits_history_including_silence():
    prompts = []
    responses = iter([
        '{"reply": true, "text": "answer-one", "positive": 0.4, "excitement": -0.2}',
        '{"reply": false}',
        '{"reply": true, "text": "answer-three"}',
        '{"reply": false}',
    ])

    async def generate(messages):
        prompts.append(messages)
        return next(responses)

    agent = HumanlikeAgent(generate, persona="custom-persona", max_history_items=2)
    for message in ["message-one", "message-two", "message-three", "message-four"]:
        await agent.reply(123, message)

    assert "custom-persona" in prompts[0][0]["content"]
    assert "message-one" not in prompts[2][1]["content"]
    assert "answer-one" in prompts[2][1]["content"]
    assert "message-two" in prompts[2][1]["content"]
    assert "answer-one" not in prompts[3][1]["content"]
    assert "message-three" in prompts[3][1]["content"]
    assert "answer-three" in prompts[3][1]["content"]
    for messages in prompts[1:]:
        assert "积极[-1,1]=0.4/激动[-1,1]=-0.2" in messages[1]["content"]
