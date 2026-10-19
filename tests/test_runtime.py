import asyncio
import json
from types import SimpleNamespace

import pytest

from qq_agent.config import Settings
from qq_agent.agent import HumanlikeAgent
from qq_agent.runtime import QQAgentRuntime, reconnect_forever


@pytest.mark.asyncio
async def test_normal_disconnect_waits_before_reconnecting():
    connections = 0
    sleeps = []

    async def connect_once():
        nonlocal connections
        connections += 1

    async def fake_sleep(delay):
        sleeps.append(delay)
        if len(sleeps) == 2:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await reconnect_forever(connect_once, sleep=fake_sleep)

    assert connections == 2
    assert sleeps == [1, 2]


@pytest.mark.asyncio
async def test_runtime_uses_configured_model_and_persona(monkeypatch):
    async def create(**kwargs):
        assert kwargs["model"] == "test-model"
        assert kwargs["temperature"] == 0.1
        assert kwargs["stream"] is False
        assert "test-persona" in kwargs["messages"][0]["content"]
        assert "hello" in kwargs["messages"][1]["content"]
        return SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content='{"reply":true,"text":"hi"}'))
        ])

    def make_client(**kwargs):
        assert kwargs == {
            "api_key": "test-key", "base_url": "https://example.invalid/v1",
            "timeout": 60.0, "max_retries": 2,
        }
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr("qq_agent.runtime.AsyncOpenAI", make_client)
    settings = Settings(
        api_key="test-key", base_url="https://example.invalid/v1", model_name="test-model",
        onebot_ws_url="ws://127.0.0.1:3001", qq_self_id=999, persona="test-persona",
    )
    runtime = QQAgentRuntime(settings)

    assert await runtime.agent.reply(123, "hello") == "hi"


@pytest.mark.asyncio
async def test_connection_routes_followups_through_turn_coordinator(monkeypatch):
    from qq_agent.pacing import ReplyPacing

    monkeypatch.setattr(ReplyPacing, "first_delay", lambda *args, **kwargs: 0)
    started, sent = asyncio.Event(), asyncio.Event()
    queue = asyncio.Queue()
    prompts, actions = [], []

    class Socket:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await queue.get()

        async def send(self, raw):
            actions.append(json.loads(raw))
            sent.set()

    async def generate(messages):
        prompts.append(messages[-1]["content"])
        if len(prompts) == 1:
            started.set()
            await asyncio.Event().wait()
        return '{"reply":true,"text":"fresh-answer"}'

    def incoming(text):
        return json.dumps({"post_type": "message", "message_type": "private",
                           "user_id": 123, "raw_message": text})

    monkeypatch.setattr("qq_agent.runtime.websockets.connect", lambda *a, **kw: Socket())
    runtime = QQAgentRuntime(Settings(
        api_key="test-key", base_url="https://example.invalid/v1", model_name="test-model",
        onebot_ws_url="ws://127.0.0.1:3001", qq_self_id=999,
    ))
    runtime.agent = HumanlikeAgent(generate)
    connection = asyncio.create_task(runtime._run_connection())
    try:
        await queue.put(incoming("first"))
        await asyncio.wait_for(started.wait(), 2)
        await queue.put(incoming("followup"))
        await asyncio.wait_for(sent.wait(), 2)
    finally:
        connection.cancel()
        await asyncio.gather(connection, return_exceptions=True)
        await runtime._client.close()
    assert "first\nfollowup" in prompts[1]
    assert actions == [{
        "action": "send_private_msg", "params": {"user_id": 123, "message": "fresh-answer"},
    }]
