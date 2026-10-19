"""输入调度回归测试：只替换模型与发送边界，不连接 QQ。"""

import asyncio

import pytest

from qq_agent.agent import HumanlikeAgent
from qq_agent.turns import TurnCoordinator
from qq_agent.pacing import ReplyPacing


@pytest.fixture(autouse=True)
def fast_pacing(monkeypatch):
    """调度测试只测并发与状态，节奏数值由 test_pacing 单独验证。"""
    monkeypatch.setattr(ReplyPacing, "first_delay", lambda *args, **kwargs: 0)
    monkeypatch.setattr(ReplyPacing, "next_delay", lambda *args: 0.05)


def make_coordinator(generate, send, **kwargs):
    return TurnCoordinator(
        HumanlikeAgent(generate), send, collection_delay=0.01,
        max_collection_delay=0.01, **kwargs,
    )


@pytest.mark.asyncio
async def test_burst_messages_are_merged_before_generation():
    prompts, sent = [], []

    async def generate(messages):
        prompts.append(messages[-1]["content"])
        return '{"reply":true,"text":"answer"}'

    async def send(user_id, text):
        sent.append((user_id, text))

    coordinator = make_coordinator(generate, send)
    first = asyncio.create_task(coordinator.handle_message(123, "first"))
    await asyncio.sleep(0)
    await coordinator.handle_message(123, "second")
    await first
    assert len(prompts) == 1
    assert "first\nsecond" in prompts[0]
    assert sent == [(123, "answer")]


@pytest.mark.asyncio
async def test_model_waits_then_reconsiders_without_new_input():
    prompts, sent = [], []

    async def generate(messages):
        prompts.append(messages[-1]["content"])
        if len(prompts) == 1:
            return '{"action":"wait","wait_seconds":0.01}'
        return '{"reply":true,"text":"answer"}'

    async def send(user_id, text):
        sent.append(text)

    coordinator = make_coordinator(generate, send)
    await coordinator.handle_message(123, "intro")
    assert len(prompts) == 2
    assert "intro" in prompts[1]
    assert sent == ["answer"]


@pytest.mark.asyncio
async def test_wait_retry_limit_stops_a_model_that_keeps_waiting():
    calls = 0

    async def generate(messages):
        nonlocal calls
        calls += 1
        return '{"action":"wait","wait_seconds":0.01}'

    async def send(user_id, text):
        pytest.fail("Waiting must not send")

    coordinator = make_coordinator(generate, send)
    await asyncio.wait_for(coordinator.handle_message(123, "intro"), 1)
    assert calls == 4


@pytest.mark.asyncio
async def test_failed_generation_does_not_leave_user_stuck():
    calls, sent = 0, []

    async def generate(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("model unavailable")
        return '{"reply":true,"text":"recovered"}'

    async def send(user_id, text):
        sent.append(text)

    coordinator = make_coordinator(generate, send)
    with pytest.raises(RuntimeError, match="model unavailable"):
        await coordinator.handle_message(123, "first")
    await coordinator.handle_message(123, "second")
    assert sent == ["recovered"]


@pytest.mark.asyncio
async def test_send_failure_preserves_only_successfully_sent_bubbles():
    prompts, sent = [], []

    async def generate(messages):
        prompts.append(messages[-1]["content"])
        return '{"reply":true,"messages":["delivered","failed"]}'

    async def send(user_id, text):
        if text == "failed":
            raise RuntimeError("connection lost")
        sent.append(text)

    agent = HumanlikeAgent(generate)
    coordinator = TurnCoordinator(agent, send, collection_delay=0.01)
    with pytest.raises(RuntimeError, match="connection lost"):
        await coordinator.handle_message(123, "original")
    await agent.decide(123, "next")
    assert sent == ["delivered"]
    assert "delivered" in prompts[1]
    assert "failed" not in prompts[1]


@pytest.mark.asyncio
async def test_new_message_discards_inflight_answer_without_ghost_history():
    started = asyncio.Event()
    cancelled = asyncio.Event()
    prompts, sent = [], []

    async def generate(messages):
        prompts.append(messages[-1]["content"])
        if len(prompts) == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return '{"reply":true,"text":"fresh-answer"}'

    async def send(user_id, text):
        sent.append(text)

    coordinator = make_coordinator(generate, send)
    first = asyncio.create_task(coordinator.handle_message(123, "first"))
    try:
        await asyncio.wait_for(started.wait(), 1)
        await coordinator.handle_message(123, "extra")
        await asyncio.wait_for(first, 1)
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
    assert cancelled.is_set()
    assert len(prompts) == 2
    assert "first\nextra" in prompts[1]
    assert sent == ["fresh-answer"]


@pytest.mark.asyncio
async def test_new_input_interrupts_remaining_bubbles_and_remembers_sent_parts():
    first_sent = asyncio.Event()
    sent, prompts = [], []

    async def generate(messages):
        prompts.append(messages[-1]["content"])
        if len(prompts) == 1:
            return '{"reply":true,"messages":["already-sent","obsolete-part"]}'
        return '{"reply":true,"text":"updated-answer"}'

    async def send(user_id, text):
        sent.append(text)
        if text == "already-sent":
            first_sent.set()

    coordinator = make_coordinator(generate, send)
    first = asyncio.create_task(coordinator.handle_message(123, "original"))
    try:
        await asyncio.wait_for(first_sent.wait(), 1)
        await coordinator.handle_message(123, "followup")
        await asyncio.wait_for(first, 1)
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
    assert sent == ["already-sent", "updated-answer"]
    assert "already-sent" in prompts[1]
    assert "obsolete-part" not in prompts[1]
    assert "followup" in prompts[1]


@pytest.mark.asyncio
async def test_new_input_interrupts_first_bubble_delay_without_ghost_reply(monkeypatch):
    ready = asyncio.Event()
    prompts, sent = [], []

    def first_delay(self, incoming, reply, *, elapsed):
        if incoming == "original":
            ready.set()
            return 30
        return 0

    monkeypatch.setattr(ReplyPacing, "first_delay", first_delay)

    async def generate(messages):
        prompts.append(messages[-1]["content"])
        text = "old-answer" if len(prompts) == 1 else "new-answer"
        return '{"reply":true,"text":"' + text + '"}'

    async def send(user_id, text):
        sent.append(text)

    coordinator = make_coordinator(generate, send)
    first = asyncio.create_task(coordinator.handle_message(123, "original"))
    try:
        await asyncio.wait_for(ready.wait(), 1)
        assert sent == []
        await coordinator.handle_message(123, "extra")
        await asyncio.wait_for(first, 1)
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
    assert sent == ["new-answer"]
    assert "original\nextra" in prompts[1]
    assert "old-answer" not in prompts[1]


@pytest.mark.asyncio
async def test_coordinator_applies_pacing_to_first_and_followup_bubbles(monkeypatch):
    delays, sent = [], []

    class FixedPacing(ReplyPacing):
        def first_delay(self, incoming, reply, *, elapsed):
            assert incoming == "question"
            assert reply == "first"
            assert elapsed > 0
            return 4.2

        def next_delay(self, reply):
            assert reply == "second"
            return 2.3

    async def generate(messages):
        return '{"reply":true,"messages":["first","second"]}'

    async def send(user_id, text):
        sent.append(text)

    async def wait(pending, seconds):
        delays.append(seconds)
        return False

    coordinator = make_coordinator(generate, send, pacing=FixedPacing())
    monkeypatch.setattr(coordinator, "_wait_for_input", wait)
    await coordinator.handle_message(123, "question")
    assert delays == [0.01, 4.2, 2.3]
    assert sent == ["first", "second"]


@pytest.mark.asyncio
async def test_wait_budget_is_bounded_and_silence_does_not_retry():
    calls = 0

    async def generate(messages):
        nonlocal calls
        calls += 1
        return '{"action":"wait","wait_seconds":1000}'

    async def send(user_id, text):
        pytest.fail("Waiting must not send a reply")

    coordinator = make_coordinator(generate, send, max_wait=0.04)
    await asyncio.wait_for(coordinator.handle_message(123, "intro"), 1)
    assert calls == 2

    calls = 0

    async def silent(messages):
        nonlocal calls
        calls += 1
        return '{"reply":false}'

    coordinator = make_coordinator(silent, send)
    await coordinator.handle_message(123, "hello")
    assert calls == 1


@pytest.mark.asyncio
async def test_cancellation_cleans_generation_and_allows_next_message():
    started, cancelled = asyncio.Event(), asyncio.Event()
    calls = 0
    sent = []

    async def generate(messages):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return '{"reply":true,"text":"recovered"}'

    async def send(user_id, text):
        sent.append(text)

    coordinator = make_coordinator(generate, send)
    task = asyncio.create_task(coordinator.handle_message(123, "old"))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    await coordinator.handle_message(123, "new")
    assert sent == ["recovered"]


@pytest.mark.asyncio
async def test_slow_user_does_not_block_another_user():
    started = asyncio.Event()
    sent = []

    async def generate(messages):
        if "slow-user" in messages[-1]["content"]:
            started.set()
            await asyncio.Event().wait()
        return '{"reply":true,"text":"fast-answer"}'

    async def send(user_id, text):
        sent.append((user_id, text))

    coordinator = make_coordinator(generate, send)
    slow = asyncio.create_task(coordinator.handle_message(123, "slow-user"))
    try:
        await asyncio.wait_for(started.wait(), 1)
        await asyncio.wait_for(coordinator.handle_message(456, "fast-user"), 1)
        assert sent == [(456, "fast-answer")]
    finally:
        slow.cancel()
        await asyncio.gather(slow, return_exceptions=True)


@pytest.mark.asyncio
async def test_new_input_wakes_semantic_wait_early():
    waiting = asyncio.Event()
    prompts, sent = [], []

    async def generate(messages):
        prompts.append(messages[-1]["content"])
        if len(prompts) == 1:
            waiting.set()
            return '{"action":"wait","wait_seconds":3}'
        return '{"reply":true,"text":"answer"}'

    async def send(user_id, text):
        sent.append(text)

    coordinator = make_coordinator(generate, send)
    first = asyncio.create_task(coordinator.handle_message(123, "intro"))
    try:
        await asyncio.wait_for(waiting.wait(), 1)
        await coordinator.handle_message(123, "question")
        await asyncio.wait_for(first, 1)
    finally:
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
    assert len(prompts) == 2
    assert "intro\nquestion" in prompts[1]
    assert sent == ["answer"]
