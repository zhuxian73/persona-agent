"""发送节奏测试固定随机因子，不依赖真实秒级等待。"""

import pytest

from qq_agent.pacing import ReplyPacing


@pytest.fixture
def pacing():
    return ReplyPacing(uniform=lambda low, high: 1.0)


def test_short_acknowledgement_is_faster_than_normal_reply(pacing):
    short = pacing.first_delay("你好", "你好呀", elapsed=0)
    normal = pacing.first_delay("今天发生了一件有意思的事，你想听吗", "当然想听，你快跟我说说发生什么了", elapsed=0)
    assert 1 <= short <= 2
    assert 3 <= normal <= 6
    assert short < normal


def test_long_input_and_reply_need_more_time_but_are_bounded(pacing):
    normal = pacing.first_delay("input" * 4, "reply" * 4, elapsed=0)
    long = pacing.first_delay("input" * 100, "reply" * 100, elapsed=0)
    assert normal < long <= 10


def test_model_and_collection_time_are_deducted(pacing):
    target = pacing.first_delay("普通消息内容", "这是一个普通的回复消息", elapsed=0)
    assert pacing.first_delay("普通消息内容", "这是一个普通的回复消息", elapsed=2) == pytest.approx(target - 2)
    assert pacing.first_delay("普通消息内容", "这是一个普通的回复消息", elapsed=30) == 0


def test_followup_bubbles_use_their_own_length(pacing):
    assert 0.5 <= pacing.next_delay("嗯") < pacing.next_delay("这是接下来需要解释的一段比较长的内容") <= 5


def test_random_variation_remains_small_and_bounded():
    fast = ReplyPacing(uniform=lambda low, high: low)
    slow = ReplyPacing(uniform=lambda low, high: high)
    assert fast.first_delay("hello there", "ordinary reply", elapsed=0) < slow.first_delay("hello there", "ordinary reply", elapsed=0)
    assert slow.first_delay("x" * 1000, "y" * 1000, elapsed=0) <= 10
    assert slow.next_delay("x" * 1000) <= 5
