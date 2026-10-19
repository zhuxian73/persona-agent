import asyncio

import pytest

from qq_agent.runtime import reconnect_forever


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
