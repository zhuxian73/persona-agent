"""命令行入口：加载配置并启动 QQ 私聊 Agent。"""

import asyncio
import logging

from qq_agent.config import Settings
from qq_agent.runtime import QQAgentRuntime


def main() -> None:
    """初始化日志，在事件循环中持续运行机器人。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    settings = Settings.from_env()
    asyncio.run(QQAgentRuntime(settings).run_forever())


if __name__ == "__main__":
    main()
