import asyncio
import logging

from qq_agent.config import Settings
from qq_agent.runtime import QQAgentRuntime


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    settings = Settings.from_env()
    asyncio.run(QQAgentRuntime(settings).run_forever())


if __name__ == "__main__":
    main()
