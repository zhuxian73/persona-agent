from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model_name: str
    onebot_ws_url: str
    qq_self_id: int
    onebot_access_token: str = ""
    allowed_qq_users: frozenset[int] = frozenset()
    persona: str = "你是一个正在使用 QQ 私聊的朋友。说话自然、简短、有个性，不使用 Markdown。"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        required = {
            "API_KEY": os.getenv("API_KEY", "").strip(),
            "QQ_SELF_ID": os.getenv("QQ_SELF_ID", "").strip(),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"缺少环境变量: {', '.join(missing)}")
        allowed = frozenset(
            int(value.strip())
            for value in os.getenv("ALLOWED_QQ_USERS", "").split(",")
            if value.strip()
        )
        return cls(
            api_key=required["API_KEY"],
            base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1").strip(),
            model_name=os.getenv("MODEL_NAME", "deepseek-reasoner").strip(),
            onebot_ws_url=os.getenv("ONEBOT_WS_URL", "ws://127.0.0.1:3001").strip(),
            qq_self_id=int(required["QQ_SELF_ID"]),
            onebot_access_token=os.getenv("ONEBOT_ACCESS_TOKEN", "").strip(),
            allowed_qq_users=allowed,
            persona=os.getenv("PERSONA", cls.persona).strip(),
        )
