"""从环境变量和本地 .env 文件加载启动配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from qq_agent.agent import DEFAULT_PERSONA


@dataclass(frozen=True)
class Settings:
    """不可变的运行配置；空白名单表示允许所有用户私聊。"""

    api_key: str
    base_url: str
    model_name: str
    onebot_ws_url: str
    qq_self_id: int
    onebot_access_token: str = ""
    allowed_qq_users: frozenset[int] = frozenset()
    persona: str = DEFAULT_PERSONA

    @classmethod
    def from_env(cls) -> "Settings":
        """加载并校验配置；缺少必填项或 QQ 号无法转换时终止启动。"""
        # 默认不覆盖进程已有的环境变量，便于部署时从外部注入密钥。
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
