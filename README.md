# QQ 私聊拟人 Agent 测试版

这个项目只复用 MaiBot 的 QQ 接入思路，不依赖 MaiBot 本体。NapCat 登录 QQ 并提供 OneBot 11 WebSocket，Python 程序接收私聊文本、调用 OpenAI 兼容模型，然后把文本回复发回原私聊。

## 1. 准备 NapCat

1. 使用测试 QQ 号安装并登录 [NapCatQQ](https://github.com/NapNeko/NapCatQQ)。非官方 QQ 自动化存在风控风险，不建议使用主账号。
2. 在 NapCat WebUI 的网络配置中添加一个 **WebSocket 服务端**（OneBot 11），监听地址建议设为 `127.0.0.1`，端口例如 `3001`。
3. 如设置了 Access Token，稍后在 `.env` 中填写相同值。
4. 不需要配置 HTTP 服务，也不需要安装 MaiBot。

> NapCat WebUI 常用端口是 `6099`，它不是 OneBot 消息端口。`ONEBOT_WS_URL` 必须填写你在网络配置里新建的 WebSocket 端口。

## 2. 安装项目

在 PowerShell 中进入本目录：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，至少填写：

```dotenv
API_KEY=你的模型密钥
QQ_SELF_ID=NapCat登录的QQ号
ONEBOT_WS_URL=ws://127.0.0.1:3001
```

API Key 只放在 `.env`，不要提交或发送给别人。

## 3. 启动

```powershell
python run.py
```

出现 `已连接 OneBot` 后，用另一个 QQ 号向机器人 QQ 私聊发送纯文本。程序只处理私聊，群消息、图片、语音、文件和空消息都会被忽略。按 `Ctrl+C` 停止。

如果只允许自己的 QQ 测试，可在 `.env` 中设置：

```dotenv
ALLOWED_QQ_USERS=你的测试QQ号
```

## 4. 本地验证

测试不连接 QQ，也不会调用模型 API：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## 与原始脚本的区别

- `input()` 改为接收 OneBot 私聊事件，`print()` 改为 `send_private_msg`。
- 每个 QQ 用户拥有独立心情与最近 16 条记录，不会串聊天。
- 默认要求模型输出 JSON，解析比正则模板稳定；仍兼容原代码的旧格式。
- `reply: false` 会直接保持沉默，不会无限重试。
- 模型请求最多由 SDK 重试两次，OneBot 断线采用 1 到 30 秒退避重连。

当前聊天状态仅保存在内存中，程序重启后会清空。
