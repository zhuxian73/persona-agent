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
- 模型按语义决定回复分段，一次发送 1 到 4 个 QQ 消息气泡；短回复不强制拆分。
- 发送节奏按输入阅读量和回复长度估计，并加入小幅波动：短回应约 1 到 2 秒，普通回复约 3 到 6 秒，长内容最多约 10 秒。目标时长从最近一次用户输入算起，会扣除收集、等待判断和模型请求已花的时间，模型较慢时不会再叠加完整延迟。
- 后续气泡按各条长度停顿约 0.5 到 5 秒。首条和后续气泡等待期间都能被新输入打断，不会为了模拟打字阻塞其他用户。这些时长是启发式目标，不保证真实网络下的到达时间。
- 如果模型返回超过 4 段，多出的内容并入第 4 条，不直接丢弃。同一用户的回复不交错；收到补充时可停止尚未发送的气泡。
- 输入先用 0.2 到 0.8 秒的小窗口收集，窗口随连发间隔调整；模型结合语义、消息数量和已安静时长，决定回复、继续听或保持沉默。
- 继续听使用独立的 `action: wait`，每次等待最多 3 秒，最多 3 次；从首条消息起 10 秒后不再接受继续等待。模型请求耗时不受这 10 秒约束，仍沿用 SDK 超时。
- 生成期间收到补充会取消旧请求并合并重评估，可能增加模型调用和费用。已发出的气泡不撤回，只有实际发送的内容进入回复历史。
- `reply: false` 会直接保持沉默，不会无限重试。
- 模型请求最多由 SDK 重试两次，OneBot 断线采用 1 到 30 秒退避重连。

当前聊天状态仅保存在内存中，程序重启后会清空。

等待判断是启发式行为，不能保证准确知道用户是否已经说完。等待预算耗尽后仍允许模型保持沉默；若模型继续返回 wait，则结束本次等待，不强制回复。
