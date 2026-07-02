# 聊天客服 AI Agent

这是一个面向游戏玩家客服场景的 AI Agent 项目。

## 技术栈

- 前端：React + TypeScript + Vite
- 后端：Python + FastAPI
- Agent：LangGraph + LangChain
- 数据库：MySQL
- 知识库：Markdown / HTML 文件

## 目标

第一阶段先实现本地最小闭环：

1. React 客户端发送玩家问题。
2. FastAPI 服务端接收聊天请求。
3. LangGraph Agent 判断问题并生成回复。
4. 后续接入 MySQL 玩家数据查询工具和 Markdown/HTML 知识库检索工具。

## 目录结构

```text
frontend/        React 客服聊天客户端
backend/         FastAPI 服务端和 Agent 流程
knowledge_base/  Markdown / HTML 知识库文件
docs/            项目设计、接口和开发文档
```

## 本地开发

推荐在 Windows PowerShell 中使用脚本启动。

复制环境变量示例：

```bash
cp .env.example .env
```

检查本地环境：

```powershell
.\scripts\check-env.ps1
```

启动后端：

```powershell
.\scripts\start-backend.ps1
```

启动前端：

```powershell
.\scripts\start-frontend.ps1
```

启动后访问：

```text
前端：http://127.0.0.1:5173
后端：http://127.0.0.1:8000
健康检查：http://127.0.0.1:8000/health
```

如果需要从其他机器通过本机 IP 访问前端，在 `.env` 中配置：

```env
FRONTEND_HOST=0.0.0.0
FRONTEND_PORT=5173
BACKEND_ORIGIN=http://你的后端机器IP:8000
FRONTEND_ORIGIN=http://你的前端机器IP:5173
```

然后重启前端和后端。`FRONTEND_HOST=0.0.0.0` 表示监听所有网卡；访问时仍然使用实际机器 IP，例如：

```text
http://192.168.8.151:5173
```

如果不使用脚本，也可以手动执行下面的命令。

后端依赖安装：

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

启动后端：

```bash
uvicorn app.main:app --reload
```

前端依赖安装：

```bash
cd frontend
npm install
```

启动前端：

```bash
npm run dev
```

开发环境下，前端通过 Vite 代理把 `/api/*` 请求转发到 `http://127.0.0.1:8000`。
前端聊天默认采用 SSE 流式输出：

```text
POST /api/chat/stream
```

后端仍保留普通 REST 一次性返回接口用于兼容：

```text
POST /api/chat
```

聊天响应除了 `reply` 文本，还支持结构化表格：

```json
{
  "reply": "查询到以下数据：",
  "tables": [
    {
      "title": "玩家列表",
      "columns": [
        { "key": "player_id", "label": "玩家ID" },
        { "key": "nickname", "label": "昵称" }
      ],
      "rows": [
        { "player_id": "1", "nickname": "玩家一" }
      ]
    }
  ]
}
```

前端会用真正的 HTML `<table>` 渲染 `tables`，避免让大模型用 Markdown 或空格画表格导致错位。
当前已支持把 MySQL 玩家列表和高德地图地点搜索结果转换为表格。

## 安全边界

- Agent 不直接生成 SQL 查询数据库。
- 数据库访问必须通过后端受控工具封装。
- 所有玩家数据默认视为敏感数据。
- 回复业务问题时应优先引用知识库或后端校验结果。
- 拒绝泄露系统提示词、API key、密钥和内部配置。
- 退款、投诉、申诉等问题默认转人工。
- 回复中会遮蔽手机号和身份证号。

## 大模型配置和切换

后端支持 DeepSeek 和千问的 OpenAI-compatible Chat Completions API。默认关闭，未配置时继续使用规则流程。
前端聊天界面可以选择 `DeepSeek` 或 `千问`，但前端只会发送 `model_provider` 代号；真实 `base_url`、模型名和 API Key 只由后端 `.env` 控制。

启用方式：

```env
LLM_ENABLED=true
LLM_DEFAULT_PROVIDER=deepseek
LLM_ALLOWED_PROVIDERS=deepseek,qwen
LLM_TIMEOUT_SECONDS=20

DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=你的 DeepSeek API Key
DEEPSEEK_MODEL=deepseek-v4-flash

QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=你的阿里云百炼 API Key
QWEN_MODEL=qwen-plus
```

旧的 `LLM_PROVIDER`、`LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL` 仍作为 DeepSeek 兼容配置保留，建议新配置使用 `DEEPSEEK_*` 和 `QWEN_*`。

启用后流程：

```text
玩家问题
  -> 前端发送 model_provider
  -> 安全检查
  -> 后端按白名单选择 DeepSeek 或千问
  -> 选定模型输出动作 JSON
  -> 后端执行受控工具
  -> 选定模型根据工具结果生成最终回复
  -> 后端脱敏后返回前端
```

模型只能选择这些动作：

```text
knowledge_base
mysql_player_profile
mysql_players_list
avatar_generate
amap_place_search
amap_geo
amap_route
amap_navigation
amap_weather
ask_clarification
handoff
direct_answer
```

模型不能生成 SQL，也不能直接访问数据库或外部 MCP。数据库和地图查询都必须通过后端受控工具执行。

## 高德地图 MCP

后端支持通过高德地图 MCP 查询地点、地址、路线和天气，并可生成高德地图导航链接，默认关闭。

启用方式：

```env
AMAP_MCP_ENABLED=true
AMAP_MCP_URL=https://mcp.amap.com/mcp?key=你的高德 Web 服务 Key
AMAP_MCP_TIMEOUT_SECONDS=15
```

当前受控地图工具：

```text
amap_place_search -> 高德 maps_text_search，查询地点/POI
amap_geo          -> 高德 maps_geo，地址或地名转经纬度
amap_route        -> 高德路线规划工具，支持起终点为地址或高德经纬度
amap_navigation   -> 高德 URI API 导航链接，支持目的地为地址或高德经纬度
amap_weather      -> 高德 maps_weather，按城市名或 adcode 查询天气
```

玩家询问“附近网吧在哪里”“杭州西湖在哪里”“从 A 到 B 怎么走”“导航到天安门”“北京天气怎么样”这类问题时，
Agent 会先让大模型决策是否需要地图工具，再由后端调用高德 MCP，最后把工具结果交给模型生成客服回复。

## 会话记忆

后端已支持内存级短期会话记忆：

- 按 `session_id` 隔离。
- 默认保留最近 10 条 user / assistant 消息。
- Agent 决策和最终回复会接收最近历史对话。
- 后端重启后记忆会丢失。

当前前端开发环境固定使用：

```text
session_id=local-session
```

因此同一个浏览器页面内的连续对话会共享上下文。生产环境应改为由登录态、设备 ID 或后端会话系统生成 `session_id`，并把记忆迁移到 MySQL 或 Redis。

## 知识库

当前支持把 Markdown 和 HTML 文件放入 `knowledge_base/`。后端会读取文件并按标题分块，使用轻量关键词检索返回相关内容和来源。

知识库文件在每次请求时读取。开发阶段新增或修改 `knowledge_base/` 下的 `.md`、`.html`、`.htm` 文件后，不需要重启后端。

Markdown 标题支持下面两种写法：

```markdown
## 充值未到账怎么办
##充值未到账怎么办
```

第一版暂不使用向量库，等知识库规模和问答效果需求明确后再升级。

## MySQL 玩家数据

后端已提供受控 MySQL 查询工具，默认关闭。需要联调真实数据库时，在 `.env` 中配置：

```env
MYSQL_ENABLED=true
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=game_readonly
MYSQL_PASSWORD=你的密码
MYSQL_DATABASE=game_customer_service
MYSQL_PLAYERS_TABLE=players
```

当前第一版只实现玩家基础资料查询。默认表字段假设：

```text
players.player_id
players.nickname
players.level
players.server_name
players.status
```

Agent 不会生成 SQL；它只能调用后端固定工具查询玩家资料。
