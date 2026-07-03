# 架构说明

## 请求流程

```text
React 客户端
  -> FastAPI /api/chat/stream（SSE 流式输出，携带 model_provider）
    -> LangGraph 客服 Agent
      -> 会话记忆（按 session_id 读取最近 10 条）
      -> MySQL 查询工具
      -> 高德地图 MCP 工具
      -> 知识库检索工具
      -> 回复生成
      -> 会话记忆（写入本轮 user / assistant）
    -> 流式返回 token，结束时返回来源和转人工状态
```

兼容接口：

```text
POST /api/chat
```

该接口仍保留普通 REST 一次性返回，方便测试和兼容旧客户端。

## 结构化展示

后端返回 `reply` 文本，同时可以返回结构化 `tables`。前端不解析大模型文本里的 Markdown 表格，
只渲染后端受控生成的表格数据：

```text
工具结构化结果
  -> TableAdapter
  -> ChatResponse.tables
  -> 前端 TableRenderer
  -> HTML table
```

当前表格转换规则：

- `mysql_players_list`：玩家列表转为“玩家列表”表格。
- `maps_text_search`：高德地点/POI 结果转为“高德地图地点结果”表格。

如果工具只返回自然语言文本，后端不会强行拆表格，避免字段错误和内容错位。

## 前端监听配置

前端 Vite 开发服务器通过环境变量配置：

- `FRONTEND_HOST`：监听地址，默认 `127.0.0.1`。局域网访问使用 `0.0.0.0`。
- `FRONTEND_PORT`：监听端口，默认 `5173`。
- `BACKEND_ORIGIN`：Vite 代理 `/api` 到后端的地址，默认 `http://127.0.0.1:8000`。

后端 CORS 使用 `FRONTEND_ORIGIN` 控制允许访问的前端来源。

## 服务端职责

- 提供 HTTP API。
- 处理登录态、玩家身份和权限。
- 保存聊天记录和审计日志。
- 封装 MySQL 查询工具。
- 封装高德地图 MCP 查询工具。
- 管理知识库索引。

## Agent 职责

- 判断玩家问题类型。
- 读取并使用同一 `session_id` 下的最近对话历史。
- 决定是否需要查询玩家数据。
- 决定是否需要检索知识库。
- 决定是否需要查询高德地图 MCP。
- 整合工具结果生成客服回复。
- 在回复完成后写入本轮对话记忆。
- 在无法确认或高风险场景下转人工。

## LangGraph 流程

当前 Agent 已使用 LangGraph 状态图编排：

```text
analyze_safety
  -> route_safety
    -> generate_refusal_reply
    -> generate_handoff_reply
    -> decide_action_with_llm
      -> classify_question（LLM 未启用或决策失败时回退）
      -> retrieve_player_data
      -> retrieve_map_data
      -> retrieve_knowledge
      -> generate_direct_reply
      -> route_question
        -> retrieve_knowledge -> generate_knowledge_reply
        -> generate_general_reply
  -> finalize
```

节点职责：

- `analyze_safety`：识别拒答、转人工和允许继续处理的请求。
- `decide_action_with_llm`：调用前端选择的 OpenAI-compatible 模型，让模型选择受控动作。
- `classify_question`：根据玩家问题判断转人工、知识库或普通咨询。
- `retrieve_knowledge`：查询 Markdown/HTML 知识库。
- `retrieve_player_data`：查询 MySQL 玩家基础资料。
- `retrieve_map_data`：通过后端受控工具调用高德地图 MCP。
- `generate_knowledge_reply`：使用知识库片段生成带来源回复。
- `generate_llm_final_reply`：模型结合玩家问题和工具结果生成最终回复。
- `generate_refusal_reply`：拒绝泄露系统提示词、密钥或内部配置。
- `generate_handoff_reply`：生成转人工回复。
- `generate_general_reply`：生成普通兜底回复。
- `finalize`：统一输出状态。

## 安全与客服策略

当前阶段使用规则型策略：

- 拒答：系统提示词、API key、密钥、内部提示词等内部信息请求。
- 转人工：退款、投诉、申诉、人工、客服等高风险或人工处理诉求。
- 脱敏：回复中会遮蔽手机号和身份证号。
- 兜底：知识库无结果时要求玩家补充服务器、角色 ID 和具体问题描述。

后续可把这些规则迁移到配置文件或后台管理界面。

## LLM 决策边界

前端只允许传模型 provider 代号，例如 `deepseek` 或 `qwen`。后端根据 `LLM_ALLOWED_PROVIDERS`
白名单选择配置，不接受前端传入 `base_url`、`api_key` 或任意模型名。

当前支持：

```text
deepseek -> DeepSeek OpenAI-compatible Chat Completions
qwen     -> 阿里云百炼/千问 OpenAI-compatible Chat Completions
```

模型只允许输出动作，不允许执行工具：

- `knowledge_base`：后端查询知识库。
- `mysql_player_profile`：后端查询玩家基础资料。
- `mysql_players_list`：后端查询玩家列表。
- `avatar_generate`：根据玩家资料生成本地 PNG 头像。
- `amap_place_search`：通过高德 MCP 查询地点/POI。
- `amap_geo`：通过高德 MCP 将地址或地名解析为经纬度。
- `amap_route`：通过高德 MCP 查询路线，支持起终点为地址或高德经纬度。
- `amap_navigation`：生成高德 URI API 导航链接，支持目的地为地址或高德经纬度。
- `amap_weather`：通过高德 MCP 查询城市天气。
- `ask_clarification`：要求玩家补充必要信息。
- `handoff`：转人工。
- `direct_answer`：直接回复。

如果模型未启用、调用失败、返回非法 JSON、返回未知动作或前端传入未允许的 provider，
系统回退到默认 provider 或规则流程。

## 高德地图 MCP

高德地图 MCP 默认关闭。启用后，后端作为 MCP Client 调用高德官方 Streamable HTTP MCP 服务：

```env
AMAP_MCP_ENABLED=true
AMAP_MCP_URL=https://mcp.amap.com/mcp?key=你的高德 Web 服务 Key
AMAP_MCP_TIMEOUT_SECONDS=15
```

Agent 不直接访问高德 MCP。模型只能选择受控地图动作，并提供结构化参数。
后端再映射到高德 MCP 工具或高德 URI API：

```text
amap_place_search -> maps_text_search
amap_geo          -> maps_geo
amap_route        -> maps_direction_driving / maps_direction_walking / maps_bicycling / maps_direction_transit_integrated
amap_navigation   -> maps_geo + https://uri.amap.com/navigation
amap_weather      -> maps_weather
```

如果未启用或调用失败，后端返回明确的不可用提示，不让模型编造地图结果。

## 会话记忆

当前实现为内存级短期记忆：

- 按 `session_id` 隔离。
- 默认保留最近 10 条 user / assistant 消息。
- 记忆会注入当前选定模型的决策 prompt 和最终回复 prompt。
- 后端进程重启后记忆会丢失。

该方案适合本地开发和第一版多轮对话验证。正式环境建议迁移到 MySQL 或 Redis，并增加过期时间、用户权限和审计策略。

## 知识库检索

当前知识库检索分为两条路径：

- 支持读取 `.md`、`.html`、`.htm` 文件。
- 按 Markdown 标题或 HTML 标题分块。
- 前端聊天页通过 `knowledge_source` 选择 `doc` 或 `vector`。
- `doc` 路径支持 `keyword`、`vector`、`hybrid` 三种本地检索模式，默认 `hybrid`。
- `keyword` 根据玩家问题和知识片段的关键词重叠排序。
- `doc` 路径里的 `vector` 使用本地 Hashing/字符 n-gram 向量索引做语义相似度检索。
- `hybrid` 合并关键词分数和向量相似度。
- `vector` 路径使用 Ollama embedding 模型把玩家问题转成向量，再查询 Chroma 持久化向量库。
- 返回回复内容和来源引用。
- `doc` 路径每次请求都会重新读取知识库文件，新增或修改文件后无需重启服务。
- 向量索引保存到 `VECTOR_STORE_DIR/knowledge_base_vector_index.json`，根据知识库文件内容哈希自动刷新。
- Chroma 向量库保存到 `CHROMA_PERSIST_DIR`，不会在聊天时自动重建。

Chroma 重建由前端“Agent 评测”页的“重建知识库向量库”按钮触发，也可以直接调用：

```text
POST /api/knowledge-base/vector-index/rebuild
```

重建流程：

```text
knowledge_base/*.md|html
  -> 文档分块
  -> Ollama /api/embed，默认模型 bge-m3
  -> Chroma collection: customer_service_knowledge
```

如果玩家在聊天页选择 `vector`，但 Chroma 还没有建立或 Ollama/Chroma 不可用，后端返回明确提示，不静默回退到 `doc` 路径。

## 数据库访问原则

Agent 不能直接写 SQL。后端只暴露受控工具，例如：

- `get_player_profile(player_id)`
- `get_recharge_records(player_id, start_date, end_date)`
- `get_ban_status(player_id)`
- `search_knowledge_base(query)`

当前已实现：

- `get_player_profile(player_id)`：查询 `players` 表中的玩家基础资料。
- `get_players(limit)`：查询 `players` 表玩家列表，默认返回 100 条，最大 1000 条。

当前未实现：

- `get_recharge_records`
- `get_ban_status`

MySQL 默认关闭，设置 `MYSQL_ENABLED=true` 后启用真实连接。默认本地开发不会连接数据库。
