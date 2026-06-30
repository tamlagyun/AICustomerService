# 架构说明

## 请求流程

```text
React 客户端
  -> FastAPI /api/chat/stream（SSE 流式输出）
    -> LangGraph 客服 Agent
      -> MySQL 查询工具
      -> 知识库检索工具
      -> 回复生成
    -> 流式返回 token，结束时返回来源和转人工状态
```

兼容接口：

```text
POST /api/chat
```

该接口仍保留普通 REST 一次性返回，方便测试和兼容旧客户端。

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
- 管理知识库索引。

## Agent 职责

- 判断玩家问题类型。
- 决定是否需要查询玩家数据。
- 决定是否需要检索知识库。
- 整合工具结果生成客服回复。
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
      -> retrieve_knowledge
      -> generate_direct_reply
      -> route_question
        -> retrieve_knowledge -> generate_knowledge_reply
        -> generate_general_reply
  -> finalize
```

节点职责：

- `analyze_safety`：识别拒答、转人工和允许继续处理的请求。
- `decide_action_with_llm`：调用 DeepSeek/OpenAI-compatible 模型，让模型选择受控动作。
- `classify_question`：根据玩家问题判断转人工、知识库或普通咨询。
- `retrieve_knowledge`：查询 Markdown/HTML 知识库。
- `retrieve_player_data`：查询 MySQL 玩家基础资料。
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

模型只允许输出动作，不允许执行工具：

- `knowledge_base`：后端查询知识库。
- `mysql_player_profile`：后端查询玩家基础资料。
- `handoff`：转人工。
- `direct_answer`：直接回复。

如果模型未启用、调用失败、返回非法 JSON 或返回未知动作，系统回退到规则流程。

## 知识库检索

当前阶段先使用轻量关键词检索：

- 支持读取 `.md`、`.html`、`.htm` 文件。
- 按 Markdown 标题或 HTML 标题分块。
- 根据玩家问题和知识片段的关键词重叠排序。
- 返回回复内容和来源引用。
- 开发阶段每次请求都会重新读取知识库文件，新增或修改文件后无需重启服务。
- 为避免单个短词误匹配，检索结果需要达到最低匹配分数才会被采用。

这能先跑通知识库问答闭环。后续当文档量增大或语义匹配要求提高时，再升级为 Embedding + 向量库检索。

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
