import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

import {
  ChatImage,
  ChatSource,
  ChatTable,
  DEFAULT_MODEL_PROVIDER,
  EvaluationRunResponse,
  ModelProvider,
  runAgentEvaluations,
  sendChatMessageStream,
} from "./api/chat";

type ActiveView = "chat" | "evaluation";

type ChatMessage = {
  role: "player" | "agent";
  content: string;
  status?: string;
  statuses?: string[];
  sources?: ChatSource[];
  images?: ChatImage[];
  tables?: ChatTable[];
};

const initialMessages: ChatMessage[] = [
  {
    role: "agent",
    content: "你好，我是游戏客服 AI Agent。请描述你遇到的问题。",
  },
];

export function App() {
  const [activeView, setActiveView] = useState<ActiveView>("chat");
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = useState("");
  const [modelProvider, setModelProvider] = useState<ModelProvider>(DEFAULT_MODEL_PROVIDER);
  const [usePlanner, setUsePlanner] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);
  const [evaluationResult, setEvaluationResult] = useState<EvaluationRunResponse | null>(null);
  const messageListRef = useRef<HTMLDivElement | null>(null);

  const scrollMessagesToBottom = useCallback(() => {
    const messageList = messageListRef.current;
    if (!messageList) {
      return;
    }

    messageList.scrollTop = messageList.scrollHeight;
  }, []);

  useEffect(() => {
    scrollMessagesToBottom();
  }, [messages, scrollMessagesToBottom]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = input.trim();
    if (!text || isSending) {
      return;
    }

    setError(null);
    setInput("");
    setIsSending(true);
    setMessages((current) => [
      ...current,
      { role: "player", content: text },
      { role: "agent", content: "" },
    ]);

    try {
      await sendChatMessageStream(text, modelProvider, usePlanner, {
        onToken(token) {
          setMessages((current) => updateLastAgentMessage(current, (message) => ({
            ...message,
            content: `${message.content}${token}`,
          })));
        },
        onStatus(status) {
          setMessages((current) => updateLastAgentMessage(current, (message) => ({
            ...message,
            status,
            statuses: [...(message.statuses ?? []), status],
          })));
        },
        onDone(response) {
          setMessages((current) => updateLastAgentMessage(current, (message) => ({
            ...message,
            status: "已完成",
            statuses: [...(message.statuses ?? []), "已完成"],
            sources: response.sources,
            images: response.images,
            tables: response.tables,
          })));
        },
      });
    } catch {
      setError("发送失败，请稍后重试。");
    } finally {
      setIsSending(false);
    }
  }

  async function handleRunEvaluations() {
    if (isEvaluating) {
      return;
    }

    setEvaluationError(null);
    setIsEvaluating(true);
    try {
      const result = await runAgentEvaluations(modelProvider, usePlanner);
      setEvaluationResult(result);
    } catch {
      setEvaluationError("评测请求失败，请确认后端已设置 AGENT_EVAL_ENABLED=true。");
    } finally {
      setIsEvaluating(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="chat-panel" aria-label="客服聊天">
        <header className="chat-header">
          <div>
            <h1>聊天客服 AI Agent</h1>
            <p>面向游戏玩家咨询、数据查询和知识库问答</p>
          </div>
          <div className="header-actions">
            <div className="view-tabs" aria-label="功能视图">
              <button
                type="button"
                className={activeView === "chat" ? "active" : ""}
                aria-pressed={activeView === "chat"}
                onClick={() => setActiveView("chat")}
              >
                聊天
              </button>
              <button
                type="button"
                className={activeView === "evaluation" ? "active" : ""}
                aria-pressed={activeView === "evaluation"}
                onClick={() => setActiveView("evaluation")}
              >
                Agent 评测
              </button>
            </div>
            <label className="model-select">
              <span>模型</span>
              <select
                aria-label="选择大模型"
                value={modelProvider}
                disabled={isSending || isEvaluating}
                onChange={(event) => setModelProvider(event.target.value as ModelProvider)}
              >
                <option value="deepseek">DeepSeek</option>
                <option value="qwen">千问</option>
              </select>
            </label>
            <label className="planner-toggle">
              <input
                type="checkbox"
                checked={usePlanner}
                disabled={isSending || isEvaluating}
                onChange={(event) => setUsePlanner(event.target.checked)}
              />
              <span>启用纯模型 Planner</span>
            </label>
            <span className="status-pill">本地开发</span>
          </div>
        </header>

        {activeView === "chat" ? (
          <>
            <div className="message-list" ref={messageListRef}>
              {messages.map((message, index) => (
                <article
                  key={`${message.role}-${index}`}
                  className={`message ${message.role}${message.tables?.length ? " has-table" : ""}`}
                >
                  <span>{message.role === "player" ? "玩家" : "客服 AI"}</span>
                  <p>{message.content}</p>
                  {message.statuses?.map((status, statusIndex) => (
                    <small className="message-status" key={`${status}-${statusIndex}`}>
                      {status}
                    </small>
                  ))}
                  {message.sources?.map((source) => (
                    <small key={source.reference}>来源：{source.title}</small>
                  ))}
                  {message.images?.map((image) => (
                    <img
                      className="message-image"
                      key={image.url}
                      src={image.url}
                      alt={image.alt}
                      onLoad={scrollMessagesToBottom}
                    />
                  ))}
                  {message.tables?.map((table) => (
                    <TableRenderer key={table.title} table={table} />
                  ))}
                </article>
              ))}
            </div>

            <form className="composer" onSubmit={handleSubmit}>
              <input
                aria-label="输入玩家问题"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="例如：充值不到账怎么办？"
                disabled={isSending}
              />
              <button type="submit" disabled={isSending}>
                {isSending ? "发送中" : "发送"}
              </button>
            </form>
            {error ? <p className="error-message">{error}</p> : null}
          </>
        ) : (
          <EvaluationView
            isEvaluating={isEvaluating}
            result={evaluationResult}
            error={evaluationError}
            onRun={handleRunEvaluations}
          />
        )}
      </section>
    </main>
  );
}

function EvaluationView({
  isEvaluating,
  result,
  error,
  onRun,
}: {
  isEvaluating: boolean;
  result: EvaluationRunResponse | null;
  error: string | null;
  onRun: () => void;
}) {
  return (
    <section className="evaluation-view" aria-label="Agent 评测">
      <div className="evaluation-toolbar">
        <div>
          <h2>Agent 评测</h2>
          <p>调用当前模型和已启用工具运行内置用例，评测接口默认关闭。</p>
        </div>
        <button type="button" onClick={onRun} disabled={isEvaluating}>
          {isEvaluating ? "评测中" : "运行评测"}
        </button>
      </div>

      {error ? <p className="error-message evaluation-error">{error}</p> : null}

      {result ? (
        <div className="evaluation-results">
          <div className="evaluation-summary">
            <span>通过 {result.summary.passed}</span>
            <span>失败 {result.summary.failed}</span>
            <span>跳过 {result.summary.skipped}</span>
            <span>总数 {result.summary.total}</span>
          </div>

          <div className="evaluation-case-list">
            {result.results.map((caseResult) => (
              <article className={`evaluation-case ${caseResult.status}`} key={caseResult.case_id}>
                <div className="evaluation-case-header">
                  <strong>{caseResult.name}</strong>
                  <span>{formatEvaluationStatus(caseResult.status)}</span>
                </div>
                {caseResult.error ? <p>{caseResult.error}</p> : null}
                {caseResult.reply ? <p>{caseResult.reply}</p> : null}
                <dl>
                  <div>
                    <dt>工具</dt>
                    <dd>{caseResult.tools.length ? caseResult.tools.join(", ") : "无"}</dd>
                  </div>
                  <div>
                    <dt>计划动作</dt>
                    <dd>
                      {caseResult.plan_actions.length ? caseResult.plan_actions.join(", ") : "无"}
                    </dd>
                  </div>
                </dl>
                <ul>
                  {caseResult.checks.map((check) => (
                    <li key={check.name}>
                      {check.passed ? "通过" : "失败"}：{check.name}
                    </li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        </div>
      ) : (
        <p className="evaluation-empty">点击运行后显示评测汇总和用例明细。</p>
      )}
    </section>
  );
}

function TableRenderer({ table }: { table: ChatTable }) {
  return (
    <section className="message-table" aria-label={table.title}>
      <strong>{table.title}</strong>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {table.columns.map((column) => (
                <th key={column.key} scope="col">
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {table.columns.map((column) => (
                  <td key={column.key}>{formatTableCell(row[column.key])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function formatTableCell(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

function formatEvaluationStatus(status: string): string {
  if (status === "passed") {
    return "通过";
  }
  if (status === "failed") {
    return "失败";
  }
  return "跳过";
}

function updateLastAgentMessage(
  messages: ChatMessage[],
  update: (message: ChatMessage) => ChatMessage,
): ChatMessage[] {
  const next = [...messages];
  for (let index = next.length - 1; index >= 0; index -= 1) {
    if (next[index].role === "agent") {
      next[index] = update(next[index]);
      break;
    }
  }
  return next;
}
