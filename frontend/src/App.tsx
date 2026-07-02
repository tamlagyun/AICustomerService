import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

import {
  ChatImage,
  ChatSource,
  ChatTable,
  DEFAULT_MODEL_PROVIDER,
  ModelProvider,
  sendChatMessageStream,
} from "./api/chat";

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
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = useState("");
  const [modelProvider, setModelProvider] = useState<ModelProvider>(DEFAULT_MODEL_PROVIDER);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
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
      await sendChatMessageStream(text, modelProvider, {
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

  return (
    <main className="page-shell">
      <section className="chat-panel" aria-label="客服聊天">
        <header className="chat-header">
          <div>
            <h1>聊天客服 AI Agent</h1>
            <p>面向游戏玩家咨询、数据查询和知识库问答</p>
          </div>
          <div className="header-actions">
            <label className="model-select">
              <span>模型</span>
              <select
                aria-label="选择大模型"
                value={modelProvider}
                disabled={isSending}
                onChange={(event) => setModelProvider(event.target.value as ModelProvider)}
              >
                <option value="deepseek">DeepSeek</option>
                <option value="qwen">千问</option>
              </select>
            </label>
            <span className="status-pill">本地开发</span>
          </div>
        </header>

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
      </section>
    </main>
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
