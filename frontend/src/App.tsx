import { FormEvent, useState } from "react";

import { ChatSource, sendChatMessageStream } from "./api/chat";

type ChatMessage = {
  role: "player" | "agent";
  content: string;
  status?: string;
  statuses?: string[];
  sources?: ChatSource[];
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
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      await sendChatMessageStream(text, {
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
          <span className="status-pill">本地开发</span>
        </header>

        <div className="message-list">
          {messages.map((message, index) => (
            <article key={`${message.role}-${index}`} className={`message ${message.role}`}>
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
