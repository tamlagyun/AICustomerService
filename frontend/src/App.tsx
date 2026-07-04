import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

import {
  DEFAULT_KNOWLEDGE_SOURCE,
  DEFAULT_MODEL_PROVIDER,
  runAgentEvaluations,
  sendChatMessageStream,
} from "./api/chat";
import type { EvaluationRunResponse, KnowledgeSource, ModelProvider } from "./api/chat";
import { ChatHeader } from "./components/ChatHeader";
import { EvaluationView } from "./components/EvaluationView";
import { MessageList } from "./components/MessageList";
import type { ActiveView, ChatMessage } from "./types/chat";

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
  const [knowledgeSource, setKnowledgeSource] = useState<KnowledgeSource>(DEFAULT_KNOWLEDGE_SOURCE);
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
      await sendChatMessageStream(text, modelProvider, usePlanner, knowledgeSource, {
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
        <ChatHeader
          activeView={activeView}
          modelProvider={modelProvider}
          knowledgeSource={knowledgeSource}
          usePlanner={usePlanner}
          disabled={isSending || isEvaluating}
          onViewChange={setActiveView}
          onModelProviderChange={setModelProvider}
          onKnowledgeSourceChange={setKnowledgeSource}
          onUsePlannerChange={setUsePlanner}
        />

        {activeView === "chat" ? (
          <>
            <MessageList
              messages={messages}
              messageListRef={messageListRef}
              onImageLoad={scrollMessagesToBottom}
            />

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
