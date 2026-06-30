export type ChatSource = {
  title: string;
  source_type: string;
  reference: string;
};

export type ChatResponse = {
  reply: string;
  sources: ChatSource[];
  handoff: boolean;
};

export async function sendChatMessage(message: string): Promise<ChatResponse> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: "local-session",
      message,
    }),
  });

  if (!response.ok) {
    throw new Error("Chat API request failed");
  }

  return response.json() as Promise<ChatResponse>;
}

export type ChatStreamHandlers = {
  onToken: (text: string) => void;
  onDone: (response: Pick<ChatResponse, "sources" | "handoff">) => void;
  onStatus?: (message: string) => void;
};

export async function sendChatMessageStream(
  message: string,
  handlers: ChatStreamHandlers,
): Promise<void> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: "local-session",
      message,
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error("Chat stream API request failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const rawEvent of events) {
      handleSseEvent(rawEvent, handlers);
    }
  }

  if (buffer.trim()) {
    handleSseEvent(buffer, handlers);
  }
}

function handleSseEvent(rawEvent: string, handlers: ChatStreamHandlers): void {
  const eventLine = rawEvent.split("\n").find((line) => line.startsWith("event:"));
  const dataLine = rawEvent.split("\n").find((line) => line.startsWith("data:"));
  if (!eventLine || !dataLine) {
    return;
  }

  const event = eventLine.replace("event:", "").trim();
  const data = JSON.parse(dataLine.replace("data:", "").trim()) as Record<string, unknown>;

  if (event === "status" && typeof data.message === "string") {
    handlers.onStatus?.(data.message);
  }
  if (event === "token" && typeof data.text === "string") {
    handlers.onToken(data.text);
  }
  if (event === "done") {
    handlers.onDone({
      sources: Array.isArray(data.sources) ? (data.sources as ChatSource[]) : [],
      handoff: data.handoff === true,
    });
  }
  if (event === "error") {
    throw new Error(typeof data.message === "string" ? data.message : "Chat stream failed");
  }
}
