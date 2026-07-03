export type ChatSource = {
  title: string;
  source_type: string;
  reference: string;
};

export type ChatImage = {
  url: string;
  alt: string;
};

export type ChatTableColumn = {
  key: string;
  label: string;
};

export type ChatTable = {
  title: string;
  columns: ChatTableColumn[];
  rows: Record<string, unknown>[];
};

export type ChatResponse = {
  reply: string;
  sources: ChatSource[];
  handoff: boolean;
  images: ChatImage[];
  tables: ChatTable[];
};

export type ModelProvider = "deepseek" | "qwen";
export type KnowledgeSource = "doc" | "vector";

export const DEFAULT_MODEL_PROVIDER: ModelProvider = "deepseek";
export const DEFAULT_KNOWLEDGE_SOURCE: KnowledgeSource = "doc";

export async function sendChatMessage(
  message: string,
  modelProvider: ModelProvider = DEFAULT_MODEL_PROVIDER,
  usePlanner = false,
  knowledgeSource: KnowledgeSource = DEFAULT_KNOWLEDGE_SOURCE,
): Promise<ChatResponse> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: "local-session",
      message,
      model_provider: modelProvider,
      use_planner: usePlanner,
      knowledge_source: knowledgeSource,
    }),
  });

  if (!response.ok) {
    throw new Error("Chat API request failed");
  }

  return response.json() as Promise<ChatResponse>;
}

export type ChatStreamHandlers = {
  onToken: (text: string) => void;
  onDone: (response: Pick<ChatResponse, "sources" | "handoff" | "images" | "tables">) => void;
  onStatus?: (message: string) => void;
};

export async function sendChatMessageStream(
  message: string,
  modelProvider: ModelProvider,
  usePlanner: boolean,
  knowledgeSource: KnowledgeSource,
  handlers: ChatStreamHandlers,
): Promise<void> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: "local-session",
      message,
      model_provider: modelProvider,
      use_planner: usePlanner,
      knowledge_source: knowledgeSource,
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
      images: Array.isArray(data.images) ? (data.images as ChatImage[]) : [],
      tables: Array.isArray(data.tables) ? (data.tables as ChatTable[]) : [],
    });
  }
  if (event === "error") {
    throw new Error(typeof data.message === "string" ? data.message : "Chat stream failed");
  }
}

export type EvaluationStatus = "passed" | "failed" | "skipped";

export type EvaluationSummary = {
  total: number;
  passed: number;
  failed: number;
  skipped: number;
};

export type EvaluationCheck = {
  name: string;
  passed: boolean;
  expected?: unknown;
  actual?: unknown;
  detail?: string;
};

export type EvaluationTool =
  | string
  | {
      tool?: string;
      status?: string;
      summary?: string;
      [key: string]: unknown;
    };

export type EvaluationResult = {
  case_id: string;
  name: string;
  status: EvaluationStatus;
  checks: EvaluationCheck[];
  reply: string;
  sources: ChatSource[];
  tables: ChatTable[];
  tools: EvaluationTool[];
  plan_actions: string[];
  error: string;
};

export type EvaluationRunResponse = {
  summary: EvaluationSummary;
  results: EvaluationResult[];
};

export type KnowledgeVectorRebuildResponse = {
  status: "rebuilt" | "failed" | string;
  chunk_count: number;
  collection_name: string;
  embedding_model: string;
  message: string;
};

export async function runAgentEvaluations(
  modelProvider: ModelProvider,
  usePlanner: boolean,
  caseIds?: string[],
): Promise<EvaluationRunResponse> {
  const body: { model_provider: ModelProvider; use_planner: boolean; case_ids?: string[] } = {
    model_provider: modelProvider,
    use_planner: usePlanner,
  };
  if (caseIds?.length) {
    body.case_ids = caseIds;
  }

  const response = await fetch("/api/evaluations/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error("Agent evaluation request failed");
  }

  return response.json() as Promise<EvaluationRunResponse>;
}

export async function rebuildKnowledgeVectorIndex(): Promise<KnowledgeVectorRebuildResponse> {
  const response = await fetch("/api/knowledge-base/vector-index/rebuild", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });

  if (!response.ok) {
    throw new Error("Knowledge vector rebuild request failed");
  }

  return response.json() as Promise<KnowledgeVectorRebuildResponse>;
}
