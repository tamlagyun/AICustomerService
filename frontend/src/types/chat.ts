import type { ChatImage, ChatSource, ChatTable } from "../api/chat";

export type ActiveView = "chat" | "evaluation";

export type ChatMessage = {
  role: "player" | "agent";
  content: string;
  status?: string;
  statuses?: string[];
  sources?: ChatSource[];
  images?: ChatImage[];
  tables?: ChatTable[];
};
