import type { RefObject } from "react";

import type { ChatMessage } from "../types/chat";
import { TableRenderer } from "./TableRenderer";

type MessageListProps = {
  messages: ChatMessage[];
  messageListRef: RefObject<HTMLDivElement>;
  onImageLoad: () => void;
};

export function MessageList({ messages, messageListRef, onImageLoad }: MessageListProps) {
  return (
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
              onLoad={onImageLoad}
            />
          ))}
          {message.tables?.map((table) => (
            <TableRenderer key={table.title} table={table} />
          ))}
        </article>
      ))}
    </div>
  );
}
