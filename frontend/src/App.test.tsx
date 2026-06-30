import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

describe("App", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders the customer service chat shell", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: "聊天客服 AI Agent" })).toBeInTheDocument();
    expect(screen.getByText("你好，我是游戏客服 AI Agent。请描述你遇到的问题。")).toBeInTheDocument();
  });

  it("streams player message to chat API and renders token updates", async () => {
    const encoder = new TextEncoder();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('event: status\ndata: {"message":"正在分析问题"}\n\n'));
            controller.enqueue(encoder.encode('event: status\ndata: {"message":"正在查询工具数据"}\n\n'));
            controller.enqueue(encoder.encode('event: token\ndata: {"text":"请提供"}\n\n'));
            controller.enqueue(encoder.encode('event: token\ndata: {"text":"订单号"}\n\n'));
            controller.enqueue(
              encoder.encode(
                'event: done\ndata: {"sources":[{"title":"充值未到账怎么办","source_type":"knowledge_base","reference":"knowledge_base/sample.md#充值未到账怎么办"}],"handoff":false}\n\n',
              ),
            );
            controller.close();
          },
        }),
        { status: 200, headers: { "Content-Type": "text/event-stream" } },
      ),
    );
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText("输入玩家问题"), "充值不到账怎么办？");
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat/stream",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: "local-session",
          message: "充值不到账怎么办？",
        }),
      }),
    );
    expect(screen.getByText("充值不到账怎么办？")).toBeInTheDocument();
    expect(await screen.findByText("正在查询工具数据")).toBeInTheDocument();
    expect(await screen.findByText("请提供订单号")).toBeInTheDocument();
    expect(screen.getByText("来源：充值未到账怎么办")).toBeInTheDocument();
  });

  it("shows an error message when REST chat API fails", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(new Response("error", { status: 500 }));
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText("输入玩家问题"), "我的角色卡住了");
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(await screen.findByText("发送失败，请稍后重试。")).toBeInTheDocument();
  });
});
