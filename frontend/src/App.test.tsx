import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

const originalScrollIntoView = Element.prototype.scrollIntoView;

describe("App", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    if (originalScrollIntoView) {
      Object.defineProperty(Element.prototype, "scrollIntoView", {
        configurable: true,
        value: originalScrollIntoView,
      });
    } else {
      delete (Element.prototype as Partial<Element>).scrollIntoView;
    }
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

  it("renders avatar images returned by the stream done event", async () => {
    const encoder = new TextEncoder();
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('event: status\ndata: {"message":"正在生成头像"}\n\n'));
            controller.enqueue(encoder.encode('event: token\ndata: {"text":"已生成头像"}\n\n'));
            controller.enqueue(
              encoder.encode(
                'event: done\ndata: {"sources":[],"handoff":false,"images":[{"url":"/generated/avatars/player-1.png","alt":"玩家头像"}]}\n\n',
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

    await user.type(screen.getByLabelText("输入玩家问题"), "根据ID=1生成头像");
    await user.click(screen.getByRole("button", { name: "发送" }));

    const image = await screen.findByRole("img", { name: "玩家头像" });
    expect(image).toHaveAttribute("src", "/generated/avatars/player-1.png");
  });

  it("scrolls to the latest message while streaming updates arrive", async () => {
    const scrollIntoView = vi.fn();
    Object.defineProperty(Element.prototype, "scrollIntoView", {
      configurable: true,
      value: scrollIntoView,
    });
    const encoder = new TextEncoder();
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('event: status\ndata: {"message":"正在分析问题"}\n\n'));
            controller.enqueue(encoder.encode('event: token\ndata: {"text":"第一段"}\n\n'));
            controller.enqueue(encoder.encode('event: token\ndata: {"text":"第二段"}\n\n'));
            controller.enqueue(encoder.encode('event: done\ndata: {"sources":[],"handoff":false}\n\n'));
            controller.close();
          },
        }),
        { status: 200, headers: { "Content-Type": "text/event-stream" } },
      ),
    );
    const user = userEvent.setup();
    render(<App />);
    scrollIntoView.mockClear();

    await user.type(screen.getByLabelText("输入玩家问题"), "请帮我查询资料");
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(await screen.findByText("第一段第二段")).toBeInTheDocument();
    await waitFor(() => {
      expect(scrollIntoView).toHaveBeenCalledWith({ behavior: "smooth", block: "end" });
    });
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
