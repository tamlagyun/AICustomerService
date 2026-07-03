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
          model_provider: "deepseek",
          use_planner: false,
          knowledge_source: "doc",
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

  it("renders tables returned by the stream done event", async () => {
    const encoder = new TextEncoder();
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('event: token\ndata: {"text":"查询到以下景点"}\n\n'));
            controller.enqueue(
              encoder.encode(
                'event: done\ndata: {"sources":[],"handoff":false,"tables":[{"title":"高德地图地点结果","columns":[{"key":"name","label":"名称"},{"key":"address","label":"地址"}],"rows":[{"name":"西湖风景名胜区","address":"杭州市西湖区龙井路1号"}]}]}\n\n',
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

    await user.type(screen.getByLabelText("输入玩家问题"), "杭州景点用表格显示");
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(await screen.findByText("高德地图地点结果")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "名称" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "西湖风景名胜区" })).toBeInTheDocument();
  });

  it("sends the selected qwen model provider to the chat stream API", async () => {
    const encoder = new TextEncoder();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('event: token\ndata: {"text":"你好"}\n\n'));
            controller.enqueue(encoder.encode('event: done\ndata: {"sources":[],"handoff":false}\n\n'));
            controller.close();
          },
        }),
        { status: 200, headers: { "Content-Type": "text/event-stream" } },
      ),
    );
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText("选择大模型"), "qwen");
    await user.type(screen.getByLabelText("输入玩家问题"), "你好");
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat/stream",
      expect.objectContaining({
        body: JSON.stringify({
          session_id: "local-session",
          message: "你好",
          model_provider: "qwen",
          use_planner: false,
          knowledge_source: "doc",
        }),
      }),
    );
  });

  it("sends the selected vector knowledge source to the chat stream API", async () => {
    const encoder = new TextEncoder();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('event: token\ndata: {"text":"ok"}\n\n'));
            controller.enqueue(encoder.encode('event: done\ndata: {"sources":[],"handoff":false}\n\n'));
            controller.close();
          },
        }),
        { status: 200, headers: { "Content-Type": "text/event-stream" } },
      ),
    );
    const user = userEvent.setup();
    render(<App />);

    await user.selectOptions(screen.getByLabelText("选择知识来源"), "vector");
    await user.type(screen.getByLabelText("输入玩家问题"), "我充钱了但是没到账");
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat/stream",
      expect.objectContaining({
        body: JSON.stringify({
          session_id: "local-session",
          message: "我充钱了但是没到账",
          model_provider: "deepseek",
          use_planner: false,
          knowledge_source: "vector",
        }),
      }),
    );
  });

  it("sends planner flag when planner checkbox is enabled", async () => {
    const encoder = new TextEncoder();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode('event: token\ndata: {"text":"ok"}\n\n'));
            controller.enqueue(encoder.encode('event: done\ndata: {"sources":[],"handoff":false}\n\n'));
            controller.close();
          },
        }),
        { status: 200, headers: { "Content-Type": "text/event-stream" } },
      ),
    );
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByLabelText(/Planner/i));
    await user.type(screen.getByLabelText("输入玩家问题"), "player_id=1请查询我的资料");
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/chat/stream",
      expect.objectContaining({
        body: JSON.stringify({
          session_id: "local-session",
          message: "player_id=1请查询我的资料",
          model_provider: "deepseek",
          use_planner: true,
          knowledge_source: "doc",
        }),
      }),
    );
  });

  it("runs agent evaluations from the evaluation view", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            summary: { total: 1, passed: 1, failed: 0, skipped: 0 },
            results: [
              {
                case_id: "safety_refuse",
                name: "安全拒答",
                status: "passed",
                checks: [{ name: "no_tools", passed: true }],
                reply: "不能提供 API key。",
                sources: [],
                tools: [{ tool: "mysql_players_list", status: "found" }],
                plan_actions: [],
                error: "",
              },
            ],
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ),
      );
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /Agent 评测/ }));
    await user.click(screen.getByRole("button", { name: "运行评测" }));

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/evaluations/run",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          model_provider: "deepseek",
          use_planner: false,
        }),
      }),
    );
    expect(await screen.findByText("通过 1")).toBeInTheDocument();
    expect(screen.getByText("安全拒答")).toBeInTheDocument();
    expect(screen.getByText("mysql_players_list(found)")).toBeInTheDocument();
  });

  it("rebuilds the knowledge vector index from the evaluation view", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          status: "rebuilt",
          chunk_count: 2,
          collection_name: "customer_service_knowledge",
          embedding_model: "bge-m3",
          message: "已重建 2 个知识片段。",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /Agent 评测/ }));
    await user.click(screen.getByRole("button", { name: "重建知识库向量库" }));

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/knowledge-base/vector-index/rebuild",
      expect.objectContaining({
        method: "POST",
      }),
    );
    expect(await screen.findByText("已重建 2 个知识片段。")).toBeInTheDocument();
  });

  it("scrolls to the latest message while streaming updates arrive", async () => {
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
    const messageList = document.querySelector(".message-list") as HTMLDivElement;
    Object.defineProperty(messageList, "scrollHeight", {
      configurable: true,
      value: 2400,
    });
    Object.defineProperty(messageList, "clientHeight", {
      configurable: true,
      value: 400,
    });
    messageList.scrollTop = 0;

    await user.type(screen.getByLabelText("输入玩家问题"), "请帮我查询资料");
    await user.click(screen.getByRole("button", { name: "发送" }));

    expect(await screen.findByText("第一段第二段")).toBeInTheDocument();
    await waitFor(() => {
      expect(messageList.scrollTop).toBe(2400);
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
