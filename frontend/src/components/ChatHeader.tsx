import type { KnowledgeSource, ModelProvider } from "../api/chat";
import type { ActiveView } from "../types/chat";

type ChatHeaderProps = {
  activeView: ActiveView;
  modelProvider: ModelProvider;
  knowledgeSource: KnowledgeSource;
  usePlanner: boolean;
  disabled: boolean;
  onViewChange: (view: ActiveView) => void;
  onModelProviderChange: (provider: ModelProvider) => void;
  onKnowledgeSourceChange: (source: KnowledgeSource) => void;
  onUsePlannerChange: (enabled: boolean) => void;
};

export function ChatHeader({
  activeView,
  modelProvider,
  knowledgeSource,
  usePlanner,
  disabled,
  onViewChange,
  onModelProviderChange,
  onKnowledgeSourceChange,
  onUsePlannerChange,
}: ChatHeaderProps) {
  return (
    <header className="chat-header">
      <div>
        <h1>聊天客服 AI Agent</h1>
        <p>面向游戏玩家咨询、数据查询和知识库问答</p>
      </div>
      <div className="header-actions">
        <div className="view-tabs" aria-label="功能视图">
          <button
            type="button"
            className={activeView === "chat" ? "active" : ""}
            aria-pressed={activeView === "chat"}
            onClick={() => onViewChange("chat")}
          >
            聊天
          </button>
          <button
            type="button"
            className={activeView === "evaluation" ? "active" : ""}
            aria-pressed={activeView === "evaluation"}
            onClick={() => onViewChange("evaluation")}
          >
            Agent 评测
          </button>
        </div>
        <label className="model-select">
          <span>模型</span>
          <select
            aria-label="选择大模型"
            value={modelProvider}
            disabled={disabled}
            onChange={(event) => onModelProviderChange(event.target.value as ModelProvider)}
          >
            <option value="deepseek">DeepSeek</option>
            <option value="qwen">千问</option>
          </select>
        </label>
        <label className="model-select">
          <span>知识来源</span>
          <select
            aria-label="选择知识来源"
            value={knowledgeSource}
            disabled={disabled}
            onChange={(event) => onKnowledgeSourceChange(event.target.value as KnowledgeSource)}
          >
            <option value="doc">doc文档</option>
            <option value="vector">向量库</option>
          </select>
        </label>
        <label className="planner-toggle">
          <input
            type="checkbox"
            checked={usePlanner}
            disabled={disabled}
            onChange={(event) => onUsePlannerChange(event.target.checked)}
          />
          <span>启用纯模型 Planner</span>
        </label>
        <span className="status-pill">本地开发</span>
      </div>
    </header>
  );
}
