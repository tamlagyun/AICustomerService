import { useState } from "react";

import {
  EvaluationRunResponse,
  EvaluationTool,
  KnowledgeVectorHealthResponse,
  checkKnowledgeVectorHealth,
  rebuildKnowledgeVectorIndex,
} from "../api/chat";

type EvaluationViewProps = {
  isEvaluating: boolean;
  result: EvaluationRunResponse | null;
  error: string | null;
  onRun: () => void;
};

export function EvaluationView({ isEvaluating, result, error, onRun }: EvaluationViewProps) {
  const [isRebuilding, setIsRebuilding] = useState(false);
  const [isCheckingHealth, setIsCheckingHealth] = useState(false);
  const [rebuildMessage, setRebuildMessage] = useState<string | null>(null);
  const [rebuildError, setRebuildError] = useState<string | null>(null);
  const [vectorHealth, setVectorHealth] = useState<KnowledgeVectorHealthResponse | null>(null);
  const [vectorHealthError, setVectorHealthError] = useState<string | null>(null);

  async function handleRebuildVectorIndex() {
    if (isRebuilding) {
      return;
    }

    setRebuildMessage(null);
    setRebuildError(null);
    setIsRebuilding(true);
    try {
      const response = await rebuildKnowledgeVectorIndex();
      if (response.status === "failed") {
        setRebuildError(response.message);
      } else {
        setRebuildMessage(response.message);
      }
    } catch {
      setRebuildError("重建知识库向量库失败，请确认后端和 Ollama embedding 服务已启动。");
    } finally {
      setIsRebuilding(false);
    }
  }

  async function handleCheckVectorHealth() {
    if (isCheckingHealth) {
      return;
    }

    setVectorHealth(null);
    setVectorHealthError(null);
    setIsCheckingHealth(true);
    try {
      setVectorHealth(await checkKnowledgeVectorHealth());
    } catch {
      setVectorHealthError("检查向量库状态失败，请确认后端服务已启动。");
    } finally {
      setIsCheckingHealth(false);
    }
  }

  return (
    <section className="evaluation-view" aria-label="Agent 评测">
      <div className="evaluation-toolbar">
        <div>
          <h2>Agent 评测</h2>
          <p>调用当前模型和已启用工具运行内置用例，评测接口默认关闭。</p>
        </div>
        <div className="evaluation-actions">
          <button type="button" onClick={handleCheckVectorHealth} disabled={isCheckingHealth}>
            {isCheckingHealth ? "检查中" : "检查向量库状态"}
          </button>
          <button type="button" onClick={handleRebuildVectorIndex} disabled={isRebuilding}>
            {isRebuilding ? "重建中" : "重建知识库向量库"}
          </button>
          <button type="button" onClick={onRun} disabled={isEvaluating}>
            {isEvaluating ? "评测中" : "运行评测"}
          </button>
        </div>
      </div>

      {rebuildMessage ? <p className="evaluation-info">{rebuildMessage}</p> : null}
      {rebuildError ? <p className="error-message evaluation-error">{rebuildError}</p> : null}
      {vectorHealth ? (
        <div className={`vector-health ${vectorHealth.status}`}>
          <strong>状态：{vectorHealth.status}</strong>
          <span>{vectorHealth.message}</span>
          <span>文档数：{vectorHealth.document_count}</span>
          <span>collection：{vectorHealth.collection_name}</span>
          {vectorHealth.metadata ? (
            <span>
              文件数：{vectorHealth.metadata.file_count}，模型：
              {vectorHealth.metadata.embedding_model}
            </span>
          ) : null}
        </div>
      ) : null}
      {vectorHealthError ? (
        <p className="error-message evaluation-error">{vectorHealthError}</p>
      ) : null}
      {error ? <p className="error-message evaluation-error">{error}</p> : null}

      {result ? (
        <div className="evaluation-results">
          <div className="evaluation-summary">
            <span>通过 {result.summary.passed}</span>
            <span>失败 {result.summary.failed}</span>
            <span>跳过 {result.summary.skipped}</span>
            <span>总数 {result.summary.total}</span>
          </div>

          <div className="evaluation-case-list">
            {result.results.map((caseResult) => (
              <article className={`evaluation-case ${caseResult.status}`} key={caseResult.case_id}>
                <div className="evaluation-case-header">
                  <strong>{caseResult.name}</strong>
                  <span>{formatEvaluationStatus(caseResult.status)}</span>
                </div>
                {caseResult.error ? <p>{caseResult.error}</p> : null}
                {caseResult.reply ? <p>{caseResult.reply}</p> : null}
                <dl>
                  <div>
                    <dt>工具</dt>
                    <dd>{formatEvaluationTools(caseResult.tools)}</dd>
                  </div>
                  <div>
                    <dt>计划动作</dt>
                    <dd>
                      {caseResult.plan_actions.length ? caseResult.plan_actions.join(", ") : "无"}
                    </dd>
                  </div>
                </dl>
                <ul>
                  {caseResult.checks.map((check) => (
                    <li key={check.name}>
                      {check.passed ? "通过" : "失败"}：{check.name}
                    </li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        </div>
      ) : (
        <p className="evaluation-empty">点击运行后显示评测汇总和用例明细。</p>
      )}
    </section>
  );
}

function formatEvaluationStatus(status: string): string {
  if (status === "passed") {
    return "通过";
  }
  if (status === "failed") {
    return "失败";
  }
  return "跳过";
}

function formatEvaluationTools(tools: EvaluationTool[]): string {
  if (!tools.length) {
    return "无";
  }
  return tools.map(formatEvaluationTool).join(", ");
}

function formatEvaluationTool(tool: EvaluationTool): string {
  if (typeof tool === "string") {
    return tool;
  }
  const name = typeof tool.tool === "string" ? tool.tool : "unknown_tool";
  const status = typeof tool.status === "string" ? tool.status : "";
  return status ? `${name}(${status})` : name;
}
