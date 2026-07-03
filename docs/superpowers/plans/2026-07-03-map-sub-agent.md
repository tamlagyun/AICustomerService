# Map Sub-Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract AMap decision execution into a dedicated in-process MapAgent that collaborates with the existing customer service Agent.

**Architecture:** The customer service Agent remains the orchestrator. It delegates map-specific AgentDecision execution to `app.agent.map_agent.run_map_agent`, which chooses the concrete AMap tool, emits map-specific status updates, and returns a typed result. The public chat API and frontend contract remain unchanged.

**Tech Stack:** FastAPI backend, LangGraph customer service workflow, Python dataclasses/protocols, pytest-asyncio, ruff.

---

### Task 1: Add MapAgent Unit Tests

**Files:**
- Create: `backend/tests/test_map_agent.py`

- [ ] **Step 1: Write failing tests**

Create tests that import the desired API before implementation:

```python
import asyncio

from app.agent.decision import AgentAction, AgentDecision
from app.agent.map_agent import MapAgentResult, run_map_agent
from app.map_tools import MapToolResult, MapToolStatus


class FakeMapTools:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def search_place(self, keywords: str | None, *, city: str | None = None, types: str | None = None) -> MapToolResult:
        self.calls.append(("search_place", {"keywords": keywords, "city": city, "types": types}))
        return MapToolResult(status=MapToolStatus.FOUND, summary="景点结果", data={"tool": "maps_text_search"})

    async def geocode(self, address: str | None, *, city: str | None = None) -> MapToolResult:
        self.calls.append(("geocode", {"address": address, "city": city}))
        return MapToolResult(status=MapToolStatus.FOUND, summary="坐标结果", data={"tool": "maps_geo"})

    async def route(self, *, origin: str | None, destination: str | None, mode: str | None = None, city: str | None = None, cityd: str | None = None) -> MapToolResult:
        self.calls.append(("route", {"origin": origin, "destination": destination, "mode": mode, "city": city, "cityd": cityd}))
        return MapToolResult(status=MapToolStatus.FOUND, summary="路线结果", data={"tool": "maps_direction_driving"})

    async def navigation(self, *, destination: str | None, destination_name: str | None = None, origin: str | None = None, origin_name: str | None = None, mode: str | None = None, city: str | None = None) -> MapToolResult:
        self.calls.append(("navigation", {"destination": destination, "destination_name": destination_name, "origin": origin, "origin_name": origin_name, "mode": mode, "city": city}))
        return MapToolResult(status=MapToolStatus.FOUND, summary="导航结果", data={"tool": "amap_navigation_uri"})

    async def weather(self, city: str | None) -> MapToolResult:
        self.calls.append(("weather", {"city": city}))
        return MapToolResult(status=MapToolStatus.FOUND, summary="天气结果", data={"tool": "maps_weather"})
```

- [ ] **Step 2: Verify tests fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_map_agent.py -q
```

Expected: failure because `app.agent.map_agent` does not exist.

### Task 2: Implement MapAgent

**Files:**
- Create: `backend/app/agent/map_agent.py`

- [ ] **Step 1: Add production module**

Implement:

```python
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.agent.decision import AgentAction, AgentDecision
from app.map_tools import MapToolResult, build_map_tools

StatusEmitter = Callable[[str], None]


@dataclass(frozen=True)
class MapAgentResult:
    decision: AgentDecision | None
    map_result: MapToolResult


async def run_map_agent(
    decision: AgentDecision | None,
    *,
    message: str,
    map_tools=None,
    emit_status: StatusEmitter | None = None,
) -> MapAgentResult:
    ...
```

Rules:
- Emit `地图 Agent 正在分析地图子任务` before selecting the concrete map action.
- Emit `地图 Agent 正在调用高德地图工具` before invoking `map_tools`.
- Use `build_map_tools()` only when no test double is injected.
- Return `MapAgentResult(decision=decision, map_result=result)`.
- If `decision` is missing or not map-related, fall back to place search with `keywords=message`.

- [ ] **Step 2: Verify unit tests pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_map_agent.py -q
```

Expected: all MapAgent tests pass.

### Task 3: Delegate From Customer Service Agent

**Files:**
- Modify: `backend/app/agent/customer_service.py`
- Modify: `backend/tests/test_llm_agent_workflow.py`

- [ ] **Step 1: Add workflow regression tests**

Add a test that patches `app.agent.customer_service.run_map_agent`, asks a map question, and asserts:
- customer service Agent calls `run_map_agent`
- returned `map_result` is used in final `ChatResponse`
- status events include both main Agent and map Agent status messages

- [ ] **Step 2: Run regression test and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_agent_workflow.py::test_customer_service_delegates_map_work_to_map_agent -q
```

Expected: failure because customer service still calls `build_map_tools()` directly.

- [ ] **Step 3: Replace direct map tool execution**

Change `retrieve_map_data` to:
- emit main status `正在委托地图 Agent`
- call `run_map_agent(_map_decision_for_tool_call(state), message=state["normalized_message"], emit_status=lambda text: _emit_status(state, text))`
- store `result.map_result` in `state["map_result"]`
- preserve `map_decision` when returned

- [ ] **Step 4: Verify regression test passes**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_llm_agent_workflow.py::test_customer_service_delegates_map_work_to_map_agent -q
```

Expected: pass.

### Task 4: Full Verification

**Files:**
- All modified backend files

- [ ] **Step 1: Run backend test suite**

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected: all tests pass.

- [ ] **Step 2: Run backend lint**

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

Expected: all checks pass.

- [ ] **Step 3: Run frontend validation**

```powershell
npm test
npm run build
```

Expected: frontend tests and build pass because API shape did not change.

- [ ] **Step 4: Run diff check**

```powershell
git diff --check
```

Expected: no whitespace errors.
