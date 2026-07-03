# Persistent Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist backend runtime logs and Agent audit records to local files.

**Architecture:** Runtime logs use Python logging with a rotating file handler at `logs/app.log`. Agent audit records use JSON Lines at `logs/agent_audit.jsonl`, written after each completed chat response. Existing chat API response shape stays unchanged.

**Tech Stack:** Python logging, RotatingFileHandler, JSONL, FastAPI, pytest, ruff.

---

### Task 1: Logging Configuration

**Files:**
- Create: `backend/tests/test_logging_config.py`
- Create: `backend/app/logging_config.py`
- Modify: `backend/app/config.py`
- Modify: `backend/app/main.py`

- [ ] Write a failing test that creates a temp log directory, calls `configure_logging(settings)`, logs one message, and asserts `app.log` contains it.
- [ ] Implement `configure_logging` with idempotent rotating file setup.
- [ ] Add env settings: `LOG_DIR`, `LOG_LEVEL`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT`, `AGENT_AUDIT_LOG_ENABLED`.
- [ ] Call `configure_logging(settings)` from `main.py`.

### Task 2: Agent Audit JSONL

**Files:**
- Create: `backend/tests/test_agent_audit.py`
- Create: `backend/app/agent_audit.py`

- [ ] Write a failing test that writes an audit event and asserts one JSON object exists in `agent_audit.jsonl`.
- [ ] Implement `write_agent_audit_event(settings, event)`.
- [ ] Ensure audit logging can be disabled by `agent_audit_log_enabled=False`.

### Task 3: Customer Service Audit Integration

**Files:**
- Modify: `backend/tests/test_llm_agent_workflow.py`
- Modify: `backend/app/agent/customer_service.py`

- [ ] Write a failing workflow test that runs `run_customer_service_agent` and asserts audit JSONL contains `session_id`, `player_id`, `message`, `reply`, `handoff`, `sources`, `tools`, and `llm_action`.
- [ ] Implement audit event assembly after `ChatResponse` is built.
- [ ] Include known tool results: MySQL player data, map result, avatar result, and knowledge source count.

### Task 4: Documentation and Verification

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

- [ ] Document where `logs/app.log` and `logs/agent_audit.jsonl` are written.
- [ ] Run backend tests, ruff, frontend tests, frontend build, and diff checks.
