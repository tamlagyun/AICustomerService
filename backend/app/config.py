from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "local"
    frontend_origin: str = "http://127.0.0.1:5173"
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "game_readonly"
    mysql_password: str = ""
    mysql_database: str = "game_customer_service"
    mysql_enabled: bool = False
    mysql_players_table: str = "players"
    llm_enabled: bool = False
    llm_provider: str = "deepseek"
    llm_default_provider: str = "deepseek"
    llm_allowed_providers: str = "deepseek,qwen"
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-v4-flash"
    llm_timeout_seconds: float = 20
    deepseek_base_url: str = ""
    deepseek_api_key: str = ""
    deepseek_model: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_api_key: str = ""
    qwen_model: str = "qwen-plus"
    amap_mcp_enabled: bool = False
    amap_mcp_url: str = ""
    amap_mcp_timeout_seconds: float = 15
    knowledge_base_dir: str = "../knowledge_base"
    vector_store_dir: str = "./data/vector_store"
    prompt_decision_version: str = "v1.0"
    prompt_followup_decision_version: str = "v1.0"
    prompt_final_reply_version: str = "v1.0"
    log_dir: str = "../logs"
    log_level: str = "INFO"
    log_max_bytes: int = 5_242_880
    log_backup_count: int = 5
    agent_audit_log_enabled: bool = True

    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
