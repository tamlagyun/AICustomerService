def __getattr__(name: str):
    if name == "run_customer_service_agent":
        from app.agent.customer_service import run_customer_service_agent

        return run_customer_service_agent
    if name == "stream_customer_service_agent":
        from app.agent.customer_service import stream_customer_service_agent

        return stream_customer_service_agent
    raise AttributeError(name)

__all__ = ["run_customer_service_agent", "stream_customer_service_agent"]
