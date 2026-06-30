from app.llm import extract_stream_delta


def test_extract_stream_delta_reads_openai_compatible_content() -> None:
    line = 'data: {"choices":[{"delta":{"content":"你好"}}]}'

    assert extract_stream_delta(line) == "你好"


def test_extract_stream_delta_ignores_done_marker() -> None:
    assert extract_stream_delta("data: [DONE]") is None
