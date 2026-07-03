from pathlib import Path

from app.knowledge_base import KnowledgeBaseSearch


def test_search_finds_markdown_section(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "recharge.md").write_text(
        "# 充值问题\n\n## 充值未到账怎么办\n\n请提供订单号、充值时间、服务器和角色 ID。",
        encoding="utf-8",
    )

    search = KnowledgeBaseSearch(kb_dir)

    results = search.search("充值不到账")

    assert len(results) == 1
    assert results[0].title == "充值未到账怎么办"
    assert "订单号" in results[0].content
    assert results[0].reference == "recharge.md#充值未到账怎么办"


def test_search_finds_html_heading_content(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "account.html").write_text(
        "<html><body><h1>账号问题</h1><h2>账号被封禁怎么办</h2><p>封禁申诉需要转人工处理。</p></body></html>",
        encoding="utf-8",
    )

    search = KnowledgeBaseSearch(kb_dir)

    results = search.search("封禁申诉")

    assert len(results) == 1
    assert results[0].title == "账号被封禁怎么办"
    assert "转人工" in results[0].content
    assert results[0].reference == "account.html#账号被封禁怎么办"


def test_search_finds_markdown_heading_without_space(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "identity.md").write_text(
        "# 示例\n\n##我不认识你怎么办？\n\n自己照照镜子就认识了。",
        encoding="utf-8",
    )

    search = KnowledgeBaseSearch(kb_dir)

    results = search.search("我不认识你怎么办")

    assert len(results) == 1
    assert results[0].title == "我不认识你怎么办？"
    assert "照照镜子" in results[0].content


def test_search_finds_short_exact_markdown_heading_without_space(tmp_path: Path) -> None:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "chat.md").write_text(
        "# 示例\n\n##你吃过屎吗？\n\n你吃过呀，那你告诉我屎是什么味道?",
        encoding="utf-8",
    )

    search = KnowledgeBaseSearch(kb_dir)

    results = search.search("你吃过屎吗？")

    assert len(results) == 1
    assert results[0].title == "你吃过屎吗？"
    assert "什么味道" in results[0].content
