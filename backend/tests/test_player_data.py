from app.player_data import PlayerDataStatus, PlayerDataTools


class FakeCursor:
    def __init__(
        self,
        row: dict[str, object] | None = None,
        rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.row = row
        self.rows = rows or []
        self.executed_sql = ""
        self.executed_params: tuple[object, ...] = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        self.executed_sql = sql
        self.executed_params = params

    def fetchone(self) -> dict[str, object] | None:
        return self.row

    def fetchall(self) -> list[dict[str, object]]:
        return self.rows


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_instance = cursor
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def close(self) -> None:
        self.closed = True


def test_get_player_profile_uses_parameterized_query() -> None:
    cursor = FakeCursor(
        {
            "player_id": "player-1",
            "nickname": "测试玩家",
            "level": 30,
            "server_name": "一区",
            "status": "active",
            "desc": "喜欢探索和社交，偏好团队协作。",
        }
    )
    connection = FakeConnection(cursor)
    tools = PlayerDataTools(
        enabled=True,
        players_table="players",
        connection_factory=lambda: connection,
    )

    result = tools.get_player_profile("player-1")

    assert result.status == PlayerDataStatus.FOUND
    assert "测试玩家" in result.summary
    assert "喜欢探索和社交" in result.summary
    assert result.data == {
        "player_id": "player-1",
        "nickname": "测试玩家",
        "level": 30,
        "server_name": "一区",
        "status": "active",
        "desc": "喜欢探索和社交，偏好团队协作。",
    }
    assert "`desc`" in cursor.executed_sql
    assert "player_id = %s" in cursor.executed_sql
    assert cursor.executed_params == ("player-1",)
    assert connection.closed is True


def test_get_player_profile_returns_disabled_without_connecting() -> None:
    did_connect = False

    def connection_factory():
        nonlocal did_connect
        did_connect = True
        raise AssertionError("should not connect when disabled")

    tools = PlayerDataTools(
        enabled=False,
        players_table="players",
        connection_factory=connection_factory,
    )

    result = tools.get_player_profile("player-1")

    assert result.status == PlayerDataStatus.DISABLED
    assert "尚未启用" in result.summary
    assert did_connect is False


def test_get_player_profile_returns_unavailable_on_connection_error() -> None:
    tools = PlayerDataTools(
        enabled=True,
        players_table="players",
        connection_factory=lambda: (_ for _ in ()).throw(RuntimeError("connection failed")),
    )

    result = tools.get_player_profile("player-1")

    assert result.status == PlayerDataStatus.UNAVAILABLE
    assert "暂时无法查询" in result.summary


def test_get_players_returns_limited_structured_rows() -> None:
    cursor = FakeCursor(
        rows=[
            {
                "player_id": "1",
                "nickname": "玩家一",
                "level": 10,
                "server_name": "一区",
                "status": "active",
                "desc": "进攻型玩家",
            },
            {
                "player_id": "2",
                "nickname": "玩家二",
                "level": 20,
                "server_name": "二区",
                "status": "active",
                "desc": "探索型玩家",
            },
        ]
    )
    connection = FakeConnection(cursor)
    tools = PlayerDataTools(
        enabled=True,
        players_table="players",
        connection_factory=lambda: connection,
    )

    result = tools.get_players()

    assert result.status == PlayerDataStatus.FOUND
    assert "共查询到 2 条玩家数据" in result.summary
    assert result.data == {
        "limit": 100,
        "players": [
            {
                "player_id": "1",
                "nickname": "玩家一",
                "level": 10,
                "server_name": "一区",
                "status": "active",
                "desc": "进攻型玩家",
            },
            {
                "player_id": "2",
                "nickname": "玩家二",
                "level": 20,
                "server_name": "二区",
                "status": "active",
                "desc": "探索型玩家",
            },
        ],
    }
    assert "SELECT player_id, nickname, level, server_name, status, `desc`" in cursor.executed_sql
    assert "ORDER BY player_id" in cursor.executed_sql
    assert cursor.executed_params == (100,)
    assert connection.closed is True


def test_get_players_clamps_limit_to_maximum() -> None:
    cursor = FakeCursor(rows=[])
    connection = FakeConnection(cursor)
    tools = PlayerDataTools(
        enabled=True,
        players_table="players",
        connection_factory=lambda: connection,
    )

    result = tools.get_players(limit=5000)

    assert result.status == PlayerDataStatus.NOT_FOUND
    assert cursor.executed_params == (1000,)


def test_get_players_returns_disabled_without_connecting() -> None:
    did_connect = False

    def connection_factory():
        nonlocal did_connect
        did_connect = True
        raise AssertionError("should not connect when disabled")

    tools = PlayerDataTools(
        enabled=False,
        players_table="players",
        connection_factory=connection_factory,
    )

    result = tools.get_players()

    assert result.status == PlayerDataStatus.DISABLED
    assert "尚未启用" in result.summary
    assert did_connect is False
