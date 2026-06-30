from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from app.config import get_settings


class PlayerDataStatus(StrEnum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PlayerDataResult:
    status: PlayerDataStatus
    summary: str
    data: dict[str, Any] | None = None


class PlayerDataTools:
    DEFAULT_PLAYERS_LIMIT = 100
    MAX_PLAYERS_LIMIT = 1000

    def __init__(
        self,
        *,
        enabled: bool,
        players_table: str,
        connection_factory: Callable[[], Any],
    ) -> None:
        self.enabled = enabled
        self.players_table = players_table
        self.connection_factory = connection_factory

    def get_player_profile(self, player_id: str | None) -> PlayerDataResult:
        if not player_id:
            return PlayerDataResult(
                status=PlayerDataStatus.NOT_FOUND,
                summary="请先提供玩家 ID，才能查询玩家资料。",
            )

        if not self.enabled:
            return PlayerDataResult(
                status=PlayerDataStatus.DISABLED,
                summary="玩家数据查询尚未启用，请配置 MySQL 后再查询。",
            )

        connection = None
        try:
            connection = self.connection_factory()
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT player_id, nickname, level, server_name, status, `desc`
                    FROM `{self.players_table}`
                    WHERE player_id = %s
                    LIMIT 1
                    """,
                    (player_id,),
                )
                row = cursor.fetchone()
        except Exception:
            return PlayerDataResult(
                status=PlayerDataStatus.UNAVAILABLE,
                summary="玩家数据暂时无法查询，请稍后重试或转人工客服处理。",
            )
        finally:
            if connection is not None:
                connection.close()

        if not row:
            return PlayerDataResult(
                status=PlayerDataStatus.NOT_FOUND,
                summary="没有查询到该玩家资料，请确认玩家 ID 是否正确。",
            )

        return PlayerDataResult(
            status=PlayerDataStatus.FOUND,
            summary=(
                f"玩家资料：玩家 ID {row.get('player_id')}，昵称 {row.get('nickname')}，"
                f"等级 {row.get('level')}，服务器 {row.get('server_name')}，状态 {row.get('status')}，"
                f"个性描述 {row.get('desc') or '暂无'}。"
            ),
            data={
                "player_id": row.get("player_id"),
                "nickname": row.get("nickname"),
                "level": row.get("level"),
                "server_name": row.get("server_name"),
                "status": row.get("status"),
                "desc": row.get("desc"),
            },
        )

    def get_players(self, limit: int = DEFAULT_PLAYERS_LIMIT) -> PlayerDataResult:
        if not self.enabled:
            return PlayerDataResult(
                status=PlayerDataStatus.DISABLED,
                summary="玩家数据查询尚未启用，请配置 MySQL 后再查询。",
            )

        safe_limit = self._safe_players_limit(limit)
        connection = None
        try:
            connection = self.connection_factory()
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT player_id, nickname, level, server_name, status, `desc`
                    FROM `{self.players_table}`
                    ORDER BY player_id
                    LIMIT %s
                    """,
                    (safe_limit,),
                )
                rows = cursor.fetchall()
        except Exception:
            return PlayerDataResult(
                status=PlayerDataStatus.UNAVAILABLE,
                summary="玩家数据暂时无法查询，请稍后重试或转人工客服处理。",
            )
        finally:
            if connection is not None:
                connection.close()

        players = [_player_row_to_data(row) for row in rows]
        if not players:
            return PlayerDataResult(
                status=PlayerDataStatus.NOT_FOUND,
                summary="没有查询到玩家数据。",
                data={"limit": safe_limit, "players": []},
            )

        return PlayerDataResult(
            status=PlayerDataStatus.FOUND,
            summary=f"共查询到 {len(players)} 条玩家数据，当前返回上限 {safe_limit} 条。",
            data={"limit": safe_limit, "players": players},
        )

    def _safe_players_limit(self, limit: int) -> int:
        if not isinstance(limit, int) or limit <= 0:
            return self.DEFAULT_PLAYERS_LIMIT
        return min(limit, self.MAX_PLAYERS_LIMIT)


def _player_row_to_data(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "player_id": row.get("player_id"),
        "nickname": row.get("nickname"),
        "level": row.get("level"),
        "server_name": row.get("server_name"),
        "status": row.get("status"),
        "desc": row.get("desc"),
    }


def build_player_data_tools() -> PlayerDataTools:
    settings = get_settings()
    return PlayerDataTools(
        enabled=settings.mysql_enabled,
        players_table=settings.mysql_players_table,
        connection_factory=lambda: pymysql.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            database=settings.mysql_database,
            charset="utf8mb4",
            cursorclass=DictCursor,
            connect_timeout=3,
            read_timeout=5,
            write_timeout=5,
        ),
    )
