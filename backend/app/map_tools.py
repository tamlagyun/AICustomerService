from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import json
import re
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx

from app.config import get_settings


class MapToolStatus(StrEnum):
    DISABLED = "disabled"
    FOUND = "found"
    NOT_FOUND = "not_found"
    INVALID_REQUEST = "invalid_request"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MapToolResult:
    status: MapToolStatus
    summary: str
    data: dict[str, object] | None = None


class AmapMcpClientProtocol(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, object]) -> dict:
        raise NotImplementedError


class AmapStreamableHttpMcpClient:
    def __init__(
        self,
        *,
        mcp_url: str,
        timeout_seconds: float,
        protocol_version: str = "2025-06-18",
    ) -> None:
        self.mcp_url = mcp_url
        self.timeout_seconds = timeout_seconds
        self.protocol_version = protocol_version
        self._session_id: str | None = None
        self._next_request_id = 1

    async def call_tool(self, name: str, arguments: dict[str, object]) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            await self._initialize(client)
            await self._notify_initialized(client)
            payload = self._request(
                "tools/call",
                {"name": name, "arguments": arguments},
            )
            response = await self._post_rpc(client, payload)
            result = response.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("MCP tools/call returned an invalid result")
            return result

    async def _initialize(self, client: httpx.AsyncClient) -> None:
        payload = self._request(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": "customer-service-agent",
                    "version": "0.1.0",
                },
            },
        )
        await self._post_rpc(client, payload)

    async def _notify_initialized(self, client: httpx.AsyncClient) -> None:
        payload = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        await self._post_rpc(client, payload)

    def _request(self, method: str, params: dict[str, object]) -> dict[str, object]:
        request_id = self._next_request_id
        self._next_request_id += 1
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

    async def _post_rpc(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, object],
    ) -> dict[str, Any]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": self.protocol_version,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        response = await client.post(self.mcp_url, json=payload, headers=headers)
        response.raise_for_status()
        session_id = response.headers.get("mcp-session-id") or response.headers.get("Mcp-Session-Id")
        if session_id:
            self._session_id = session_id

        if not response.content:
            return {}

        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            message = _parse_sse_json(response.text)
        else:
            message = response.json()

        if not isinstance(message, dict):
            raise RuntimeError("MCP returned an invalid JSON-RPC message")
        if "error" in message:
            raise RuntimeError(json.dumps(message["error"], ensure_ascii=False))
        return message


class AmapMapTools:
    def __init__(
        self,
        *,
        enabled: bool,
        mcp_url: str,
        timeout_seconds: float,
        mcp_client_factory: Callable[[], AmapMcpClientProtocol] | None = None,
    ) -> None:
        self.enabled = enabled
        self.mcp_url = mcp_url.strip()
        self.timeout_seconds = timeout_seconds
        self.mcp_client_factory = mcp_client_factory or self._build_client

    async def search_place(
        self,
        keywords: str | None,
        *,
        city: str | None = None,
        types: str | None = None,
    ) -> MapToolResult:
        cleaned_keywords = _clean_optional_text(keywords)
        if not cleaned_keywords:
            return MapToolResult(
                status=MapToolStatus.INVALID_REQUEST,
                summary="请提供要查询的地点关键词，例如网吧、酒店、餐厅或具体地名。",
            )

        arguments = {"keywords": cleaned_keywords}
        _add_optional_argument(arguments, "city", city)
        _add_optional_argument(arguments, "types", types)
        return await self._call_tool("maps_text_search", arguments, "高德地图查询结果")

    async def geocode(self, address: str | None, *, city: str | None = None) -> MapToolResult:
        cleaned_address = _clean_optional_text(address)
        if not cleaned_address:
            return MapToolResult(
                status=MapToolStatus.INVALID_REQUEST,
                summary="请提供要解析的地址或地点名称。",
            )

        arguments = {"address": cleaned_address}
        _add_optional_argument(arguments, "city", city)
        return await self._call_tool("maps_geo", arguments, "高德地图地址解析结果")

    async def route(
        self,
        *,
        origin: str | None,
        destination: str | None,
        mode: str | None = None,
        city: str | None = None,
        cityd: str | None = None,
    ) -> MapToolResult:
        cleaned_origin = _clean_optional_text(origin)
        cleaned_destination = _clean_optional_text(destination)
        if not cleaned_origin or not cleaned_destination:
            return MapToolResult(
                status=MapToolStatus.INVALID_REQUEST,
                summary="请提供路线查询的起点和终点。",
            )

        if not self.enabled or not self.mcp_url:
            return _disabled_result()

        normalized_mode = (_clean_optional_text(mode) or "driving").lower()
        tool_name = {
            "walking": "maps_direction_walking",
            "walk": "maps_direction_walking",
            "driving": "maps_direction_driving",
            "drive": "maps_direction_driving",
            "bicycling": "maps_bicycling",
            "cycling": "maps_bicycling",
            "bike": "maps_bicycling",
            "transit": "maps_direction_transit_integrated",
            "bus": "maps_direction_transit_integrated",
        }.get(normalized_mode, "maps_direction_driving")

        try:
            client = self.mcp_client_factory()
            origin_location = await self._resolve_location(client, cleaned_origin, city)
            destination_location = await self._resolve_location(client, cleaned_destination, cityd or city)
            if not origin_location or not destination_location:
                return MapToolResult(
                    status=MapToolStatus.NOT_FOUND,
                    summary="无法解析路线起点或终点，请补充更准确的地址或经纬度。",
                )

            arguments = {
                "origin": origin_location,
                "destination": destination_location,
            }
            if tool_name == "maps_direction_transit_integrated":
                _add_optional_argument(arguments, "city", city)
                _add_optional_argument(arguments, "cityd", cityd)

            result = await client.call_tool(tool_name, arguments)
        except Exception:
            return MapToolResult(
                status=MapToolStatus.UNAVAILABLE,
                summary="地图查询暂时不可用，请稍后再试或转人工客服。",
            )

        return _map_result_from_mcp_result(
            tool_name,
            arguments,
            result,
            "高德地图路线查询结果",
        )

    async def weather(self, city: str | None) -> MapToolResult:
        cleaned_city = _clean_optional_text(city)
        if not cleaned_city:
            return MapToolResult(
                status=MapToolStatus.INVALID_REQUEST,
                summary="请提供要查询天气的城市名称或 adcode。",
            )

        return await self._call_tool("maps_weather", {"city": cleaned_city}, "高德地图天气查询结果")

    async def navigation(
        self,
        *,
        destination: str | None,
        destination_name: str | None = None,
        origin: str | None = None,
        origin_name: str | None = None,
        mode: str | None = None,
        city: str | None = None,
    ) -> MapToolResult:
        cleaned_destination = _clean_optional_text(destination)
        if not cleaned_destination:
            return MapToolResult(
                status=MapToolStatus.INVALID_REQUEST,
                summary="请提供导航目的地。",
            )

        if not self.enabled or not self.mcp_url:
            return _disabled_result()

        try:
            client = self.mcp_client_factory()
            destination_location = await self._resolve_location(client, cleaned_destination, city)
            if not destination_location:
                return MapToolResult(
                    status=MapToolStatus.NOT_FOUND,
                    summary="无法解析导航目的地，请补充更准确的地址或经纬度。",
                )

            origin_location = None
            cleaned_origin = _clean_optional_text(origin)
            if cleaned_origin:
                origin_location = await self._resolve_location(client, cleaned_origin, city)
                if not origin_location:
                    return MapToolResult(
                        status=MapToolStatus.NOT_FOUND,
                        summary="无法解析导航起点，请补充更准确的地址或经纬度。",
                    )
        except Exception:
            return MapToolResult(
                status=MapToolStatus.UNAVAILABLE,
                summary="地图查询暂时不可用，请稍后再试或转人工客服。",
            )

        url = _build_navigation_url(
            destination_location=destination_location,
            destination_name=_clean_optional_text(destination_name) or cleaned_destination,
            origin_location=origin_location,
            origin_name=_clean_optional_text(origin_name) or cleaned_origin,
            mode=mode,
        )
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary=f"高德地图导航链接：{url}",
            data={
                "tool": "amap_navigation_uri",
                "url": url,
                "destination": destination_location,
                "origin": origin_location,
                "mode": _uri_mode(mode),
            },
        )

    async def _resolve_location(
        self,
        client: AmapMcpClientProtocol,
        value: str,
        city: str | None,
    ) -> str | None:
        if _is_location(value):
            return value

        arguments = {
            "address": value,
        }
        _add_optional_argument(arguments, "city", city)
        result = await client.call_tool("maps_geo", arguments)
        if result.get("isError"):
            return None

        return _extract_location(result)

    async def _call_tool(
        self,
        tool_name: str,
        arguments: dict[str, object],
        summary_prefix: str,
    ) -> MapToolResult:
        if not self.enabled or not self.mcp_url:
            return _disabled_result()

        try:
            result = await self.mcp_client_factory().call_tool(tool_name, arguments)
        except Exception:
            return MapToolResult(
                status=MapToolStatus.UNAVAILABLE,
                summary="地图查询暂时不可用，请稍后再试或转人工客服。",
            )

        return _map_result_from_mcp_result(tool_name, arguments, result, summary_prefix)

    def _build_client(self) -> AmapStreamableHttpMcpClient:
        return AmapStreamableHttpMcpClient(
            mcp_url=self.mcp_url,
            timeout_seconds=self.timeout_seconds,
        )


def build_map_tools() -> AmapMapTools:
    settings = get_settings()
    return AmapMapTools(
        enabled=settings.amap_mcp_enabled,
        mcp_url=settings.amap_mcp_url,
        timeout_seconds=settings.amap_mcp_timeout_seconds,
    )


def _clean_optional_text(value: object) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def _add_optional_argument(arguments: dict[str, object], key: str, value: object) -> None:
    cleaned = _clean_optional_text(value)
    if cleaned:
        arguments[key] = cleaned


def _extract_mcp_text(result: dict[str, Any]) -> str:
    content = result.get("content")
    texts: list[str] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())

    if texts:
        return "\n".join(texts)

    structured_content = result.get("structuredContent")
    if structured_content:
        return json.dumps(structured_content, ensure_ascii=False)

    return ""


def _map_result_from_mcp_result(
    tool_name: str,
    arguments: dict[str, object],
    result: dict[str, Any],
    summary_prefix: str,
) -> MapToolResult:
    text = _extract_mcp_text(result)
    if result.get("isError"):
        return MapToolResult(
            status=MapToolStatus.UNAVAILABLE,
            summary=text or "地图查询暂时不可用，请稍后再试或转人工客服。",
            data={"tool": tool_name, "arguments": arguments, "result": result},
        )
    if not text:
        return MapToolResult(
            status=MapToolStatus.NOT_FOUND,
            summary="没有查询到匹配的地图结果，请补充城市、地址或更准确的地点名称。",
            data={"tool": tool_name, "arguments": arguments, "result": result},
        )

    return MapToolResult(
        status=MapToolStatus.FOUND,
        summary=f"{summary_prefix}：{text}",
        data={"tool": tool_name, "arguments": arguments, "result": result},
    )


def _disabled_result() -> MapToolResult:
    return MapToolResult(
        status=MapToolStatus.DISABLED,
        summary="地图查询功能尚未启用，请配置高德地图 MCP 后再查询。",
    )


def _is_location(value: str) -> bool:
    return bool(re.fullmatch(r"\s*-?\d{1,3}(?:\.\d+)?\s*,\s*-?\d{1,2}(?:\.\d+)?\s*", value))


def _extract_location(result: dict[str, Any]) -> str | None:
    structured_content = result.get("structuredContent")
    location = _extract_location_from_object(structured_content)
    if location:
        return location

    text_location = re.search(r"-?\d{1,3}(?:\.\d+)?\s*,\s*-?\d{1,2}(?:\.\d+)?", _extract_mcp_text(result))
    if text_location:
        return text_location.group(0).replace(" ", "")

    return None


def _extract_location_from_object(value: object) -> str | None:
    if isinstance(value, dict):
        raw_location = value.get("location")
        if isinstance(raw_location, str) and _is_location(raw_location):
            return raw_location.strip().replace(" ", "")
        for nested in value.values():
            location = _extract_location_from_object(nested)
            if location:
                return location
    if isinstance(value, list):
        for item in value:
            location = _extract_location_from_object(item)
            if location:
                return location
    return None


def _build_navigation_url(
    *,
    destination_location: str,
    destination_name: str,
    origin_location: str | None,
    origin_name: str | None,
    mode: str | None,
) -> str:
    query = {
        "to": _format_uri_position(destination_location, destination_name),
        "mode": _uri_mode(mode),
        "policy": "0",
        "src": "customer-service-agent",
        "callnative": "1",
    }
    if origin_location:
        query["from"] = _format_uri_position(origin_location, origin_name or "起点")

    return f"https://uri.amap.com/navigation?{urlencode(query)}"


def _format_uri_position(location: str, name: str) -> str:
    return f"{location},{name}"


def _uri_mode(mode: str | None) -> str:
    normalized = (_clean_optional_text(mode) or "driving").lower()
    return {
        "driving": "car",
        "drive": "car",
        "car": "car",
        "walking": "walk",
        "walk": "walk",
        "bicycling": "ride",
        "cycling": "ride",
        "bike": "ride",
        "ride": "ride",
        "transit": "bus",
        "bus": "bus",
    }.get(normalized, "car")


def _parse_sse_json(raw_text: str) -> dict[str, Any]:
    for event in raw_text.split("\n\n"):
        data_lines = []
        for line in event.splitlines():
            if line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
        if not data_lines:
            continue

        data = "\n".join(data_lines)
        if data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise RuntimeError("MCP SSE response did not contain a JSON payload")
