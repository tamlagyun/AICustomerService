from app.map_tools import AmapMapTools, MapToolResult, MapToolStatus


class FakeMcpClient:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_tool(self, name: str, arguments: dict[str, object]) -> dict:
        self.calls.append((name, arguments))
        return self.result


class SequentialFakeMcpClient:
    def __init__(self, results: list[dict]) -> None:
        self.results = results
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_tool(self, name: str, arguments: dict[str, object]) -> dict:
        self.calls.append((name, arguments))
        return self.results.pop(0)


async def test_search_place_returns_disabled_without_calling_mcp() -> None:
    did_build_client = False

    def client_factory():
        nonlocal did_build_client
        did_build_client = True
        raise AssertionError("should not build MCP client when disabled")

    tools = AmapMapTools(
        enabled=False,
        mcp_url="",
        timeout_seconds=15,
        mcp_client_factory=client_factory,
    )

    result = await tools.search_place("网吧", city="北京")

    assert result.status == MapToolStatus.DISABLED
    assert "地图查询功能尚未启用" in result.summary
    assert did_build_client is False


async def test_search_place_calls_amap_text_search_with_controlled_arguments() -> None:
    fake_client = FakeMcpClient(
        {
            "content": [
                {
                    "type": "text",
                    "text": "海淀区网吧：地址 北京市海淀区中关村大街 1 号",
                }
            ],
            "isError": False,
        }
    )
    tools = AmapMapTools(
        enabled=True,
        mcp_url="https://mcp.amap.com/mcp?key=test",
        timeout_seconds=15,
        mcp_client_factory=lambda: fake_client,
    )

    result = await tools.search_place(" 网吧 ", city=" 北京 ", types="娱乐场所")

    assert result.status == MapToolStatus.FOUND
    assert "海淀区网吧" in result.summary
    assert fake_client.calls == [
        (
            "maps_text_search",
            {"keywords": "网吧", "city": "北京", "types": "娱乐场所"},
        )
    ]
    assert result.data == {
        "tool": "maps_text_search",
        "arguments": {"keywords": "网吧", "city": "北京", "types": "娱乐场所"},
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": "海淀区网吧：地址 北京市海淀区中关村大街 1 号",
                }
            ],
            "isError": False,
        },
    }


async def test_geocode_calls_amap_geo() -> None:
    fake_client = FakeMcpClient(
        {"content": [{"type": "text", "text": "西湖 经纬度 120.148872,30.246026"}]}
    )
    tools = AmapMapTools(
        enabled=True,
        mcp_url="https://mcp.amap.com/mcp?key=test",
        timeout_seconds=15,
        mcp_client_factory=lambda: fake_client,
    )

    result = await tools.geocode("杭州西湖", city="杭州")

    assert result.status == MapToolStatus.FOUND
    assert fake_client.calls == [
        ("maps_geo", {"address": "杭州西湖", "city": "杭州"}),
    ]


async def test_route_uses_mode_specific_tool() -> None:
    fake_client = FakeMcpClient(
        {"content": [{"type": "text", "text": "驾车距离 15 公里，预计 35 分钟"}]}
    )
    tools = AmapMapTools(
        enabled=True,
        mcp_url="https://mcp.amap.com/mcp?key=test",
        timeout_seconds=15,
        mcp_client_factory=lambda: fake_client,
    )

    result = await tools.route(
        origin="116.397428,39.90923",
        destination="116.481488,39.990464",
        mode="driving",
    )

    assert result.status == MapToolStatus.FOUND
    assert "预计 35 分钟" in result.summary
    assert fake_client.calls == [
        (
            "maps_direction_driving",
            {
                "origin": "116.397428,39.90923",
                "destination": "116.481488,39.990464",
            },
        )
    ]


async def test_route_geocodes_address_arguments_before_direction_call() -> None:
    fake_client = SequentialFakeMcpClient(
        [
            {"structuredContent": {"location": "116.427,39.903"}},
            {"structuredContent": {"geocodes": [{"location": "116.397,39.908"}]}},
            {"content": [{"type": "text", "text": "驾车距离 4 公里，预计 18 分钟"}]},
        ]
    )
    tools = AmapMapTools(
        enabled=True,
        mcp_url="https://mcp.amap.com/mcp?key=test",
        timeout_seconds=15,
        mcp_client_factory=lambda: fake_client,
    )

    result = await tools.route(
        origin="北京站",
        destination="天安门",
        mode="driving",
        city="北京",
    )

    assert result.status == MapToolStatus.FOUND
    assert "预计 18 分钟" in result.summary
    assert fake_client.calls == [
        ("maps_geo", {"address": "北京站", "city": "北京"}),
        ("maps_geo", {"address": "天安门", "city": "北京"}),
        (
            "maps_direction_driving",
            {"origin": "116.427,39.903", "destination": "116.397,39.908"},
        ),
    ]


async def test_weather_calls_amap_weather() -> None:
    fake_client = FakeMcpClient(
        {"content": [{"type": "text", "text": "北京今天晴，气温 26 到 34 度。"}]}
    )
    tools = AmapMapTools(
        enabled=True,
        mcp_url="https://mcp.amap.com/mcp?key=test",
        timeout_seconds=15,
        mcp_client_factory=lambda: fake_client,
    )

    result = await tools.weather("北京")

    assert result.status == MapToolStatus.FOUND
    assert "北京今天晴" in result.summary
    assert fake_client.calls == [("maps_weather", {"city": "北京"})]


async def test_navigation_builds_amap_uri_after_geocoding_destination() -> None:
    fake_client = FakeMcpClient({"structuredContent": {"location": "116.397,39.908"}})
    tools = AmapMapTools(
        enabled=True,
        mcp_url="https://mcp.amap.com/mcp?key=test",
        timeout_seconds=15,
        mcp_client_factory=lambda: fake_client,
    )

    result = await tools.navigation(
        destination="天安门",
        destination_name="天安门",
        mode="walking",
        city="北京",
    )

    assert result.status == MapToolStatus.FOUND
    assert "高德地图导航链接" in result.summary
    assert fake_client.calls == [("maps_geo", {"address": "天安门", "city": "北京"})]
    assert result.data is not None
    assert result.data["tool"] == "amap_navigation_uri"
    assert "https://uri.amap.com/navigation" in str(result.data["url"])
    assert "mode=walk" in str(result.data["url"])


async def test_search_place_returns_unavailable_when_mcp_call_fails() -> None:
    class FailingMcpClient:
        async def call_tool(self, name: str, arguments: dict[str, object]) -> dict:
            raise RuntimeError("network failed")

    tools = AmapMapTools(
        enabled=True,
        mcp_url="https://mcp.amap.com/mcp?key=test",
        timeout_seconds=15,
        mcp_client_factory=FailingMcpClient,
    )

    result = await tools.search_place("网吧")

    assert result == MapToolResult(
        status=MapToolStatus.UNAVAILABLE,
        summary="地图查询暂时不可用，请稍后再试或转人工客服。",
        data=None,
    )
