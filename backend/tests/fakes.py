from app.agent.decision import AgentAction, AgentDecision
from app.avatar_generation import AvatarGenerationResult, AvatarGenerationStatus
from app.llm import LLMResponse
from app.map_tools import MapToolResult, MapToolStatus
from app.player_data import PlayerDataResult, PlayerDataStatus


class FakeLLMClient:
    def __init__(self, *, decision: AgentDecision, final_reply: str) -> None:
        self.decision = decision
        self.final_reply = final_reply
        self.decision_messages: list[dict[str, str]] | None = None
        self.final_messages: list[dict[str, str]] | None = None

    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        self.decision_messages = messages
        return self.decision

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        self.final_messages = messages
        return LLMResponse(content=self.final_reply)


class MultiStepFakeLLMClient:
    def __init__(self, *, decisions: list[AgentDecision], final_reply: str) -> None:
        self.decisions = decisions
        self.final_reply = final_reply
        self.decision_messages: list[list[dict[str, str]]] = []
        self.final_messages: list[dict[str, str]] | None = None

    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        self.decision_messages.append(messages)
        return self.decisions.pop(0)

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        self.final_messages = messages
        return LLMResponse(content=self.final_reply)


class PlannerFakeLLMClient:
    def __init__(
        self,
        *,
        plan_reply: str,
        final_reply: str,
        fallback_decision: AgentDecision | None = None,
    ) -> None:
        self.plan_reply = plan_reply
        self.final_reply = final_reply
        self.fallback_decision = fallback_decision
        self.decision_messages: list[list[dict[str, str]]] = []
        self.generate_messages: list[list[dict[str, str]]] = []

    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        self.decision_messages.append(messages)
        return self.fallback_decision or AgentDecision(
            action=AgentAction.FALLBACK,
            reason="Planner test should not use decide_action",
        )

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        self.generate_messages.append(messages)
        if len(self.generate_messages) == 1:
            return LLMResponse(content=self.plan_reply)
        return LLMResponse(content=self.final_reply)


class FailingDecisionLLMClient:
    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        raise TimeoutError("llm timeout")

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        return LLMResponse(content="不应调用最终生成")


class FailingFinalReplyLLMClient:
    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        return AgentDecision(
            action=AgentAction.MYSQL_PLAYERS_LIST,
            reason="查询所有玩家资料",
        )

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        raise TimeoutError("llm final timeout")


class FailingStreamLLMClient:
    async def decide_action(self, messages: list[dict[str, str]]) -> AgentDecision:
        return AgentDecision(
            action=AgentAction.MYSQL_PLAYERS_LIST,
            reason="查询所有玩家资料",
        )

    async def generate_reply(self, messages: list[dict[str, str]]) -> LLMResponse:
        return LLMResponse(content="流式失败后使用普通生成回复")

    async def stream_reply(self, messages: list[dict[str, str]]):
        raise TimeoutError("llm stream timeout")
        yield ""


class FakePlayerDataTools:
    def __init__(self) -> None:
        self.requested_player_id: str | None = None
        self.requested_limit: int | None = None

    def get_player_profile(self, player_id: str | None) -> PlayerDataResult:
        self.requested_player_id = player_id
        return PlayerDataResult(
            status=PlayerDataStatus.FOUND,
            summary="玩家资料：玩家 ID 1，昵称 ai大名，等级 2，服务器 1服，状态 1，个性描述 喜欢研究机制。",
            data={
                "player_id": "1",
                "nickname": "ai大名",
                "level": 2,
                "server_name": "1服",
                "status": "1",
                "desc": "喜欢研究机制。",
            },
        )

    def get_players(self, limit: int = 100) -> PlayerDataResult:
        self.requested_limit = limit
        return PlayerDataResult(
            status=PlayerDataStatus.FOUND,
            summary="共查询到 2 条玩家数据，当前返回上限 100 条。",
            data={
                "limit": limit,
                "players": [
                    {
                        "player_id": "1",
                        "nickname": "ai大名",
                        "level": 2,
                        "server_name": "1服",
                        "status": "1",
                        "desc": "进攻型玩家",
                    },
                    {
                        "player_id": "2",
                        "nickname": "beta",
                        "level": 8,
                        "server_name": "1服",
                        "status": "1",
                        "desc": "探索型玩家",
                    },
                ],
            },
        )


class FakeAvatarGenerator:
    def __init__(self) -> None:
        self.profile: dict | None = None
        self.session_id: str | None = None

    def generate_player_avatar(
        self,
        profile: dict,
        *,
        session_id: str,
    ) -> AvatarGenerationResult:
        self.profile = profile
        self.session_id = session_id
        return AvatarGenerationResult(
            status=AvatarGenerationStatus.GENERATED,
            summary="已生成本地 PNG 头像：/generated/avatars/player-1.png",
            url="/generated/avatars/player-1.png",
            alt="ai大名 的个性头像",
            data={"style": "策略研究型"},
        )


class FakeMapTools:
    def __init__(self) -> None:
        self.place_query: dict[str, object] | None = None

    async def search_place(
        self,
        keywords: str | None,
        *,
        city: str | None = None,
        types: str | None = None,
    ) -> MapToolResult:
        self.place_query = {"keywords": keywords, "city": city, "types": types}
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="高德地图查询结果：中关村电竞网吧，地址 北京市海淀区中关村大街 1 号。",
            data={
                "tool": "maps_text_search",
                "arguments": {"keywords": keywords, "city": city, "types": types},
                "result": {
                    "structuredContent": {
                        "pois": [
                            {
                                "name": "中关村电竞网吧",
                                "address": "北京市海淀区中关村大街 1 号",
                                "type": "休闲娱乐",
                                "distance": "800",
                            }
                        ]
                    },
                    "content": [
                        {
                            "type": "text",
                            "text": "中关村电竞网吧，地址 北京市海淀区中关村大街 1 号。",
                        }
                    ],
                },
            },
        )

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
        self.place_query = {
            "destination": destination,
            "destination_name": destination_name,
            "origin": origin,
            "origin_name": origin_name,
            "mode": mode,
            "city": city,
        }
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="高德地图导航链接：https://uri.amap.com/navigation?to=116.397,39.908,天安门&mode=walk",
            data={
                "tool": "amap_navigation_uri",
                "url": "https://uri.amap.com/navigation?to=116.397,39.908,天安门&mode=walk",
            },
        )

    async def weather(self, city: str | None) -> MapToolResult:
        self.place_query = {"city": city}
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="高德地图天气查询结果：北京今天晴，气温 26 到 34 度。",
            data={
                "tool": "maps_weather",
                "arguments": {"city": city},
                "result": {
                    "content": [{"type": "text", "text": "北京今天晴，气温 26 到 34 度。"}]
                },
            },
        )


class FakeMultiPoiMapTools:
    def __init__(self) -> None:
        self.place_query: dict[str, object] | None = None

    async def search_place(
        self,
        keywords: str | None,
        *,
        city: str | None = None,
        types: str | None = None,
    ) -> MapToolResult:
        self.place_query = {"keywords": keywords, "city": city, "types": types}
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="高德地图查询结果：广州景点列表。",
            data={
                "tool": "maps_text_search",
                "arguments": {"keywords": keywords, "city": city, "types": types},
                "result": {
                    "structuredContent": {
                        "pois": [
                            {
                                "name": "较远景点",
                                "address": "广州市越秀区",
                                "type": "风景名胜",
                                "distance": "2km",
                            },
                            {
                                "name": "较近景点",
                                "address": "广州市天河区",
                                "type": "风景名胜",
                                "distance": "120",
                            },
                            {
                                "name": "中等景点",
                                "address": "广州市海珠区",
                                "type": "风景名胜",
                                "distance": "800米",
                            },
                        ]
                    },
                    "content": [{"type": "text", "text": "广州景点列表。"}],
                },
            },
        )


class FakeGuangzhouAttractionMapTools:
    def __init__(self) -> None:
        self.place_query: dict[str, object] | None = None

    async def search_place(
        self,
        keywords: str | None,
        *,
        city: str | None = None,
        types: str | None = None,
    ) -> MapToolResult:
        self.place_query = {"keywords": keywords, "city": city, "types": types}
        return MapToolResult(
            status=MapToolStatus.FOUND,
            summary="高德地图查询结果：广州塔、广东省博物馆。",
            data={
                "tool": "maps_text_search",
                "arguments": {"keywords": keywords, "city": city, "types": types},
                "result": {
                    "structuredContent": {
                        "pois": [
                            {
                                "name": "广州塔",
                                "address": "广州市海珠区阅江西路222号",
                                "type": "风景名胜",
                                "distance": "600",
                            },
                            {
                                "name": "广东省博物馆",
                                "address": "广州市天河区珠江东路2号",
                                "type": "科教文化服务",
                                "distance": "900",
                            },
                        ]
                    },
                    "content": [{"type": "text", "text": "广州塔、广东省博物馆。"}],
                },
            },
        )
