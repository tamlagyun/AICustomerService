from app.avatar_generation import AvatarGenerationStatus, AvatarGenerator

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_avatar_generator_creates_local_png_from_player_profile(tmp_path) -> None:
    generator = AvatarGenerator(
        output_dir=tmp_path,
        public_url_prefix="/generated/avatars",
    )

    result = generator.generate_player_avatar(
        {
            "player_id": "1",
            "nickname": "ai大名",
            "level": 12,
            "server_name": "1服",
            "status": "1",
            "desc": "喜欢研究机制，偏好策略搭配。",
        },
        session_id="session-1",
    )

    assert result.status == AvatarGenerationStatus.GENERATED
    assert result.url is not None
    assert result.url.startswith("/generated/avatars/")
    assert result.url.endswith(".png")
    assert result.alt == "ai大名 的个性头像"
    assert result.data is not None
    assert result.data["format"] == "png"

    png_path = tmp_path / result.url.rsplit("/", maxsplit=1)[-1]
    assert png_path.exists()
    assert png_path.read_bytes().startswith(PNG_SIGNATURE)
