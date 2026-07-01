import binascii
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
import re
import struct
from typing import Any
import zlib

AVATAR_SIZE = 512


class AvatarGenerationStatus(StrEnum):
    GENERATED = "generated"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class AvatarGenerationResult:
    status: AvatarGenerationStatus
    summary: str
    url: str | None = None
    alt: str | None = None
    data: dict[str, Any] | None = None


class AvatarGenerator:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        public_url_prefix: str = "/generated/avatars",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.public_url_prefix = public_url_prefix.rstrip("/")

    def generate_player_avatar(
        self,
        profile: dict[str, Any],
        *,
        session_id: str,
    ) -> AvatarGenerationResult:
        player_id = _profile_text(profile.get("player_id"), fallback="unknown")
        nickname = _profile_text(profile.get("nickname"), fallback=f"玩家 {player_id}")
        level = _profile_text(profile.get("level"), fallback="未知")
        server_name = _profile_text(profile.get("server_name"), fallback="未知服务器")
        desc = _profile_text(profile.get("desc"), fallback="暂无个性描述")
        seed = f"{session_id}:{player_id}:{nickname}:{level}:{server_name}:{desc}"
        digest = sha256(seed.encode("utf-8")).hexdigest()
        filename = f"player-{_safe_filename_part(player_id)}-{digest[:10]}.png"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        png_path = self.output_dir / filename
        png_path.write_bytes(
            _render_avatar_png(
                digest=digest,
                desc=desc,
            )
        )

        url = f"{self.public_url_prefix}/{filename}"
        alt = f"{nickname} 的个性头像"
        return AvatarGenerationResult(
            status=AvatarGenerationStatus.GENERATED,
            summary=f"已生成本地 PNG 头像：{url}",
            url=url,
            alt=alt,
            data={
                "player_id": player_id,
                "nickname": nickname,
                "level": level,
                "server_name": server_name,
                "desc": desc,
                "url": url,
                "format": "png",
            },
        )


def build_avatar_generator() -> AvatarGenerator:
    generated_dir = Path(__file__).resolve().parents[2] / "generated" / "avatars"
    return AvatarGenerator(output_dir=generated_dir)


def _profile_text(value: object, *, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _safe_filename_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-")
    return safe or "unknown"


def _render_avatar_png(*, digest: str, desc: str) -> bytes:
    primary, secondary = _colors_from_digest(digest)
    primary_rgb = _hex_to_rgb(primary)
    secondary_rgb = _hex_to_rgb(secondary)
    pixels = bytearray(AVATAR_SIZE * AVATAR_SIZE * 4)
    personality_value = int(sha256(desc.encode("utf-8")).hexdigest()[:2], 16)

    for y in range(AVATAR_SIZE):
        for x in range(AVATAR_SIZE):
            ratio = (x + y) / (AVATAR_SIZE * 2 - 2)
            color = _mix_rgb(primary_rgb, secondary_rgb, ratio)
            _set_pixel(pixels, x, y, (*color, 255))

    for index in range(0, 12):
        raw = int(digest[index * 2 : index * 2 + 2], 16)
        block_x = 42 + (raw * 37 + index * 29) % 420
        block_y = 36 + (raw * 19 + index * 43) % 420
        block_size = 18 + raw % 42
        color = (*_mix_rgb(primary_rgb, secondary_rgb, (raw % 100) / 100), 95)
        _fill_rect(pixels, block_x, block_y, block_size, block_size, color)

    head_color = (247, 250, 252, 255)
    shadow_color = (17, 32, 51, 65)
    accent_color = (*primary_rgb, 255)
    secondary_accent = (*secondary_rgb, 255)
    _fill_circle(pixels, 256, 184, 118, shadow_color)
    _fill_circle(pixels, 256, 170, 104, head_color)
    _fill_rect(pixels, 144, 294, 224, 116, (247, 250, 252, 235))
    _fill_circle(pixels, 204, 160, 14, accent_color)
    _fill_circle(pixels, 308, 160, 14, accent_color)

    mouth_width = 58 + personality_value % 42
    _fill_rect(pixels, 256 - mouth_width // 2, 214, mouth_width, 10, secondary_accent)

    for index in range(6):
        bar_height = 14 + int(digest[20 + index], 16) * 4
        _fill_rect(
            pixels,
            178 + index * 28,
            376 - bar_height,
            16,
            bar_height,
            (*_mix_rgb(primary_rgb, secondary_rgb, index / 5), 255),
        )

    return _encode_png_rgba(AVATAR_SIZE, AVATAR_SIZE, bytes(pixels))


def _colors_from_digest(digest: str) -> tuple[str, str]:
    hue = int(digest[:2], 16)
    palettes = [
        ("#2454a6", "#4f8bd6"),
        ("#276749", "#57b894"),
        ("#7c3aed", "#a78bfa"),
        ("#b45309", "#f59e0b"),
        ("#be123c", "#fb7185"),
        ("#0f766e", "#5eead4"),
    ]
    return palettes[hue % len(palettes)]


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _mix_rgb(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
    ratio: float,
) -> tuple[int, int, int]:
    return (
        round(first[0] * (1 - ratio) + second[0] * ratio),
        round(first[1] * (1 - ratio) + second[1] * ratio),
        round(first[2] * (1 - ratio) + second[2] * ratio),
    )


def _set_pixel(
    pixels: bytearray,
    x: int,
    y: int,
    color: tuple[int, int, int, int],
) -> None:
    if x < 0 or y < 0 or x >= AVATAR_SIZE or y >= AVATAR_SIZE:
        return
    offset = (y * AVATAR_SIZE + x) * 4
    alpha = color[3]
    if alpha >= 255:
        pixels[offset : offset + 4] = bytes(color)
        return

    inverse_alpha = 255 - alpha
    pixels[offset] = (color[0] * alpha + pixels[offset] * inverse_alpha) // 255
    pixels[offset + 1] = (color[1] * alpha + pixels[offset + 1] * inverse_alpha) // 255
    pixels[offset + 2] = (color[2] * alpha + pixels[offset + 2] * inverse_alpha) // 255
    pixels[offset + 3] = 255


def _fill_rect(
    pixels: bytearray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int, int],
) -> None:
    for row in range(y, y + height):
        for column in range(x, x + width):
            _set_pixel(pixels, column, row, color)


def _fill_circle(
    pixels: bytearray,
    cx: int,
    cy: int,
    radius: int,
    color: tuple[int, int, int, int],
) -> None:
    radius_squared = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius_squared:
                _set_pixel(pixels, x, y, color)


def _encode_png_rgba(width: int, height: int, rgba: bytes) -> bytes:
    scanlines = bytearray()
    stride = width * 4
    for y in range(height):
        scanlines.append(0)
        row_start = y * stride
        scanlines.extend(rgba[row_start : row_start + stride])

    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=9))
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(chunk_type)
    checksum = binascii.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)
