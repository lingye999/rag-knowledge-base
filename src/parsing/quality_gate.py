"""解析结果质量门控：决定是否需要用全页 OCR 复核。"""
from __future__ import annotations

from dataclasses import dataclass

from .parse_result import TextBlock


@dataclass(frozen=True, slots=True)
class ParseQuality:
    """用于 OCR 回退决策的轻量解析质量指标。"""

    total_chars: int
    readable_ratio: float
    garbled_ratio: float

    @property
    def score(self) -> float:
        """用于比较混合解析与 OCR 结果的相对质量。"""
        return self.total_chars * self.readable_ratio * (1.0 - self.garbled_ratio)


def assess_parse_quality(blocks: list[TextBlock]) -> ParseQuality:
    """计算文本量、可读字符占比和显式乱码占比。"""
    text = "".join(block.text for block in blocks).strip()
    if not text:
        return ParseQuality(total_chars=0, readable_ratio=0.0, garbled_ratio=1.0)

    visible = [char for char in text if not char.isspace()]
    if not visible:
        return ParseQuality(total_chars=0, readable_ratio=0.0, garbled_ratio=1.0)

    readable = sum(
        char.isalnum() or "\u4e00" <= char <= "\u9fff"
        for char in visible
    )
    garbled = sum(char in {"�", "□", "▯", "�"} for char in visible)
    return ParseQuality(
        total_chars=len(visible),
        readable_ratio=readable / len(visible),
        garbled_ratio=garbled / len(visible),
    )


def needs_ocr_fallback(quality: ParseQuality, settings: dict) -> bool:
    """根据保守阈值判断混合解析是否应由 OCR 复核。"""
    if not settings.get("enabled", True):
        return False
    return (
        quality.total_chars < int(settings.get("min_total_chars", 80))
        or quality.readable_ratio < float(settings.get("min_readable_ratio", 0.55))
        or quality.garbled_ratio > float(settings.get("max_garbled_ratio", 0.03))
    )
