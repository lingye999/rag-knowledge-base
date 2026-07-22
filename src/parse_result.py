from dataclasses import dataclass, field


BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class TextBlock:
    text: str
    source: str
    block_type: str = "text"
    page: int | None = None
    bbox: BBox | None = None
    confidence: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class ParseResult:
    path: str
    blocks: list[TextBlock]
    parser: str
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(block.text for block in self.blocks if block.text.strip())
