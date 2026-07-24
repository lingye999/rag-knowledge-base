"""文本分块策略：sentence / paragraph / jieba / size / auto，均支持重叠"""
import re
import jieba

from .chunk import Chunk
from ..parsing.parse_result import TextBlock


def chunk_by_sentence(text: str, max_sentences: int = 8,
                      overlap: int = 1) -> list[str]:
    """按句子分块，支持重叠

    默认 overlap=1，相邻 chunk 共享 1 句（约 12.5%），
    在保持语义连贯和控制冗余之间取得平衡。
    """
    sentences = re.split(r"[。！？.!?]", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    chunks = []
    step = max(1, max_sentences - overlap)
    for i in range(0, len(sentences), step):
        end = min(i + max_sentences, len(sentences))
        chunk = "".join(sentences[i:end])
        chunks.append(chunk)
        if end >= len(sentences):
            break
    return chunks


def chunk_by_paragraph(text: str) -> list[str]:
    """按段落分割（段落本身已是不连续的单位，不做重叠）"""
    paragraphs = text.split("\n")
    return [s.strip() for s in paragraphs if s.strip()]


def chunk_by_size(text: str, chunk_size: int = 400,
                  overlap: int = 80) -> list[str]:
    """固定窗口大小分割，带重叠"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


_SENTENCE_BOUNDARY = re.compile(r"[。！？!?；;\n]")
_UNNATURAL_CHUNK_STARTS = frozenset({"的", "了", "和", "及", "与", "在", "为", "而", "但", "或"})


def _sentence_token_units(words: list[str]) -> list[list[str]]:
    """Keep punctuation and newlines with the sentence they terminate."""
    units = []
    current = []
    for word in words:
        current.append(word)
        if _SENTENCE_BOUNDARY.search(word):
            units.append(current)
            current = []
    if current:
        units.append(current)
    return units


def _tail_units(units: list[list[str]], max_words: int) -> list[list[str]]:
    """Return complete trailing sentences for overlap without splitting one."""
    tail = []
    size = 0
    for unit in reversed(units):
        if size + len(unit) > max_words:
            break
        tail.insert(0, unit)
        size += len(unit)
    return tail


def _chunk_long_token_unit(words: list[str], max_words: int, overlap: int) -> list[str]:
    """Fallback for a single overlong sentence or a table-like row stream."""
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append("".join(words[start:end]))
        if end >= len(words):
            break

        next_start = max(start + 1, end - overlap)
        while (next_start < len(words) and
               words[next_start].strip() in _UNNATURAL_CHUNK_STARTS):
            next_start += 1
        start = next_start
    return chunks


def chunk_by_jieba(text: str, max_words: int = 120,
                   overlap: int = 20) -> list[str]:
    """Chunk Chinese text by complete sentences, with token windows as fallback.

    The previous implementation advanced a fixed token window. That could make
    a chunk start in the middle of a phrase, such as ``的社会``. Here normal
    prose is packed from complete sentence units. Only a sentence longer than
    ``max_words`` falls back to a token window, which is needed for tables and
    malformed PDF text without sentence punctuation.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    words = jieba.lcut(text)
    if not words:
        return []

    chunks = []
    current_units = []
    current_size = 0
    for unit in _sentence_token_units(words):
        unit_size = len(unit)
        if unit_size > max_words:
            if current_units:
                chunks.append("".join(word for part in current_units for word in part))
                current_units = []
                current_size = 0
            chunks.extend(_chunk_long_token_unit(unit, max_words, overlap))
            continue

        if current_units and current_size + unit_size > max_words:
            chunks.append("".join(word for part in current_units for word in part))
            overlap_budget = min(overlap, max_words - unit_size)
            current_units = _tail_units(current_units, overlap_budget)
            current_size = sum(len(part) for part in current_units)

        current_units.append(unit)
        current_size += unit_size

    if current_units:
        chunks.append("".join(word for part in current_units for word in part))
    return chunks


def resolve_chunk_method(text: str, method: str = "auto") -> str:
    """Resolve ``auto`` once so a document uses a consistent chunking policy."""
    if method != "auto":
        return method

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    total_chars = len(text.strip())
    chinese_chars = len(re.findall(r'[一-鿿]', text))
    chinese_ratio = chinese_chars / total_chars if total_chars else 0
    sentence_ends = len(re.findall(r'[。！？.!?]', text))
    sentence_density = sentence_ends / max(1, len(paragraphs))

    if sentence_density < 0.5 and len(paragraphs) >= 3:
        resolved = "paragraph"
    elif chinese_ratio > 0.3:
        resolved = "jieba"
    elif len(paragraphs) >= 3:
        resolved = "paragraph"
    else:
        resolved = "sentence"

    print(f"[Auto] 中文占比 {chinese_ratio:.0%}，段落 {len(paragraphs)} 个，"
          f"句密度 {sentence_density:.1f}，使用 {resolved} 分块")
    return resolved


def chunk_text(text: str, method: str = "auto") -> list[str]:
    """统一分块调度，支持 auto 自动选择"""
    if not text or not text.strip():
        print("[警告] 文本为空，无内容可分块")
        return []

    method = resolve_chunk_method(text, method)

    if method == "sentence":
        return chunk_by_sentence(text)
    elif method == "paragraph":
        return chunk_by_paragraph(text)
    elif method == "jieba":
        return chunk_by_jieba(text)
    elif method == "size":
        return chunk_by_size(text)
    else:
        raise ValueError(
            f"不支持的分块方法: {method}，可选: "
            f"sentence / paragraph / jieba / size / auto"
        )


def chunk_blocks(
    blocks: list[TextBlock],
    method: str,
    doc: str,
    include_page_context: bool = False,
    page_context_max_chars: int = 900,
) -> list[Chunk]:
    """Chunk parsed blocks without dropping their page and parser provenance.

    A chunk remains inside its source block. For PDFs this means a chunk never
    silently crosses a page boundary, so its page stays a valid citation anchor.
    Short pages split into several chunks can additionally retain one complete
    page-context chunk, preventing related table fields from being separated.
    """
    document_text = "\n".join(block.text for block in blocks)
    resolved_method = resolve_chunk_method(document_text, method)
    chunks = []
    for block in blocks:
        block_chunks = chunk_text(block.text, resolved_method)
        for text in block_chunks:
            chunks.append(Chunk(
                text=text,
                doc=doc,
                page=block.page,
                source=block.source,
                chunk_type="content",
            ))
        if (
            include_page_context
            and block.page is not None
            and len(block_chunks) > 1
            and len(block.text) <= page_context_max_chars
        ):
            chunks.append(Chunk(
                text=block.text,
                doc=doc,
                page=block.page,
                source=block.source,
                chunk_type="page_context",
            ))
    return chunks
