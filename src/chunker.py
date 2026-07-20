"""文本分块策略：sentence / paragraph / jieba / size / auto，均支持重叠"""
import re
import jieba


def chunk_by_sentence(text: str, max_sentences: int = 8,
                      overlap: int = 2) -> list[str]:
    """按句子分块，支持重叠

    overlap=1 时相邻 chunk 共享 1 个句子，减少切在语义中间的信息丢失。
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


def chunk_by_jieba(text: str, max_words: int = 120,
                   overlap: int = 20) -> list[str]:
    """按中文分词分块，支持重叠"""
    words = jieba.lcut(text)
    if not words:
        return []

    chunks = []
    step = max(1, max_words - overlap)
    for i in range(0, len(words), step):
        end = min(i + max_words, len(words))
        chunk = "".join(words[i:end])
        chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def chunk_text(text: str, method: str = "auto") -> list[str]:
    """统一分块调度，支持 auto 自动选择"""
    if not text or not text.strip():
        print("[警告] 文本为空，无内容可分块")
        return []

    if method == "auto":
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        total_chars = len(text.strip())
        if total_chars > 0:
            chinese_chars = len(re.findall(r'[一-鿿]', text))
            chinese_ratio = chinese_chars / total_chars
        else:
            chinese_ratio = 0

        # 扫描件/OCR 检测：句号密度低 + 回车多 → 用 paragraph 分块
        sentence_ends = len(re.findall(r'[。！？.!?]', text))
        line_count = max(1, len(paragraphs))
        sentence_density = sentence_ends / line_count if line_count > 0 else 0

        if sentence_density < 0.5 and len(paragraphs) >= 3:
            method = "paragraph"
        elif chinese_ratio > 0.3:
            method = "jieba"
        elif len(paragraphs) >= 3:
            method = "paragraph"
        else:
            method = "sentence"

        print(f"[Auto] 中文占比 {chinese_ratio:.0%}，段落 {len(paragraphs)} 个，"
              f"句密度 {sentence_density:.1f}，使用 {method} 分块")

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
