"""文本分块策略：sentence / paragraph / jieba / size / auto"""
import re
import jieba


def chunk_by_sentence(text: str) -> list[str]:
    """按照【。！？.!?】分割句子"""
    sentences = re.split(r"[。！？.!?]", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_by_paragraph(text: str) -> list[str]:
    """按照段落分割"""
    paragraph = text.split("\n")
    return [s.strip() for s in paragraph if s.strip()]


def chunk_by_size(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """固定窗口大小分割，带重叠"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_by_jieba(text: str, max_words: int = 50) -> list[str]:
    """按中文分词结果分块"""
    words = jieba.lcut(text)
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = "".join(words[i:i + max_words])
        chunks.append(chunk)
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
        raise ValueError(f"不支持的分块方法: {method}，可选: sentence / paragraph / jieba / size / auto")
