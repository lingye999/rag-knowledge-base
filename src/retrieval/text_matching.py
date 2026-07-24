"""查询词、锚点词和近重复判断。"""
from __future__ import annotations

import jieba


def normalize_compact(text: str) -> str:
    """去掉空白并大小写归一，便于中英文混合匹配。"""
    return "".join(text.split()).casefold()


def content_tokens(text: str) -> set[str]:
    """抽取长度不小于 2 的内容词，过滤掉太碎的符号。"""
    return {
        token.strip().casefold()
        for token in jieba.lcut(text)
        if len(token.strip()) >= 2
    }


def token_overlap_score(query_tokens: set[str], text: str) -> float:
    """计算候选文本覆盖查询内容词的比例。"""
    text_tokens = content_tokens(text)
    if not text_tokens:
        return 0.0
    overlap = query_tokens & text_tokens
    return len(overlap) / max(len(query_tokens), 1)


def anchor_terms(query_text: str) -> list[str]:
    """抽取强锚点：引号内容、数字词和英文标识符。"""
    quoted = []
    for left, right in (("“", "”"), ('"', '"'), ("'", "'")):
        parts = query_text.split(left)
        for part in parts[1:]:
            term = part.split(right, 1)[0].strip()
            if term:
                quoted.append(term)

    tokens = [
        token for token in content_tokens(query_text)
        if any(char.isdigit() for char in token)
        or any("A" <= char <= "Z" or "a" <= char <= "z" for char in token)
    ]
    anchors = []
    for term in quoted + tokens:
        if term not in anchors:
            anchors.append(term)
    return anchors


def anchor_overlap_score(anchors: list[str], text: str) -> float:
    """候选文本命中强锚点的比例。"""
    if not anchors:
        return 0.0
    normalized = normalize_compact(text)
    matches = [
        term for term in anchors
        if normalize_compact(term) in normalized
    ]
    return len(matches) / len(anchors)


def is_definition_query(query_text: str) -> bool:
    """判断查询是否像定义/含义类问题。"""
    normalized = normalize_compact(query_text)
    cues = (
        "是什么",
        "指的是什么",
        "指什么",
        "定义",
        "含义",
        "whatis",
        "define",
        "meaning",
    )
    return any(cue in normalized for cue in cues)


def definition_match_score(anchors: list[str], text: str) -> float:
    """给术语定义条目加分，避免表格字段压过真正定义。"""
    if not anchors:
        return 0.0

    normalized = normalize_compact(text)
    score = 0.0
    for term in anchors:
        compact_term = normalize_compact(term)
        if not compact_term or compact_term not in normalized:
            continue
        score = max(score, 0.4)
        term_pos = normalized.find(compact_term)
        following = normalized[term_pos:term_pos + 80]
        if any(cue in following for cue in ("是", "指", "为", "功能", "定义")):
            score = max(score, 0.75)
        if looks_like_definition_entry(compact_term, normalized):
            score = max(score, 1.0)
    return score


def looks_like_definition_entry(compact_term: str, normalized: str) -> bool:
    """识别“3.4 术语名 ...”或“术语和定义”章节里的定义项。"""
    entry_pos = normalized.find(compact_term)
    if entry_pos < 0:
        return False
    prefix = normalized[max(0, entry_pos - 8):entry_pos]
    if any(char.isdigit() for char in prefix) and "." in prefix:
        return True
    return "术语和定义" in normalized[:max(entry_pos, 1)]


def is_near_duplicate(left: str, right: str, threshold: float) -> bool:
    """用字符 n-gram Jaccard 判断最终上下文是否近重复。"""
    left_grams = char_ngrams(left)
    right_grams = char_ngrams(right)
    if not left_grams or not right_grams:
        return False
    jaccard = len(left_grams & right_grams) / len(left_grams | right_grams)
    return jaccard >= threshold


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """生成紧凑文本的字符 n-gram。"""
    compact = normalize_compact(text)
    if len(compact) <= n:
        return {compact} if compact else set()
    return {compact[i:i + n] for i in range(len(compact) - n + 1)}
