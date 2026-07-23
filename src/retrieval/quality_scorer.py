"""轻量级 chunk 质量评分：无需外部模型，chunk 后顺便算完

指标：
  - 信息熵（entropy）：词汇分布越均匀，信息量越大
  - 唯一词率（TTR）：不重复词占比，越高越精炼
  - 非停用词比例：实质内容占比
  - 句长标准差：句子长度变化大，写作质量更高

OCR 专项检测（v2 新增）：
  - 重复字模式：检测 "汽汽车车" "宇宇航航" 类 OCR 错误
  - 碎片化检测：过短且无连续语义的文本片段

最终分 = 基础分 × OCR惩罚 × 碎片惩罚
"""
import math
import re
import jieba


# 常用中文停用词（内置轻量版）
_STOP_WORDS: set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "从",
    "为", "以", "及", "与", "但", "而", "或", "被", "把", "对", "等",
    "能", "可", "将", "并", "中", "让", "向", "所", "得", "地", "过",
    "个", "之", "来", "用", "还", "做", "其", "最", "比", "于", "更",
    "些", "每", "种", "样", "什么", "怎么", "因为", "所以", "如果", "虽然",
    "但是", "而且", "然后", "已经", "可以", "应该", "可能", "这个", "那个",
    "这些", "那些", "这里", "那里", "之", "乎", "者", "也", "啊", "呢",
    "哦", "嗯", "哈", "嘛", "吗", "吧", "呀", "啦",
}


def compute_quality_scores(chunks: list[str]) -> list[float]:
    """对每个 chunk 计算质量分，返回 0~1 之间的分数列表"""
    if not chunks:
        return []
    return [_score_single(c) for c in chunks]


def _score_single(text: str) -> float:
    """计算单个 chunk 的质量分"""
    if not text or not text.strip():
        return 0.0

    # ── OCR 专项检测 ──
    ocr_penalty = _ocr_repeat_penalty(text)
    fragment_penalty = _fragment_penalty(text)

    # ── 基础指标 ──
    words = jieba.lcut(text)
    total = len(words)
    if total == 0:
        return 0.0

    # 1. 信息熵
    entropy = _entropy(words)

    # 2. 唯一词率（TTR）
    ttr = len(set(words)) / total

    # 3. 非停用词比例
    non_stop = sum(1 for w in words if w not in _STOP_WORDS)
    content_ratio = non_stop / total if total > 0 else 0

    # 4. 句长标准差
    sentences = [s.strip() for s in
                 text.replace("。", ".").replace("？", "?").replace("！", "!")
                 .replace("\n", ".").split(".")
                 if len(s.strip()) > 2]
    length_var = _sentence_length_variety(sentences)

    # 加权合成
    ent_norm = _norm_sigmoid(entropy, 2.5, 1.0)
    base = (0.35 * ent_norm +
            0.30 * ttr +
            0.25 * content_ratio +
            0.10 * length_var)

    # 应用 OCR 和碎片惩罚
    score = base * ocr_penalty * fragment_penalty
    return round(max(0.0, min(1.0, score)), 4)


def _ocr_repeat_penalty(text: str) -> float:
    """检测 OCR 重复字模式，返回 0~1 的惩罚因子

    "汽汽车车" → 重复字密度高 → 惩罚 0.5
    "宇宇航航" → 同上
    """
    if len(text) < 4:
        return 1.0

    # 统计连续相同字符的数量
    repeat_chars = 0
    i = 0
    while i < len(text) - 1:
        if text[i] == text[i + 1]:
            # 找到一个重复对，跳过这对
            repeat_chars += 2
            i += 2
        else:
            i += 1

    if repeat_chars == 0:
        return 1.0

    # 重复字占比
    ratio = repeat_chars / len(text)
    # 超过 25% 的字符是重复的 → 严重惩罚
    if ratio >= 0.5:
        return 0.4
    elif ratio >= 0.25:
        return 0.6
    else:
        return 0.8


def _fragment_penalty(text: str) -> float:
    """检测碎片化文本，返回 0~1 的惩罚因子

    "阻电" → 2字无上下文 → 惩罚 0.5
    "线跳" → 同上
    "科室" → 虽短但是独立语义单位 → 不惩罚
    """
    text = text.strip()
    length = len(text)

    if length >= 15:
        return 1.0

    # 极短
    if length <= 3:
        chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese == length and length >= 2:
            # jieba 分出来后是单个词 → 干净短词，不惩罚
            words = jieba.lcut(text)
            if len(words) == 1:
                return 0.8
            return 0.5
        if chinese == 0:
            return 0.3
        return 0.5

    # 4-14 字符：检查是否有完整句子结构
    if length <= 8:
        chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese < length * 0.5:
            return 0.5  # 大量非中文字符
        # 检查是否有动词/形容词（简单启发：包含常见词缀）
        has_verb_like = any(w in jieba.lcut(text) for w in
                           ["是", "有", "在", "为", "可", "能", "会", "要", "让",
                            "提供", "使用", "支持", "包括", "用于", "采用", "具有"])
        if not has_verb_like and chinese <= 4:
            return 0.6

    return 1.0


def _entropy(words: list[str]) -> float:
    """计算信息熵"""
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    n = len(words)
    ent = 0.0
    for count in freq.values():
        p = count / n
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def _norm_sigmoid(x: float, mid: float, slope: float) -> float:
    """用 sigmoid 将连续值映射到 0~1"""
    return 1.0 / (1.0 + math.exp(-slope * (x - mid)))


def _sentence_length_variety(sentences: list[str]) -> float:
    """句长标准差归一化"""
    if len(sentences) < 2:
        return 0.3
    lengths = [len(s) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    if mean_len == 0:
        return 0.0
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    std = math.sqrt(variance)
    norm = std / mean_len
    return min(1.0, norm / 0.7)
