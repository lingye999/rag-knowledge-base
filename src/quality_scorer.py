"""轻量级 chunk 质量评分：无需外部模型，chunk 后顺便算完

指标：
  - 信息熵（entropy）：词汇分布越均匀，信息量越大
  - 唯一词率（TTR）：不重复词占比，越高越精炼
  - 非停用词比例：实质内容占比
  - 句长标准差：句子长度变化大，写作质量更高

最终分 = 0.35 × ent + 0.30 × ttr + 0.25 × content + 0.10 × length_var
"""
import math
import jieba


# 常用中文停用词（内置轻量版，避免加载外部文件）
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

    scores = []
    for chunk in chunks:
        scores.append(_score_single(chunk))
    return scores


def _score_single(text: str) -> float:
    """计算单个 chunk 的质量分"""
    if not text or not text.strip():
        return 0.0

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
    content_ratio = non_stop / total

    # 4. 句长标准差（归一化到 0~1）
    sentences = [s.strip() for s in text.replace("。", ".").replace("？", "?").replace("！", "!").replace("\n", ".").split(".") if len(s.strip()) > 2]
    length_var = _sentence_length_variety(sentences)

    # 加权合成（权重经验值）
    ent_norm = _norm_sigmoid(entropy, 2.5, 1.0)
    score = (0.35 * ent_norm +
             0.30 * ttr +
             0.25 * content_ratio +
             0.10 * length_var)

    return round(max(0.0, min(1.0, score)), 4)


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
        return 0.3  # 单句 chunk 不给高分也不给低分
    lengths = [len(s) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    if mean_len == 0:
        return 0.0
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    std = math.sqrt(variance)
    # 归一化：理想 std 在句子平均长度的 30%~70%
    norm = std / mean_len
    return min(1.0, norm / 0.7)
