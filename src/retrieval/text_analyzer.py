"""为 BM25 提供统一的领域文本归一化和分词。"""
from __future__ import annotations

import re
import unicodedata

import jieba


class TextAnalyzer:
    """让文档 chunk 与查询在进入 BM25 前使用完全相同的分析规则。"""

    _HYPHENS = str.maketrans({"–": "-", "—": "-", "−": "-", "～": "~"})
    _UNIT_REPLACEMENTS = (
        ("毫秒", "ms"),
        ("秒钟", "s"),
        ("千伏", "kv"),
        ("伏特", "v"),
        ("千安", "ka"),
        ("安培", "a"),
        ("毫米", "mm"),
        ("厘米", "cm"),
    )
    _DOMAIN_PATTERN = re.compile(
        r"(?:"
        r"\b[a-z]{1,6}/[a-z]{1,6}\d+(?:\.\d+)*\b"  # GB/Z185.4
        r"|\b[a-z]+(?:-[a-z0-9]+)+\b"                # E-VAC
        r"|\b\d+(?:\.\d+)?~\d+(?:\.\d+)?(?:ms|s|min|kv|v|ka|ma|a|hz|khz|mhz|mm|cm|m)\b"
        r"|\b\d+(?:\.\d+)?(?:ms|s|min|kv|v|ka|ma|a|hz|khz|mhz|mm|cm|m)\b"
        r")",
        re.IGNORECASE,
    )

    def __init__(self, settings: dict | None = None):
        """读取领域词典；词典只影响 BM25，不改变原始 chunk 文本。"""
        settings = settings or {}
        self.enable_query_rewrite = bool(settings.get("enable_query_rewrite", True))
        self.domain_terms = self._load_domain_terms(settings.get("domain_terms", []))
        self.query_rewrites = self._load_query_rewrites(
            settings.get("query_rewrites", [])
        )

    def _load_domain_terms(self, entries: list[dict]) -> list[tuple[str, tuple[str, ...]]]:
        """规范化领域词及别名，忽略格式不完整的配置项。"""
        terms = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            canonical = self.normalize(str(entry.get("canonical", "")))
            aliases = entry.get("aliases", [])
            if not canonical or not isinstance(aliases, list):
                continue
            variants = [canonical]
            variants.extend(self.normalize(str(alias)) for alias in aliases)
            terms.append((canonical, tuple(dict.fromkeys(filter(None, variants)))))
        return terms

    def _load_query_rewrites(self, entries: list[dict]) -> list[tuple[str, tuple[str, ...]]]:
        """加载有限的查询扩展规则，不使用不透明的模型改写。"""
        rewrites = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            trigger = self.normalize(str(entry.get("trigger", "")))
            expansions = entry.get("expansions", [])
            if not trigger or not isinstance(expansions, list):
                continue
            normalized_expansions = tuple(
                filter(None, (self.normalize(str(value)) for value in expansions))
            )
            if normalized_expansions:
                rewrites.append((trigger, normalized_expansions))
        return rewrites

    def normalize(self, text: str) -> str:
        """归一化字符、领域单位和常见型号/标准号的书写差异。"""
        normalized = unicodedata.normalize("NFKC", text or "").translate(self._HYPHENS)
        normalized = re.sub(r"\b([A-Z])\s+([A-Z]{2,}[A-Z0-9-]*)\b", r"\1-\2", normalized)
        normalized = re.sub(r"\b([A-Za-z]{1,6})\s*/\s*([A-Za-z]{1,6})\s*(\d+(?:\.\d+)*)\b", r"\1/\2\3", normalized)
        for source, target in self._UNIT_REPLACEMENTS:
            normalized = normalized.replace(source, target)
        normalized = re.sub(r"(\d+(?:\.\d+)?)\s*(?:~|-|至|到)\s*(\d+(?:\.\d+)?)\s*", r"\1~\2", normalized)
        normalized = re.sub(r"(\d+(?:\.\d+)?)\s+(ms|s|min|kv|v|ka|ma|a|hz|khz|mhz|mm|cm|m)\b", r"\1\2", normalized, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", normalized).strip().casefold()

    def analyze(self, text: str) -> list[str]:
        """分析文档文本，保护型号和数值 token 并追加领域规范词。"""
        normalized = self.normalize(text)
        return self._tokens_from_normalized(normalized)

    def analyze_query(self, text: str) -> list[str]:
        """分析查询并追加受控同义词扩展，供 BM25 查询侧使用。"""
        normalized = self.normalize(text)
        tokens = self._tokens_from_normalized(normalized)
        if self.enable_query_rewrite:
            for trigger, expansions in self.query_rewrites:
                if trigger not in normalized:
                    continue
                for expansion in expansions:
                    tokens.extend(self._tokens_from_normalized(expansion))
        return list(dict.fromkeys(tokens))

    def _tokens_from_normalized(self, normalized: str) -> list[str]:
        """保护领域 token 后分词，避免 jieba 将型号和数值参数打散。"""
        domain_tokens = self._DOMAIN_PATTERN.findall(normalized)
        residual = self._DOMAIN_PATTERN.sub(" ", normalized)
        tokens = [
            token.strip().casefold()
            for token in jieba.lcut(residual)
            if token.strip() and not token.isspace()
        ]
        tokens.extend(domain_tokens)
        tokens.extend(self._matched_domain_terms(normalized))
        return list(dict.fromkeys(tokens))

    def _matched_domain_terms(self, normalized: str) -> list[str]:
        """命中任一别名时追加同一个规范词，使文档和查询可以稳定对齐。"""
        compact = re.sub(r"\s+", "", normalized)
        return [
            canonical
            for canonical, aliases in self.domain_terms
            if any(
                alias in normalized or re.sub(r"\s+", "", alias) in compact
                for alias in aliases
            )
        ]
