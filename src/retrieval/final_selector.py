"""最终候选重打分、精排、过滤和去重。"""
from __future__ import annotations

from collections import defaultdict
import re

from .result_utils import dedupe_by_index
from .text_analyzer import TextAnalyzer
from .text_matching import (
    anchor_overlap_score,
    anchor_terms,
    content_tokens,
    definition_match_score,
    is_definition_query,
    is_near_duplicate,
    token_overlap_score,
)


class FinalContextSelector:
    """把扩展候选收敛成最终返回给 RAG 的上下文。"""

    def __init__(self, settings: dict):
        self.settings = settings
        self._analyzer = TextAnalyzer(settings.get("bm25", {}))

    def select(
        self,
        query_text: str,
        candidates: list[dict],
        top_k: int,
        threshold: float,
        reranker=None,
    ) -> tuple[list[dict], bool, dict]:
        """返回最终上下文、精排失败标记和候选筛选决策。"""
        candidates = dedupe_by_index(candidates)
        candidates = self.rescore(query_text, candidates)
        if reranker and self._should_rerank(query_text, candidates):
            try:
                reranker_settings = self.settings.get("reranker", {})
                max_candidates = max(
                    top_k,
                    int(reranker_settings.get("max_candidates", top_k * 10)),
                )
                pre_candidates = sorted(
                    candidates,
                    key=lambda item: item["score"],
                    reverse=True,
                )[:max_candidates]
                reranked = reranker.rerank(
                    query_text,
                    pre_candidates,
                    top_k=len(pre_candidates),
                )
                selected, decisions = self.diversify_and_filter(
                    reranked,
                    top_k,
                    threshold=None,
                    with_trace=True,
                )
                return selected, False, decisions
            except Exception as exc:
                print(f"[Reranker] failed, falling back to base retrieval: {exc}")
                selected, decisions = self.diversify_and_filter(
                    candidates,
                    top_k,
                    threshold,
                    with_trace=True,
                )
                return selected, True, decisions

        selected, decisions = self.diversify_and_filter(
            candidates,
            top_k,
            threshold,
            with_trace=True,
        )
        return selected, False, decisions

    def _should_rerank(self, query_text: str, candidates: list[dict]) -> bool:
        """仅在复杂或候选分数接近时启用精排，控制在线 CPU 开销。"""
        reranker_settings = self.settings.get("reranker", {})
        min_candidates = int(reranker_settings.get("min_candidates", 2))
        if len(candidates) < min_candidates:
            return False

        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
        score_gap = ranked[0]["score"] - ranked[1]["score"]
        ambiguous_gap = float(reranker_settings.get("ambiguous_score_gap", 0.06))
        if score_gap <= ambiguous_gap:
            return True

        compact_query = "".join(query_text.split()).casefold()
        cues = reranker_settings.get("complex_query_cues", [])
        return any(str(cue).casefold() in compact_query for cue in cues)

    def rescore(self, query_text: str, candidates: list[dict]) -> list[dict]:
        """用词面、锚点、定义和数值答案形态做最终轻量重打分。"""
        if not candidates or not self.settings.get("enable_final_rescore", True):
            return candidates

        query_tokens = self._analysis_tokens(query_text, is_query=True)
        anchors = anchor_terms(query_text)
        definition_like = is_definition_query(query_text)
        lexical_weight = float(self.settings.get("final_lexical_weight", 0.06))
        anchor_weight = float(self.settings.get("final_anchor_weight", 0.08))
        definition_weight = float(self.settings.get("final_definition_weight", 0.06))
        definition_entry_weight = float(
            self.settings.get("final_definition_entry_weight", 0.12)
        )
        phrase_weight = float(self.settings.get("final_phrase_weight", 0.08))
        numeric_weight = float(self.settings.get("final_numeric_weight", 0.18))
        anchor_min_coverage = float(
            self.settings.get("final_anchor_min_lexical_coverage", 0.25)
        )

        rescored = []
        for candidate in candidates:
            text = candidate.get("text", "")
            lexical_score = self._token_overlap_score(query_tokens, text)
            anchor_score = anchor_overlap_score(anchors, text)
            phrase_score = self._query_phrase_score(query_text, text)
            numeric_score = self._numeric_answer_score(
                query_text,
                text,
                lexical_score,
                phrase_score,
            )
            definition_score = (
                definition_match_score(anchors, text)
                if definition_like else 0.0
            )
            definition_entry_score = max(definition_score - 0.75, 0.0) / 0.25
            # 型号等英文锚点必须有一定的正文词面支撑，避免目录或示意图页泛化加分。
            effective_anchor_score = anchor_score * min(
                lexical_score / max(anchor_min_coverage, 0.01),
                1.0,
            )
            base_score = float(candidate.get("score", 0.0))
            final_score = (
                base_score
                + lexical_weight * lexical_score
                + anchor_weight * effective_anchor_score
                + definition_weight * definition_score
                + definition_entry_weight * definition_entry_score
                + phrase_weight * phrase_score
                + numeric_weight * numeric_score
            )
            item = dict(candidate)
            item["base_score"] = round(base_score, 4)
            item["score"] = round(final_score, 4)
            item["lexical_score"] = round(lexical_score, 4)
            item["anchor_score"] = round(anchor_score, 4)
            item["effective_anchor_score"] = round(effective_anchor_score, 4)
            item["phrase_score"] = round(phrase_score, 4)
            item["numeric_score"] = round(numeric_score, 4)
            item["definition_score"] = round(definition_score, 4)
            item["definition_entry_score"] = round(definition_entry_score, 4)
            rescored.append(item)
        return rescored

    def _analysis_tokens(self, text: str, is_query: bool = False) -> set[str]:
        """让最终词面重打分与 BM25 共用领域词典和查询扩展规则。"""
        analyzed = (
            self._analyzer.analyze_query(text)
            if is_query
            else self._analyzer.analyze(text)
        )
        return content_tokens(text) | set(analyzed)

    def _token_overlap_score(self, query_tokens: set[str], text: str) -> float:
        """按统一分析后的 token 计算覆盖率，避免召回和最终筛选规则分叉。"""
        text_tokens = self._analysis_tokens(text)
        if not text_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / max(len(query_tokens), 1)

    @staticmethod
    def _query_phrase_score(query_text: str, text: str) -> float:
        """优先识别查询中的连续领域字段，兼容中文分词拆分。"""
        phrases = re.findall(r"[\u4e00-\u9fff]{2,}", query_text)
        if not phrases:
            return 0.0

        normalized = "".join(text.split()).casefold()
        scores = []
        for phrase in phrases:
            # 去除提问尾部和连接词，只保留能够定位正文的核心字段。
            phrase = re.sub(r"(范围|是多少|多少|什么|几|吗|呢|是)+$", "", phrase)
            phrase = phrase.replace("的", "")
            if len(phrase) < 2:
                continue
            if phrase.casefold() in normalized:
                scores.append(1.0)
                continue
            # 连续四字以上字段（例如“分闸时间”）优先于通用名词命中。
            max_size = min(6, len(phrase))
            for size in range(max_size, 1, -1):
                fragments = {
                    phrase[start:start + size]
                    for start in range(len(phrase) - size + 1)
                }
                if any(fragment.casefold() in normalized for fragment in fragments):
                    scores.append(size / max_size)
                    break
        return max(scores, default=0.0)

    @staticmethod
    def _numeric_answer_score(
        query_text: str,
        text: str,
        lexical_score: float,
        phrase_score: float,
    ) -> float:
        """为数值型问题的“核心语义 + 数值单位”答案页加分。"""
        compact_query = "".join(query_text.split()).casefold()
        compact_text = "".join(text.split()).casefold()
        numeric_cues = ("多少", "范围", "几", "多大", "数值", "参数")
        if not any(cue in compact_query for cue in numeric_cues):
            return 0.0
        if not re.search(r"\d+(?:\.\d+)?", compact_text):
            return 0.0

        unit_groups = (
            (("时间", "时长", "延时"), r"(?:ms|毫秒|秒|分钟|min)"),
            (("电压",), r"(?:kv|v|伏)"),
            (("电流",), r"(?:ka|ma|a|安)"),
            (("频率",), r"(?:hz|khz|mhz)"),
            (("温度",), r"(?:℃|°c|摄氏度)"),
            (("距离", "长度", "尺寸", "宽度", "高度"), r"(?:mm|cm|m|毫米|厘米|米)"),
        )
        expected_unit = None
        for cues, unit_pattern in unit_groups:
            if any(cue in compact_query for cue in cues):
                expected_unit = unit_pattern
                break

        unit_score = 1.0 if expected_unit and re.search(expected_unit, compact_text) else 0.0
        range_score = 1.0 if re.search(r"\d+(?:\.\d+)?\s*(?:~|-|至|到)\s*\d+", compact_text) else 0.0
        # 连续领域字段比通用分词更能说明该数字是在回答当前问题。
        semantic_score = 0.4 * lexical_score + 0.6 * phrase_score
        # 数值本身不足以说明相关性，必须同时命中查询正文语义。
        if semantic_score < 0.2:
            return 0.0
        return 0.5 * semantic_score + 0.3 * unit_score + 0.2 * range_score

    def diversify_and_filter(
        self,
        candidates: list[dict],
        top_k: int,
        threshold: float | None,
        with_trace: bool = False,
    ) -> list[dict] | tuple[list[dict], dict]:
        """按分数筛选，并可返回每个候选被保留或淘汰的原因。"""
        sorted_candidates = sorted(
            candidates,
            key=lambda item: item["score"],
            reverse=True,
        )
        selected = []
        decisions = []
        doc_page_counts: dict[tuple[str, int | None], int] = defaultdict(int)
        page_context_count = 0
        page_context_limit = int(self.settings.get("page_context_max_final_chunks", 1))
        duplicate_threshold = self.settings.get("duplicate_threshold", 0.82)
        for candidate in sorted_candidates:
            decision = {
                "index": candidate.get("index"),
                "doc": candidate.get("doc"),
                "page": candidate.get("page"),
                "score": candidate.get("score"),
                "base_score": candidate.get("base_score"),
                "lexical_score": candidate.get("lexical_score"),
                "phrase_score": candidate.get("phrase_score"),
                "numeric_score": candidate.get("numeric_score"),
                "anchor_score": candidate.get("anchor_score"),
                "definition_score": candidate.get("definition_score"),
                "definition_entry_score": candidate.get("definition_entry_score"),
            }
            if threshold is not None and candidate["score"] < threshold:
                decision["reason"] = "below_threshold"
                decisions.append(decision)
                continue
            duplicate = next(
                (
                    item for item in selected
                    if is_near_duplicate(candidate["text"], item["text"], duplicate_threshold)
                ),
                None,
            )
            if duplicate is not None:
                decision["reason"] = "near_duplicate"
                decision["duplicate_of"] = duplicate.get("index")
                decisions.append(decision)
                continue
            if (
                candidate.get("chunk_type") == "page_context"
                and page_context_count >= page_context_limit
            ):
                decision["reason"] = "page_context_limit"
                decisions.append(decision)
                continue
            key = (candidate.get("doc", ""), candidate.get("page"))
            if doc_page_counts[key] >= 3:
                decision["reason"] = "page_limit"
                decisions.append(decision)
                continue
            if len(selected) >= top_k:
                decision["reason"] = "top_k_limit"
                decisions.append(decision)
                continue
            selected.append(candidate)
            doc_page_counts[key] += 1
            if candidate.get("chunk_type") == "page_context":
                page_context_count += 1
            decision["reason"] = "selected"
            decisions.append(decision)

        trace = {
            "decisions": decisions,
            "reason_counts": dict(
                (reason, sum(item["reason"] == reason for item in decisions))
                for reason in sorted({item["reason"] for item in decisions})
            ),
        }
        return (selected, trace) if with_trace else selected
