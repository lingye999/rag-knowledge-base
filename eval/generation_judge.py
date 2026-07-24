"""生成侧 LLM 裁判：忠实度、答案相关性和上下文利用率。"""
from __future__ import annotations

from typing import Any


def _context_text(contexts: list[dict]) -> str:
    sections = []
    for context in contexts:
        chunk_id = context.get("id", "未知 chunk")
        doc = context.get("doc", "未知文档")
        page = context.get("page")
        location = f"，第 {page} 页" if page is not None else ""
        sections.append(
            f"[{chunk_id}] 文档：{doc}{location}\n{context.get('text', '')}"
        )
    return "\n\n---\n\n".join(sections)


def _score(value: Any) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0


class GenerationJudge:
    """通过独立 LLM 对生成结果做可追溯的 RAGAS 风格评判。"""

    def __init__(self, llm):
        self.llm = llm

    def judge_faithfulness(self, answer: str, contexts: list[dict]) -> dict:
        """判断答案中的每个原子结论是否能由实际上下文支持。"""
        prompt = (
            "你是严谨的 RAG 忠实度裁判。只根据给出的参考 chunk 判断答案，"
            "不得使用外部知识。将答案拆成最小、可验证的事实结论；对每个结论判断"
            "是否能被一个或多个 chunk 明确支持。答案中‘没有找到依据’这类拒答，"
            "只有在上下文确实没有相反证据时才可视为 supported。\n"
            "只输出 JSON："
            "{\"claims\":[{\"claim\":\"...\",\"supported\":true,"
            "\"supporting_chunk_ids\":[\"chunk-1\"]}]}"
        )
        result = self.llm.complete_json(
            prompt,
            f"参考 chunk：\n{_context_text(contexts)}\n\n答案：\n{answer}",
        )
        available_ids = {str(context.get("id")) for context in contexts}
        claims = []
        for item in result.get("claims", []):
            if not isinstance(item, dict) or not str(item.get("claim", "")).strip():
                continue
            supporting_ids = [
                str(chunk_id) for chunk_id in item.get("supporting_chunk_ids", [])
                if str(chunk_id) in available_ids
            ]
            supported = bool(item.get("supported")) and bool(supporting_ids)
            claims.append({
                "claim": str(item["claim"]).strip(),
                "supported": supported,
                "supporting_chunk_ids": supporting_ids,
            })
        score = sum(item["supported"] for item in claims) / len(claims) if claims else 0.0
        return {
            "score": score,
            "claims": claims,
            "unsupported_claims": [item["claim"] for item in claims if not item["supported"]],
        }

    def judge_answer_relevancy(self, question: str, answer: str) -> dict:
        """判断答案是否直接、完整地回答问题，不评价事实是否有来源。"""
        prompt = (
            "你是问答相关性裁判。仅根据问题和答案判断答案是否直接回答问题，"
            "是否遗漏问题的主要要求，是否包含大量无关内容。不要评价事实真假。"
            "score 取 0 到 1，1 表示完全相关。只输出 JSON："
            "{\"score\":0.0,\"reason\":\"...\"}"
        )
        result = self.llm.complete_json(prompt, f"问题：{question}\n\n答案：{answer}")
        return {
            "score": _score(result.get("score")),
            "reason": str(result.get("reason", "")),
        }

    def judge_context_utilization(self, question: str, answer: str,
                                  contexts: list[dict]) -> dict:
        """判断 Top K 中哪些 chunk 实际有助于回答，用于诊断上下文噪声。"""
        prompt = (
            "你是 RAG 上下文利用率裁判。给定问题、答案和按排名提供的 chunk，"
            "选出真正帮助回答问题或支持答案结论的 chunk ID。不要因为 chunk 来自"
            "同一主题就判为有用。只输出 JSON：{\"useful_chunk_ids\":[\"chunk-1\"]}"
        )
        result = self.llm.complete_json(
            prompt,
            f"问题：{question}\n\n答案：{answer}\n\n参考 chunk：\n{_context_text(contexts)}",
        )
        available_ids = {str(context.get("id")) for context in contexts}
        useful_ids = []
        for chunk_id in result.get("useful_chunk_ids", []):
            chunk_id = str(chunk_id)
            if chunk_id in available_ids and chunk_id not in useful_ids:
                useful_ids.append(chunk_id)
        return {
            "score": len(useful_ids) / len(contexts) if contexts else 0.0,
            "useful_chunk_ids": useful_ids,
            "unused_chunk_ids": sorted(available_ids - set(useful_ids)),
        }
