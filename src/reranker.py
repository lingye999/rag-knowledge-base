"""Cross-Encoder 重排序器：对召回结果精排，带长度校正避免短文本偏置"""
from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-Encoder 重排序（懒加载模式，带长度校正）

    Cross-Encoder 天然倾向给短文本高分（因为匹配信号更容易集中），
    这里用长度因子做校正——太短的 chunk 会被适当拉低。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base",
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        print(f"[Reranker] 就绪（懒加载: {model_name}）")

    def _ensure_loaded(self):
        if self._model is None:
            print(f"[Reranker] 首次使用，加载模型 {self.model_name} ...")
            self._model = CrossEncoder(self.model_name, device=self.device)
            print("[Reranker] 加载完成")

    def rerank(self, query: str, candidates: list[dict],
               top_k: int = 5) -> list[dict]:
        """对候选集重新排序（带长度校正）"""
        if not candidates:
            return []

        self._ensure_loaded()

        # 截断避免超长文本拖慢推理，但保留更多上下文（1024 而非 512）
        pairs = [(query, c["text"][:1024]) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        for c, s in zip(candidates, scores):
            # 长度校正：短文本（< 100 字）适当降权
            length = len(c["text"])
            if length < 100:
                factor = 0.85 + 0.15 * (length / 100)  # 0.85 → 1.0
                s = s * factor
            c["score"] = round(float(s), 4)
            c["rerank"] = True

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
