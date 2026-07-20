"""Cross-Encoder 重排序器：对召回结果精排，懒加载避免启动卡顿"""
from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-Encoder 重排序（懒加载模式）

    用法:
        reranker = Reranker("BAAI/bge-reranker-base")
        ranked = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        print(f"[Reranker] 就绪（懒加载: {model_name}）")

    def _ensure_loaded(self):
        if self._model is None:
            print(f"[Reranker] 首次使用，加载模型 {self.model_name} ...")
            self._model = CrossEncoder(self.model_name, device=self.device)
            print("[Reranker] 加载完成")

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """对候选集重新排序"""
        if not candidates:
            return []

        self._ensure_loaded()

        pairs = [(query, c["text"][:512]) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        for c, s in zip(candidates, scores):
            c["score"] = round(float(s), 4)
            c["rerank"] = True

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
