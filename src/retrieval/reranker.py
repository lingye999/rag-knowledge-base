"""Cross-Encoder 重排序器：对召回结果精排，log 归一化消除短文本偏置"""
import math
from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-Encoder 重排序（懒加载，log 归一化校正长度偏置）

    Cross-Encoder 天然倾向给短文本高分（信息密度大、匹配信号集中）。
    用 score × log(len) / log(100) 做归一化：100 字以上不降分，更短的自然减弱。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base",
                 device: str = "cpu", local_files_only: bool = False):
        self.model_name = model_name
        self.device = device
        self.local_files_only = local_files_only
        self._model = None
        print(f"[Reranker] 就绪（懒加载: {model_name}）")

    @property
    def is_loaded(self) -> bool:
        """模型权重是否已经驻留在当前进程中。"""
        return self._model is not None

    def preload(self) -> None:
        """在服务启动阶段预加载模型，避免首个检索请求承担加载延迟。"""
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self._model is None:
            print(f"[Reranker] 首次使用，加载模型 {self.model_name} ...")
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                local_files_only=self.local_files_only,
            )
            print("[Reranker] 加载完成")

    def rerank(self, query: str, candidates: list[dict],
               top_k: int = 5) -> list[dict]:
        """对候选集重新排序（log 归一化校正长度偏置）"""
        if not candidates:
            return []

        self._ensure_loaded()

        pairs = [(query, c["text"][:1024]) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        for c, s in zip(candidates, scores):
            # log 归一化：100字=1.0, 10字≈0.5, 2字≈0.15
            length = max(2, len(c["text"]))
            factor = min(1.0, math.log(length) / math.log(100))
            c["score"] = round(float(s * factor), 4)
            c["rerank"] = True

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]
