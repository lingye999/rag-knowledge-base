from fastembed import TextEmbedding

class EmbeddingService:
    """文本向量化服务封装"""
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 device: str = "cpu"):
        # 1. 存储参数
        self.model_name = model_name
        self.device = device

        self.model, self._dimension = self._init_model_()

    def _init_model_(self):
        # 加载模型
        model = TextEmbedding(self.model_name, device=self.device)

        # 获取维度
        dummy = list(model.embed([""]))[0]  # ← 修正：model 不是 self.model
        dimension = len(dummy)

        # 打印日志
        print(f"[Embedding] 模型 {self.model_name} 设备 {self.device} 加载成功，维度: {dimension}")

        return model, dimension

    def encode(self, text: str) -> list[float]:
        """单条文本 → 向量"""
        vec = list(self.model.embed([text]))[0]
        return vec.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量文本 → 向量列表"""
        if not texts:
            print("[警告] 传入空文本列表，跳过编码")
            return []
        # 分批处理，避免单次过大
        batch_size = 256
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend([v.tolist() for v in self.model.embed(batch)])
        return results

    @property
    def dimension(self) -> int:
        return self._dimension

# if __name__ == "__main__":
#         emb = EmbeddingService()
#         print("创建成功！")

