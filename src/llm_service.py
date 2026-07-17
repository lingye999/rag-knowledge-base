from openai import OpenAI
import os

# 默认配置（DeepSeek 官方 API）
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
ENV_KEY = "DEEPSEEK_API_KEY"

class LLMService:
    """LLM 服务封装（支持任意 OpenAI 兼容 API）"""

    def __init__(self, api_key: str = "", model: str = "",
                 base_url: str = ""):
        if not base_url:
            base_url = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL)
        if not api_key:
            api_key = os.environ.get(ENV_KEY, "")
        if not api_key:
            raise ValueError(f"需要 {ENV_KEY}。\n"
                             f"设置环境变量: set {ENV_KEY}=sk-xxx")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model or DEFAULT_MODEL

    def ask(self, query: str, context: list[str], top_k: int = 5) -> str:
        """检索 + LLM 生成回答

        参数:
            query: 用户问题
            context: 检索到的文本块列表
            top_k: 取前几条送给 LLM（防超长）
        """
        if not context:
            return "未检索到相关内容。"

        # 截取 top_k 条作为上下文
        docs = "\n---\n".join(context[:top_k])

        system_prompt = (
            "你是一个知识库问答助手。"
            "根据用户的问题和检索到的参考文档，给出准确、简洁的回答。"
            "如果参考文档中有不相关或乱码的内容，请忽略它们。"
            "回答时仅基于提供的参考文档，不要编造信息。"
        )
        user_prompt = f"参考文档：\n{docs}\n\n问题：{query}"

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[LLM 调用失败: {e}]\n\n检索到以下相关内容（共 {len(context)} 条）：\n" + \
                   "\n---\n".join(context[:top_k])

    def rewrite(self, query: str) -> str:
        """将自然语言查询改写为关键词，适合向量检索"""
        system_prompt = (
            "你是一个查询改写助手。"
            "把用户输入的自然语言问题改写成适合搜索的短句关键词。"
            "只输出改写后的文本，不要解释。"
        )
        user_prompt = f"改写为搜索关键词：{query}"

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=128,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[改写失败: {e}] 原始: {query}"
