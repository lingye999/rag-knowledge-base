"""配置加载器：YAML 优先，环境变量可覆写

用法:
    from config import config
    top_k = config["retrieval"]["top_k"]
"""
import os
import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default.yaml")


def _load() -> dict:
    """加载 YAML 配置，环境变量覆写"""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 环境变量覆写（支持点号路径: CHUNKING_SENTENCE_OVERLAP=2）
    for key, value in os.environ.items():
        if "_" not in key:
            continue
        # 尝试匹配路径: EMBEDDING_MODEL → embedding.model
        parts = key.lower().split("__")  # 使用 __ 作为分隔符
        if len(parts) < 2:
            continue
        # 匹配第一层
        if parts[0] in cfg:
            node = cfg[parts[0]]
            for part in parts[1:-1]:
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    node = None
                    break
            if node is not None and isinstance(node, dict):
                last = parts[-1]
                if last in node:
                    # 类型转换
                    orig = node[last]
                    if isinstance(orig, bool):
                        node[last] = value.lower() in ("true", "1", "yes")
                    elif isinstance(orig, int):
                        node[last] = int(value)
                    elif isinstance(orig, float):
                        node[last] = float(value)
                    else:
                        node[last] = value

        # 简化覆写: CHUNK_SIZE=500
        if key in _SIMPLE_OVERRIDES and value:
            path = _SIMPLE_OVERRIDES[key]
            node = cfg
            for p in path[:-1]:
                node = node.get(p, {})
            if path[-1] in node:
                orig = node[path[-1]]
                if isinstance(orig, bool):
                    node[path[-1]] = value.lower() in ("true", "1", "yes")
                elif isinstance(orig, int):
                    node[path[-1]] = int(value)
                elif isinstance(orig, float):
                    node[path[-1]] = float(value)
                else:
                    node[path[-1]] = value

    return cfg


# 简单环境变量映射（兼容性）
_SIMPLE_OVERRIDES = {
    "LLM_MODEL": ("llm", "model"),
    "LLM_BASE_URL": ("llm", "base_url"),
    "EMBEDDING_MODEL": ("embedding", "model"),
    "INDEX_TYPE": ("index", "type"),
    "RETRIEVAL_TOP_K": ("retrieval", "top_k"),
}

config = _load()
