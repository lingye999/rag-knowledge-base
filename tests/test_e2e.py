"""端到端测试：完整检索链路验证

在 PyCharm 中直接运行此文件即可。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding import EmbeddingService
from src.vector_store.faiss_store import FaissVectorStore
from src.vector_store.hybrid import HybridRetriever
from src.retriever import Retriever
from src.ingestion import IngestionService
import jieba
import time

# ── 尝试初始化 LLM（可选） ──
llm = None
try:
    from src.llm_service import LLMService
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if api_key:
        llm = LLMService(api_key=api_key, model="deepseek-v4-flash")
        print("[OK] LLM 已就绪")
    else:
        print("[SKIP] 未设置 DEEPSEEK_API_KEY，跳过 LLM 测试")
except Exception as e:
    print(f"[SKIP] LLM 初始化失败: {e}")

# ── 初始化 ──
print("\n=== 初始化 ===")
emb = EmbeddingService()
db = FaissVectorStore(emb.dimension)
hybrid = HybridRetriever(db)
retriever = Retriever(db)
ingestion = IngestionService(emb, db, hybrid)
print(f"维度: {emb.dimension}, 索引类型: flat, 当前数据: {db.count}")

# ── 测试 1: 添加文件（使用 IngestionService） ──
print("\n=== 测试 1: 添加文件 ===")
test_file = "data/sample.txt"
if os.path.exists(test_file):
    n, fname = ingestion.add(test_file, chunk_method="sentence")
    print(f"[OK] {fname}: {n} chunks, 总数={db.count}, "
          f"doc_registry={list(db.doc_registry.keys())}")

test_file2 = "data/Python入门.docx"
if os.path.exists(test_file2):
    n2, fname2 = ingestion.add(test_file2, chunk_method="paragraph")
    print(f"[OK] {fname2}: {n2} chunks, 总数={db.count}, "
          f"doc_registry={list(db.doc_registry.keys())}")
else:
    print(f"[SKIP] {test_file2} 不存在")

# ── 测试 2: /search（通过 retriever） ──
print("\n=== 测试 2: /search（retriever 三维度加权）===")
for q in ["Python", "文件", "算法"]:
    t0 = time.time()
    vec = emb.encode(q)
    results = retriever.search(vec, top_k=3)
    t1 = time.time()
    print(f"\n[搜索] \"{q}\" ({len(results)}条, {(t1-t0)*1000:.0f}ms):")
    for i, r in enumerate(results):
        doc = r.get("doc", "?")
        text_preview = r["text"][:60].replace("\n", " ")
        print(f"  {i+1}. [{r['score']:.4f}] [{doc}] {text_preview}...")

# ── 测试 3: /search 限定文档 ──
print("\n=== 测试 3: /search 限定文档 ===")
if os.path.exists(test_file):
    doc_name = os.path.basename(test_file)
    vec = emb.encode("Python")
    results = retriever.search(vec, top_k=3, doc_filter=doc_name)
    print(f"[限定 {doc_name}] {len(results)} 条")
    for r in results:
        print(f"  [{r['score']:.4f}] [{r['doc']}] {r['text'][:50]}...")

# ── 测试 4: /hybrid_search ──
print("\n=== 测试 4: /hybrid_search ===")
if hybrid.bm25 is not None:
    for q in ["编程", "数据"]:
        t0 = time.time()
        vec = emb.encode(q)
        tokens = jieba.lcut(q)
        results = hybrid.search(vec, tokens, top_k=3)
        t1 = time.time()
        print(f"\n[混合] \"{q}\" ({len(results)}条, {(t1-t0)*1000:.0f}ms):")
        for i, r in enumerate(results):
            doc = db.meta[r["index"]].get("doc", "未知") if db.meta else "?"
            print(f"  {i+1}. [hybrid {r['score']:.4f}] [{doc}] {r['text'][:50]}...")
else:
    print("[SKIP] BM25 未初始化")

# ── 测试 5: get_chunks_by_doc ──
print("\n=== 测试 5: get_chunks_by_doc ===")
if os.path.exists(test_file):
    doc_name = os.path.basename(test_file)
    chunks = db.get_chunks_by_doc(doc_name)
    print(f"[{doc_name}] {len(chunks)} 个 chunk")

# ── 测试 6: count ──
print(f"\n=== 测试 6: /count ===\n当前总数: {db.count}")

# ── 测试 7: 删除 + compact + save/load ──
print("\n=== 测试 7: 删除 + 保存 + 加载 ===")
if os.path.exists(test_file2):
    doc_name = os.path.basename(test_file2)
    db.delete_doc(doc_name)
    print(f"[标记删除] {doc_name}, deleted数: {len(db.deleted)}, 总数仍为: {db.count}")

    db._compact()
    print(f"[compact后] 总数: {db.count}, deleted数: {len(db.deleted)}, "
          f"doc_registry: {list(db.doc_registry.keys())}")

    db.save("test_session")
    print("[保存] test_session")

    db2 = FaissVectorStore(emb.dimension)
    db2.load("test_session")
    print(f"[加载] 总数: {db2.count}, doc_registry: {list(db2.doc_registry.keys())}")
    assert db2.texts == db.texts, "texts 不一致！"
    assert db2.doc_registry == db.doc_registry, "doc_registry 不一致！"
    print("[OK] save/load 验证通过")

# ── 测试 8: /ask（LLM 问答） ──
print("\n=== 测试 8: /ask ===")
if llm is not None:
    vec = emb.encode("Python")
    results = retriever.search(vec, top_k=5)
    if results:
        chunks = [r["text"] for r in results]
        print(f"检索到 {len(chunks)} 条，LLM 生成中...")
        answer = llm.ask("Python是什么", chunks)
        print(f"[LLM回答]\n{answer[:300]}...")
    else:
        print("无搜索结果")
else:
    print("[SKIP] LLM 未配置")

# ── 清理测试文件 ──
for f in ["test_session.faiss", "test_session_texts.json"]:
    if os.path.exists(f):
        os.remove(f)

print("\n=== 全部测试完成 ===")
