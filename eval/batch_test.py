"""批量导入测试：丢一堆文件进去，看结果

用法:
    python eval/batch_test.py                  # 用 data/ 目录的文档
    python eval/batch_test.py <文件夹路径>      # 自定义文件夹
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding import EmbeddingService
from src.vector_store.faiss_store import FaissVectorStore
from src.retriever import Retriever
from src.ingestion import IngestionService


def batch_ingest(folder_path: str, emb, db, retriever, ingestion):
    """批量导入文件夹内所有支持的文件"""
    supported = (".pdf", ".docx", ".txt", ".md", ".html", ".doc")

    files = []
    for f in sorted(os.listdir(folder_path)):
        if f.lower().endswith(supported):
            files.append(os.path.join(folder_path, f))

    if not files:
        print(f"❌ 文件夹 {folder_path} 中没有支持的文件")
        return

    print(f"\n📂 找到 {len(files)} 个文件")
    print("=" * 60)

    results = {
        "total": len(files),
        "success": [],
        "skipped": [],
        "failed": [],
        "total_chunks": 0,
        "total_quality_sum": 0.0,
        "low_quality_chunks": 0,
    }

    for fpath in files:
        fname = os.path.basename(fpath)
        t0 = time.time()

        try:
            n, doc_name = ingestion.add(fpath)
            t1 = time.time()

            if n == 0:
                results["skipped"].append({
                    "file": fname,
                    "reason": "无有效文本",
                    "time": f"{(t1-t0):.1f}s",
                })
                print(f"  ⏭️  {fname} — 跳过（无文本）")
            else:
                # 统计质量分
                qualities = []
                for entry in db.meta[-n:]:
                    q = entry.get("quality", 0.5)
                    qualities.append(q)
                    if q < 0.3:
                        results["low_quality_chunks"] += 1

                avg_q = sum(qualities) / len(qualities) if qualities else 0
                results["success"].append({
                    "file": fname,
                    "chunks": n,
                    "avg_quality": round(avg_q, 3),
                    "time": f"{(t1-t0):.1f}s",
                })
                results["total_chunks"] += n
                results["total_quality_sum"] += sum(qualities)
                print(f"  ✅ {fname} — {n} chunks, "
                      f"平均质量 {avg_q:.2f}")

        except Exception as e:
            results["failed"].append({
                "file": fname,
                "reason": str(e)[:80],
                "time": "-",
            })
            print(f"  ❌ {fname} — 失败: {e}")

    # ── 汇总报告 ──
    print("\n" + "=" * 60)
    print("  📊 批量导入报告")
    print("=" * 60)

    s, k, f = len(results["success"]), len(results["skipped"]), len(
        results["failed"])
    print(f"  成功: {s}  |  跳过: {k}  |  失败: {f}  |  总计: {results['total']}")

    if results["success"]:
        avg_q = (results["total_quality_sum"] / results["total_chunks"]
                 if results["total_chunks"] > 0 else 0)
        low_pct = (results["low_quality_chunks"] / results["total_chunks"] * 100
                   if results["total_chunks"] > 0 else 0)
        print(f"\n  Chunk 总数:     {results['total_chunks']}")
        print(f"  平均质量分:     {avg_q:.3f}")
        print(f"  低质量 chunk:   {results['low_quality_chunks']} 个 "
              f"({low_pct:.1f}%)")
        print(f"  文档注册:       {list(db.doc_registry.keys())}")

        # 质量预警
        if avg_q < 0.4:
            print("\n  ⚠️  平均质量分偏低，建议检查文档格式或 OCR 配置")
        if low_pct > 10:
            print(f"  ⚠️  低质量 chunk 比例偏高 ({low_pct:.1f}% > 10%)")

    if results["failed"]:
        print(f"\n  失败详情:")
        for item in results["failed"]:
            print(f"    ❌ {item['file']}: {item['reason']}")

    if results["skipped"]:
        print(f"\n  跳过详情:")
        for item in results["skipped"]:
            print(f"    ⏭️ {item['file']}: {item['reason']}")

    print("\n" + "=" * 60)

    return results


if __name__ == "__main__":
    # 批量导入测试脚本
    # 使用方式: python eval/batch_test.py [文件夹路径]

    emb = EmbeddingService()
    db = FaissVectorStore(emb.dimension)
    retriever = Retriever(db)
    ingestion = IngestionService(emb, db, retriever)

    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    batch_ingest(folder, emb, db, retriever, ingestion)
