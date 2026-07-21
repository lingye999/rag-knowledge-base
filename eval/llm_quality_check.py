"""LLM 质量审计：用 LLM 抽样检查 chunk 质量，验证统计评分的准确性

用法:
    python eval/llm_quality_check.py                    # 用 DeepSeek
    python eval/llm_quality_check.py --sample-size 50   # 抽查 50 个
    python eval/llm_quality_check.py --free-api         # 用免费 API
"""
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embedding import EmbeddingService
from src.vector_store.faiss_store import FaissVectorStore
from src.retriever import Retriever
from src.ingestion import IngestionService

PROMPT = """判断文本块是否包含 OCR 乱码、重复字、或无法理解的碎片文字。

输出JSON：
{"bad":true,"why":"乱码"} 或 {"bad":false,"why":"正常"}

规则：
- "汽汽车车"、"宇宇航航"、"锁联械机" → {"bad":true,"why":"重复字/乱码"}
- "：9Y~7Y"、"线跳"、"阻电" → {"bad":true,"why":"碎片/无意义"}
- 完整的句子或段落 → {"bad":false,"why":"正常"}
- 只有数字、符号、单个词 → {"bad":true,"why":"太短"}

只输出一行JSON，不要解释。"""


def check_with_llm(llm, chunk: str) -> dict:
    """调用 LLM 检查单个 chunk"""
    try:
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chunk[:500]},
            ],
            temperature=0,
            max_tokens=80,
        )
        raw = resp.choices[0].message.content.strip()
        import json
        try:
            result = json.loads(raw)
            # 映射 LLM 字段到内部字段
            return {"is_bad": result.get("bad"), "reason": result.get("why", "")}
        except json.JSONDecodeError:
            import re
            raw_clean = raw.strip()
            for pattern in [r'```json\s*(\{.*?\})\s*```', r'\{.*?\}']:
                match = re.search(pattern, raw_clean, re.DOTALL)
                if match:
                    try:
                        result = json.loads(match.group(1) if '```' in pattern else match.group())
                        return {"is_bad": result.get("bad"), "reason": result.get("why", "")}
                    except json.JSONDecodeError:
                        continue
            return _fallback_judge(chunk, raw[:80])
    except Exception as e:
        return {"is_bad": None, "reason": "API错误",
                "note": str(e)[:80]}


def _fallback_judge(chunk: str, llm_raw: str) -> dict:
    """LLM 返回无法解析时，用规则预判"""
    text = chunk.strip()
    if len(text) <= 3 and not any('\u4e00' <= c <= '\u9fff' for c in text):
        return {"is_bad": True, "reason": "乱码(纯符号)", "note": llm_raw}
    if not any('\u4e00' <= c <= '\u9fff' for c in text):
        return {"is_bad": True, "reason": "乱码(无中文)", "note": llm_raw}
    if len(text) > 3:
        cn = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if cn / len(text) < 0.3:
            return {"is_bad": True, "reason": "乱码(中英混杂)", "note": llm_raw}
    return {"is_bad": None, "reason": "解析失败", "note": llm_raw}


def run_audit(sample_size: int = 30, use_free_api: bool = False):
    """主流程"""
    # ── 初始化 ──
    emb = EmbeddingService()
    db = FaissVectorStore(emb.dimension)
    retriever = Retriever(db)
    ingestion = IngestionService(emb, db, retriever)

    # 批量导入测试文档
    doc_dir = "data"
    import_count = 0
    for f in sorted(os.listdir(doc_dir)):
        fp = os.path.join(doc_dir, f)
        if f.lower().endswith((".pdf", ".docx", ".txt")):
            try:
                ingestion.add(fp)
                import_count += 1
            except Exception:
                pass
    if db.count == 0:
        print("❌ 没有可检查的文档")
        return

    print(f"📚 已导入 {import_count} 个文档，共 {db.count} 个 chunk")

    # ── 初始化 LLM ──
    if use_free_api:
        from openai import OpenAI
        llm = type('obj', (object,), {
            'client': OpenAI(
                api_key="sk-0oIlUXCqnd4xuTlU20AmGLJfMyTDW97KQ9Ktaq5cnZ5hg6M7",
                base_url="https://api.llsdog.cn/v1"
            ),
            'model': 'mimo-v2.5-free'
        })()
    else:
        try:
            from src.llm_service import LLMService
            api_key = os.environ.get("DEEPSEEK_API_KEY", "") or "sk-c2419e869b7f4123a1fd0c69fcabc9c0"
            llm = LLMService(api_key=api_key)
        except ValueError:
            print("❌ 需要设置 DEEPSEEK_API_KEY")
            return

    # ── 分层抽样 ──
    all_data = [(db.texts[i], db.meta[i].get("quality", 0.5))
                for i in range(len(db.texts))]
    random.shuffle(all_data)

    high = [c for c in all_data if c[1] >= 0.7]
    mid = [c for c in all_data if 0.4 <= c[1] < 0.7]
    low = [c for c in all_data if c[1] < 0.4]

    samples = []
    n_each = max(5, sample_size // 3)
    for group, label in [(high, "高分段(≥0.7)"), (mid, "中间段(0.4~0.7)"),
                         (low, "低分段(<0.4)")]:
        picked = random.sample(group, min(n_each, len(group)))
        for text, stat_q in picked:
            samples.append((text, stat_q, label))

    print(f"\n🔍 分层抽样 {len(samples)} 个 chunk，LLM 质量审计中...\n")

    # ── LLM 审计 ──
    results = []
    for i, (text, stat_q, label) in enumerate(samples):
        llm_r = check_with_llm(llm, text)
        llm_bad = llm_r.get("is_bad")
        if llm_bad is None:
            status = "❓ 未知"
        elif llm_bad:
            status = "❌ BAD"
        else:
            status = "✅ OK"
        results.append({
            "text": text[:80],
            "stat_q": stat_q,
            "label": label,
            "llm_bad": llm_bad,
            "llm_reason": llm_r.get("reason", "?"),
            "llm_note": llm_r.get("note", ""),
        })
        print(f"  [{i+1:2d}/{len(samples)}] {status} "
              f"| stat={stat_q:.3f} | {llm_r.get('reason','?')} "
              f"| {text[:40]}...")

    # ── 报告 ──
    print("\n" + "=" * 60)
    print("  📊 LLM vs 统计评分 对比报告")
    print("=" * 60)

    for label in ["高分段(≥0.7)", "中间段(0.4~0.7)", "低分段(<0.4)"]:
        group = [r for r in results if r["label"] == label]
        if not group:
            continue
        bad_count = sum(1 for r in group if r["llm_bad"] is True)
        unknown_count = sum(1 for r in group if r["llm_bad"] is None)
        pct = bad_count / len(group) * 100
        print(f"\n  {label} ({len(group)} 个):")
        print(f"    LLM 判定有问题: {bad_count} ({pct:.0f}%)")
        if unknown_count:
            print(f"    解析/API 失败:  {unknown_count} ({unknown_count/len(group)*100:.0f}%)")
        if bad_count > 0:
            for r in group:
                if r["llm_bad"] is True:
                    print(f"       [{r['stat_q']:.3f}] {r['llm_reason']}: "
                          f"{r['text'][:40]}...")
        if unknown_count > 0:
            for r in group:
                if r["llm_bad"] is None:
                    print(f"       ❓ [{r['stat_q']:.3f}] {r['llm_reason']}: "
                          f"{r['llm_note'][:60]}")

    # ── 结论 ──
    total_bad = sum(1 for r in results if r["llm_bad"] is True)
    total_unknown = sum(1 for r in results if r["llm_bad"] is None)
    total_ok = sum(1 for r in results if r["llm_bad"] is False)
    high_bad = sum(1 for r in results if r["llm_bad"] is True and
                   r["label"] == "高分段(≥0.7)")
    low_ok = sum(1 for r in results if r["llm_bad"] is False and
                 r["label"] == "低分段(<0.4)")

    print(f"\n{'='*60}")
    print(f"  🎯 审计结论")
    print(f"{'='*60}")
    print(f"  总抽查: {len(results)} | BAD: {total_bad} | OK: {total_ok} | 未知: {total_unknown}")
    if total_unknown > 0:
        print(f"  ⚠️  {total_unknown} 个 chunk LLM 未能正常评判（需检查 Prompt/网络）")
    if high_bad > 0:
        print(f"  ⚠️  高分段误判: {high_bad} 个")
    if low_ok > 0:
        print(f"  ⚠️  低分段误判: {low_ok} 个")
    if total_unknown == 0 and high_bad == 0 and low_ok == 0:
        print(f"  ✅ 统计评分与 LLM 判断一致")
    print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample-size", type=int, default=30)
    p.add_argument("--free-api", action="store_true")
    args = p.parse_args()
    run_audit(sample_size=args.sample_size, use_free_api=args.free_api)
