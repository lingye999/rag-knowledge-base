"""纯 OCR 读取测试：对比 pdfplumber vs OCR 的文字质量"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from src.document import read_file
from src.chunker import chunk_text

PDF_PATH = "data/E-VAC固封系列中压真空断路器-中文-2024-07-09.pdf"

print("=" * 60)
print("对比测试：pdfplumber vs EasyOCR")
print("=" * 60)

# ── 1. pdfplumber（默认） ────────────────────────
print("\n[1/2] pdfplumber 模式...")
t0 = time.time()
text_plumber = read_file(PDF_PATH, force_ocr=False)
t1 = time.time()
chunks_plumber = chunk_text(text_plumber, "auto")
print(f"  耗时: {t1-t0:.1f}s | 总字数: {len(text_plumber)} | 分块: {len(chunks_plumber)}")
print(f"  前200字: {text_plumber[:200]}...")

# ── 2. EasyOCR ───────────────────────────────────
print("\n[2/2] EasyOCR 模式...")
t0 = time.time()
text_ocr = read_file(PDF_PATH, force_ocr=True)
t1 = time.time()
chunks_ocr = chunk_text(text_ocr, "auto")
print(f"  耗时: {t1-t0:.1f}s | 总字数: {len(text_ocr)} | 分块: {len(chunks_ocr)}")
print(f"  前200字: {text_ocr[:200]}...")

# ── 对比 ─────────────────────────────────────────
print("\n" + "=" * 60)
print("对比总结")
print("=" * 60)
print(f"  {'指标':<15} {'pdfplumber':<20} {'EasyOCR':<20}")
print(f"  {'─'*15} {'─'*20} {'─'*20}")
print(f"  {'总字数':<15} {len(text_plumber):<20} {len(text_ocr):<20}")
print(f"  {'分块数':<15} {len(chunks_plumber):<20} {len(chunks_ocr):<20}")
