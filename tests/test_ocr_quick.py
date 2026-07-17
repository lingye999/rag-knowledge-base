"""快速 OCR 测试：只处理前 2 页，验证流程"""
import fitz, numpy as np, time

PDF_PATH = "data/E-VAC固封系列中压真空断路器-中文-2024-07-09.pdf"

print("加载 EasyOCR（只需一次）...")
import easyocr
reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
print("就绪\n")

doc = fitz.open(PDF_PATH)
for page_num in [0, 1]:  # 只测前2页
    page = doc[page_num]
    t0 = time.time()
    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    result = reader.readtext(img)
    elapsed = time.time() - t0

    texts = [t for _, t, conf in result if conf > 0.3]
    full = "".join(texts)

    print(f"=== 第 {page_num+1} 页 [{elapsed:.1f}s] ===")
    print(full[:300])
    print(f"...(共 {len(full)} 字)\n")

doc.close()
print("OCR 流程验证通过 ✓")
