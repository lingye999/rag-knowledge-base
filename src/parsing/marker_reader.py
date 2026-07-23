"""Marker PDF 解析器：高质量 PDF → Markdown 文本"""
import tempfile


def read_pdf_marker(path: str) -> str:
    """用 Marker 将 PDF 转为纯文本

    Marker 比 pdfplumber 更强：
    - 保留文档结构（标题、列表、表格）
    - 对扫描件有内置 OCR
    - 输出接近人类阅读的格式
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(path)

        # Marker 返回 MarkdownOutput 对象，取 markdown 字段
        if hasattr(rendered, 'markdown'):
            text = rendered.markdown
        elif hasattr(rendered, 'text'):
            text = rendered.text
        else:
            text = str(rendered)

        if not text or not text.strip():
            return ""

        print(f"[Marker] {path} 解析成功 ({len(text)} 字符)")
        return text

    except ImportError:
        raise RuntimeError("Marker 未安装，请执行: pip install marker-pdf")
    except Exception as e:
        print(f"[Marker] {path} 解析失败: {e}")
        return ""
