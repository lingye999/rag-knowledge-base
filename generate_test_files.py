"""生成测试文件：.txt / .docx / .pdf，内容各不同"""
import os
import sys

data_dir = os.path.join(os.path.dirname(__file__), "data")

# ========== 测试内容（每个文件不同主题） ==========

txt_content = """人工智能概述

人工智能（Artificial Intelligence，简称 AI）是计算机科学的一个重要分支，旨在创建能够模拟人类智能的系统。自 1956 年达特茅斯会议以来，AI 已经经历了多次发展和变革。

机器学习是 AI 的核心技术之一。它通过算法让计算机从数据中学习规律，而无需人工编写明确的规则。机器学习主要分为监督学习、无监督学习和强化学习三大类别。

监督学习是最常用的方法。它使用标记好的训练数据来训练模型。常见的算法包括线性回归、决策树、支持向量机和神经网络。监督学习广泛应用于图像分类、垃圾邮件检测和房价预测等任务。

无监督学习则不使用标记数据。算法需要自己从数据中发现隐藏的模式和结构。常见的方法有聚类分析、主成分分析和自编码器。无监督学习常用于客户分群、异常检测和降维处理。

强化学习通过让智能体与环境交互来学习。智能体通过试错来最大化累积奖励。AlphaGo 就是强化学习的经典案例。强化学习在游戏、机器人和自动驾驶领域表现出色。

深度学习是机器学习的一个子领域。它使用多层神经网络来模拟人脑的学习过程。神经网络由输入层、隐藏层和输出层组成，每一层包含多个神经元。近年来，深度学习在图像识别、语音识别和自然语言处理等领域取得了突破性进展。

自然语言处理（NLP）让计算机能够理解和生成人类语言。NLP 的应用包括机器翻译、情感分析、聊天机器人和文本摘要。近年来，大型语言模型（如 GPT 系列）的出现极大地推动了 NLP 技术的发展。

计算机视觉使计算机能够理解和分析图像和视频。卷积神经网络（CNN）是计算机视觉中最常用的深度学习架构。应用包括人脸识别、自动驾驶、医学影像分析和工业质检。

AI 伦理是当前讨论的热点话题。我们需要确保 AI 系统的公平性、透明性和安全性。偏见消除、隐私保护和可解释性是目前面临的主要挑战。

未来，AI 将继续深刻改变我们的生活方式和工作方式。自动化将取代部分重复性工作，但也会创造新的就业机会。人机协作将是未来工作模式的重要方向。"""

docx_content = """Python 编程语言入门

Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。它以简洁的语法和强大的功能而闻名。Python 的名字来源于英国喜剧团体 Monty Python，而不是蛇。

Python 的设计哲学强调代码的可读性。它使用缩进来定义代码块，而不是像其他语言那样使用大括号。这使得 Python 代码看起来整洁且易于理解。Python 之禅是一组指导 Python 设计的哲学原则。

Python 是解释型语言，这意味着代码可以逐行执行，无需编译。这使得开发和调试过程更加高效。Python 也支持交互式编程，可以在终端中直接运行代码并立即看到结果。

Python 拥有丰富的标准库，涵盖了文件操作、网络通信、数据处理等各个方面。此外，还有大量的第三方库可供使用。Python 社区维护了 PyPI 包索引，包含超过 40 万个包。

在数据科学领域，Python 是最流行的编程语言之一。NumPy 提供了高效的数组运算，Pandas 用于数据处理和分析，Matplotlib 和 Seaborn 用于数据可视化。Jupyter Notebook 是数据科学家常用的交互式开发环境。

在机器学习领域，Scikit-learn 提供了各种常用的机器学习算法。TensorFlow 和 PyTorch 是深度学习框架的主流选择。Keras 提供了高级 API，简化了神经网络的构建过程。

在 Web 开发领域，Django 和 Flask 是两个流行的 Web 框架。Django 功能全面，适合大型项目。Flask 轻量灵活，适合小型应用和微服务。FastAPI 是新兴的高性能 Web 框架。

Python 在自动化脚本编写方面也非常出色。你可以用 Python 编写脚本来自动处理文件、发送邮件、爬取网页数据等。Selenium 可以模拟浏览器操作，实现 Web 自动化。

学习 Python 的最佳方式是动手实践。建议从小项目开始，逐步增加难度。阅读优秀的开源代码也是提高编程水平的有效方法。GitHub 上有大量优秀的 Python 开源项目可以学习。

Python 社区非常活跃，有大量的学习资源可供参考。官方文档是最权威的学习资料。Stack Overflow 是解决编程问题的最佳平台。"""

pdf_content = """数据结构与算法基础

数据结构是计算机存储和组织数据的方式。选择合适的数据结构可以显著提高算法的效率。常见的数据结构包括数组、链表、栈、队列、树和图。

数组是最基本的数据结构。它使用连续的内存空间存储相同类型的元素。数组的优点是可以通过索引快速访问元素，时间复杂度为 O(1)。缺点是插入和删除操作效率较低，需要移动大量元素。

链表由一系列节点组成，每个节点包含数据和指向下一个节点的指针。链表分为单向链表、双向链表和循环链表。链表的优点是插入和删除操作效率高，缺点是不能随机访问元素，查找需要遍历。

栈是一种先进后出的数据结构。元素的添加和移除都在同一端进行。栈常用于函数调用管理、括号匹配检测、浏览器的前进后退功能和撤销操作。

队列是一种先进先出的数据结构。元素在一端添加，在另一端移除。队列常用于任务调度、广度优先搜索、缓存系统和打印机任务管理。

树是一种层次化的数据结构。二叉树是最常用的树结构，每个节点最多有两个子节点。二叉搜索树要求左子节点的值小于父节点，右子节点的值大于父节点。平衡二叉树如 AVL 树和红黑树可以保证查找效率。

图由顶点和边组成。图可以分为有向图和无向图。带权图是边带有权重的图。图的遍历算法包括深度优先搜索和广度优先搜索。最短路径算法有 Dijkstra 算法和 Floyd 算法。

算法的时间复杂度用大 O 表示法来描述。常见的时间复杂度有常数时间 O(1)、对数时间 O(log n)、线性时间 O(n)、线性对数时间 O(n log n) 和平方时间 O(n²)。选择高效的算法可以大幅提升程序性能。

排序算法是算法学习的基础。冒泡排序、选择排序和插入排序是简单但效率较低的排序算法，时间复杂度为 O(n²)。快速排序和归并排序是效率较高的排序算法，时间复杂度为 O(n log n)。

二分查找是一种在有序数组中查找目标值的高效算法。它的时间复杂度为 O(log n)，比线性查找快得多。但前提是数组必须是有序的。二分查找的思想也可以应用于其他问题。

动态规划是解决最优化问题的常用方法。它通过将复杂问题分解为子问题来求解，避免重复计算。背包问题和最长公共子序列问题是动态规划的经典案例。动态规划的关键是找到状态转移方程。"""

print(f"Python 版本: {sys.version}")
print(f"目标目录: {data_dir}")
os.makedirs(data_dir, exist_ok=True)

# ========== 生成 .txt ==========
txt_path = os.path.join(data_dir, "AI_概述.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(txt_content.strip())
print(f"[OK] {txt_path}")

# ========== 生成 .docx ==========
print("\n正在生成 .docx ...")
from docx import Document

docx_path = os.path.join(data_dir, "Python入门.docx")
doc = Document()
for line in docx_content.strip().split("\n"):
    doc.add_paragraph(line.strip())
doc.save(docx_path)
print(f"[OK] {docx_path}")

# ========== 生成 .pdf ==========
print("\n正在生成 .pdf ...")
import fitz

pdf_path = os.path.join(data_dir, "数据结构与算法.pdf")

# 创建全新的 PDF 文档
pdf_doc = fitz.open()

# 查找系统自带的中文字体
import subprocess
import re

# Windows 下查找中文字体
possible_fonts = [
    "C:/Windows/Fonts/msyh.ttc",         # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",       # 黑体
    "C:/Windows/Fonts/simsun.ttc",       # 宋体
    "C:/Windows/Fonts/msyhbd.ttc",       # 微软雅黑加粗
    "C:/Windows/Fonts/yahei.ttf",
    "C:/Windows/Fonts/msyh.ttf",
]
font_found = None
for fp in possible_fonts:
    if os.path.exists(fp):
        font_found = fp
        break

lines = pdf_content.strip().split("\n")
y = 820
page_height = 842  # A4
margin_left = 50

page = pdf_doc.new_page(width=595, height=page_height)  # A4 大小

for line in lines:
    if y < 60:
        page = pdf_doc.new_page(width=595, height=page_height)
        y = 820

    text = line.strip()
    if not text:
        y -= 15
        continue

    # 判断是否为标题（行较短且不以标点结尾）
    is_title = len(text) < 30 and not any(text.endswith(p) for p in "。？！，；")

    if font_found:
        # 使用系统字体支持中文
        page.insert_text(
            (margin_left, y),
            text,
            fontsize=14 if is_title else 11,
            fontname="helv",  # 使用 base14 字体，通过 fontfile 指定中文字体
            fontfile=font_found,
        )
    else:
        # 不指定字体（可能无法显示中文，但至少文件能生成）
        page.insert_text(
            (margin_left, y),
            text,
            fontsize=14 if is_title else 11,
            fontname="helv",
        )

    y -= 20 if is_title else 16

pdf_doc.save(pdf_path)
pdf_doc.close()

if font_found:
    print(f"[OK] {pdf_path} (使用字体: {os.path.basename(font_found)})")
else:
    print(f"[OK] {pdf_path} (注意：未找到中文字体，PDF 中文可能显示为乱码)")

print("\n全部生成完毕！")
print(f"文件列表:")
for f in os.listdir(data_dir):
    fpath = os.path.join(data_dir, f)
    size = os.path.getsize(fpath)
    print(f"  {f:30s} {size:>6d} 字节")
