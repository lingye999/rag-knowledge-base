# Day1-2 总结与知识图谱

> 日期：2026-07-16 ~ 2026-07-17  
> 项目：RAG 知识库系统  
> 分支：day3

---

## 一、已完成功能清单

### Day 1：检索系统搭建

| 模块 | 功能 | 文件 |
|------|------|------|
| 向量编码 | BAAI/bge-small-zh-v1.5 嵌入模型（512维） | `embedding.py` |
| 向量存储 | FAISS Flat / IVF / HNSW 三种索引，可切换 | `Vector_Store/vector_store_faiss.py` |
| 混合检索 | Dense向量 + BM25关键词 + RRF融合 | `Vector_Store/hybrid_retriever.py` |
| 文档读取 | txt / docx / pdf 多格式支持 | `document_loader.py` |
| 分块策略 | sentence / paragraph / jieba / size / auto | `document_loader.py` |
| PDF 提取 | pdfplumber（文本提取） + EasyOCR（扫描件兜底） | `document_loader.py` |
| 文本清洗 | 乱码页检测 + 表格噪声过滤 + 停用词过滤 | `document_loader.py` |

### Day 2：LLM 集成

| 模块 | 功能 | 文件 |
|------|------|------|
| LLM 服务 | OpenAI 兼容格式封装，支持 DeepSeek 官方及任意中转 API | `llm_service.py` |
| /ask 命令 | 检索 + LLM 生成回答，容错降级 | `main.py` |
| /rewrite 命令 | 自然语言 → 搜索关键词改写 | `main.py` |
| API 安全 | Key 通过环境变量 `DEEPSEEK_API_KEY` 读取 | `main.py` |

### 交互命令一览

```
/search <query>        纯向量检索
/search_jieba <query>  分词后检索
/hybrid_search <query> 混合检索（Dense + BM25 + RRF）
/ask <query>           LLM 问答（检索 + AI 生成）
/rewrite <query>       LLM 改写查询为关键词
/add <file> [ocr]      添加文件（可选 OCR 优先模式）
/count                 查看文档总数
/switch <type>         切换索引类型（flat / ivf / hnsw）
/save <path>           保存数据库
/load <path>           加载数据库
/exit                  退出
```

---

## 二、当前存在的问题

### 🔴 核心问题：pdfplumber 提取质量差

```
PDF 原文：E-VAC配用伊顿电气新一代的真空灭弧室，更适合电力系统的技术和运行条件

提取结果：E-VAC配用伊顿电气新一代的真 ← 视觉换行
空灭弧室，更适合电力系统的          ← 被切断
技术和运行条件；                    ← 不连续
```

**根因**：pdfplumber 按 PDF 页面坐标逐行抓取文字，不理解段落语义。多栏排版、表格、技术图纸更是灾难。

**影响**：分块断裂 → 检索召回碎片化 → 用户体验差

**方案**：
- 短期：段落合并后处理（视觉换行 → 语义段落）
- 长期：Day 4 换用 Marker 解析器

### 🟡 次要问题

| 问题 | 状态 | 说明 |
|------|------|------|
| LLM 网络波动 | 偶发 | DeepSeek 官方 API 偶尔 Connection error，已有容错降级 |
| OCR 速度慢 | 已知 | CPU 模式每页 30-90s，GPU 可大幅加速 |
| 分块策略单一 | 待优化 | 目前仅按 jieba 词数分块，未考虑语义边界 |
| 无重排序 | 待做 | Day 3 任务 |

---

## 三、重点知识点

### 1. RAG 架构

```
用户问题 → 向量编码 → 检索召回 → LLM生成 → 回答
              ↑                       ↑
         嵌入模型                  DeepSeek API
```

RAG 的本质：用外部知识库"喂"给 LLM，让 LLM 基于真实文档回答，减少幻觉。

### 2. 向量检索（Dense Retrieval）

- **Embedding**：把文本映射成 512 维空间中的一个点
- **余弦相似度**：两个向量夹角越小 → 语义越接近
- **FAISS**：Meta 开源的高维向量检索引擎，支持 Flat / IVF / HNSW 等多种索引

```
文本 → BGE模型 → [0.12, -0.34, 0.67, ..., 0.05]  ← 512维向量
                    ↓ L2归一化
                 单位球面上的一个点
```

### 3. BM25（Sparse Retrieval）

基于词频和逆文档频率的关键词匹配算法：

```
BM25(q, d) = Σ IDF(词) × (TF(词, d) × (k1+1)) / (TF(词, d) + k1×(1-b+b×|d|/avgdl))
```

- **TF 饱和**：k1 控制词频增长曲线，避免"一个词出现100次就是100倍重要"
- **IDF**：稀有词权重高，常见词（"的"、"是"）权重低
- **文档长度归一化**：b 参数防止长文档天然占优势

### 4. RRF（Reciprocal Rank Fusion）

融合多个排序列表的方法：

```
RRF_score(d) = 1/(k + rank_dense) + 1/(k + rank_bm25)
```

- k=60（经典值），防止排名 1 和排名 2 之间差距过大
- 不依赖原始分数，只关心相对排名 → 两种不同量纲的分数可以直接融合

### 5. OCR 原理（EasyOCR）

```
PDF页面 → fitz渲染为图片 → numpy array → EasyOCR模型 → 文字+置信度
                                                ↓
                                    CRAFT检测（文字区域定位）
                                    + CRNN识别（区域→文字）
```

**EasyOCR vs PaddleOCR**：
- EasyOCR：基于 PyTorch，API 简单，中文识别不错
- PaddleOCR：基于 PaddlePaddle，中文识别更优，但框架兼容性差（Windows 环境踩坑）

### 6. LLM 集成（OpenAI 兼容格式）

```python
from openai import OpenAI

client = OpenAI(api_key="sk-xxx", base_url="https://api.deepseek.com")
resp = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "你是知识库问答助手"},
        {"role": "user", "content": f"参考文档：\n{docs}\n\n问题：{query}"},
    ],
)
answer = resp.choices[0].message.content
```

**system prompt 技巧**："忽略乱码内容"——让 LLM 自动过滤检索结果中的噪声。

### 7. Git 分支管理

```
main          ← 稳定主干，只合并不直接改
  └─ day3     ← 开发分支，做完 PR 合回 main
  └─ day4     ← 下一个功能
  └─ ...
```

- `git checkout -b day3`：从当前分支创建新分支
- `git push origin day3`：推送到远程
- `git reset --soft`：合并多个 commit 为一个
- `git push --force`：覆盖远程历史（慎用，仅在自己分支）

---

## 四、思维导图

```
                                    RAG 知识库系统
                                          │
        ┌─────────────┬─────────────┬─────┴─────┬─────────────┬─────────────┐
        │             │             │           │             │             │
     文档加载       文本处理       向量检索     混合检索       LLM 服务      Git 管理
        │             │             │           │             │             │
   ┌────┴────┐   ┌────┴────┐   ┌───┴───┐   ┌───┴───┐   ┌───┴───┐   ┌───┴───┐
   │         │   │         │   │       │   │       │   │       │   │       │
 txt      docx  pdfplumber EasyOCR FAISS BM25  Dense BM25  /ask /rewrite main day3
   │         │   │         │   │       │   │       │   │       │   │       │
   ▼         ▼   ▼         ▼   ▼       ▼   ▼       ▼   ▼       ▼   ▼       ▼
 UTF-8   段落+ 坐标提取  图片识别 Flat   TF-IDF 余弦    RRF   DeepSeek  稳定主 开发
 GBK     表格   │         │  IVF   词频   相似度  k=60   V4 Flash  干    分支
         提取   │         │  HNSW  饱和    L2归一   │    │
                │         │   │     │       化     │    │
                ▼         ▼   │     ▼               │    │
             乱码检测    CRAFT  │  jieba分词          │    │
             表格清洗    +CRNN  │                     │    │
             停用词过滤  模型   │                     │    │
                │               │                     │    │
                └───────┬───────┘                     │    │
                        ▼                             │    │
                  分块策略 ────────────────────────────┘    │
                  (sentence/paragraph/jieba/size/auto)      │
                        │                                  │
                        ▼                                  │
                  向量数据库 ←──────────────────────────────┘
                        │
                        ▼
                  检索结果 → LLM → 用户回答
                        │
                        ▼
                  容错降级（API失败时显示原始结果）
```

---

## 五、下一步计划

| 天数 | 内容 | 状态 |
|------|------|------|
| Day 1 | 混合检索 + 代码重构 | ✅ 完成 |
| Day 2 | LLM 集成（DeepSeek） | ✅ 完成 |
| Day 3 | 多路召回 + 重排序 | 🔜 待开始 |
| Day 4 | Marker 解析器替换 pdfplumber | ⏳ 待排期 |
| Day 5 | 监控与优化 | ⏳ 待排期 |
