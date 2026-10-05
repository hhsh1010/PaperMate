# golden 标注集（A-01）

本目录是检索评测的**标准答案集**，由 `eval/run_eval.py` 读取。

## 文件

| 文件 | 类型 | 用例数 | 语料 | 用途 |
|---|---|---|---|---|
| `golden.jsonl` | **合成**（内联短句） | 20 | 无（`docs` 字段内联） | 指标公式与链路自检；**不代表真实质量** |
| `golden_real.jsonl` | **真实文档**（L0-2，2026-09-16） | 20（可答 15 / 负样本 5） | `eval/corpus/*.pdf`（3 篇真实 PDF） | **对外结论只认这一套** |

### ⚠️ 两套集的本质差别（选错等于自欺）

`golden.jsonl` 的 `docs` 是 `make_golden.py` 里**硬编码的内联句子**，
`gold` 由人工按列表位置指定 —— 它**完全绕过真实链路**（PDF 抽取、换行重排、语义分块、
表格与公式文本）。20 条合成短句上的高指标**不代表真实可用**。
该集保留的正当用途只有两个：① 指标公式回归（`eval/test_eval.py`）；② 秒级链路自检。

`golden_real.jsonl` 的构造方式（`eval/make_golden_real.py`）：

1. 用 PyMuPDF 生成 3 篇**真实 PDF** 到 `eval/corpus/`
   （中文论文 / 英文论文 / 中英混排长文，含章节标题、编号段落、表格行、公式行）；
2. 交给**生产同款链路**（`RAGPipeline.add_document` → `DocumentParser` + `Chunker`）解析分块，
   拿到**真实 chunk id**；
3. `gold` **不是手写的**：把每个问题的"答案原文片段"在真实 chunk 集合里**定位**，
   命中哪个块哪个块就是 gold（gain=3）；同主题块按关键词标 gain=2；
4. **定位失败即 `SystemExit`** —— 宁可脚本红，也不让评测集悄悄退化成测不到真实证据；
5. 附 **5 条负样本**（语料中确实没有答案的问题，`answerable: false`）。

## 字段

| 字段 | 必填 | 说明 |
|---|---|---|
| `id` | ✅ | 样例编号（`q01`… / `r01`… / `m01`… / `n01`…） |
| `query` | ✅ | 查询语句 |
| `docs` | 二选一 | **内联**：`{文档名: [分块文本, ...]}`，脚本按位置生成 id `文档名-c{序号}` |
| `corpus` | 二选一 | **真实文档**：`[doc_id, ...]`，从 `eval/corpus/{doc_id}.pdf` 走生产链路加载 |
| `answerable` | 真实集必填 | `false` = 负样本（语料无答案，期望如实拒答） |
| `answer_sentence` | 真实集必填 | 答案**原文片段**（gold 的来源，供可复现性测试重新定位） |
| `gold` | ✅ | **必须命中**的核心证据 chunk id 列表（决定 Recall/Precision/MAP/Hit） |
| `gains` | 可选 | **分级相关性** `{chunk_id: 3/2/1}`，供 nDCG 使用；缺省视为不相关 |
| `reference` | 建议 | 参考答案（人工核对用） |

## 分级定义（`gains`）

| 等级 | 含义 | 判据 |
|---|---|---|
| **3** | 直接证据 | 单独拿出这一段就能回答问题 |
| **2** | 强相关 | 同主题且提供关键上下文/前提，但不直接给出答案 |
| **1** | 弱相关 | 同主题但只沾边（如链路相邻环节、对比概念） |
| 不写 | 不相关 | 与问题无关 |

**约定**：`gold` 中的 chunk 必须在 `gains` 里且取**最高档**（3）——
由写入脚本与 `eval/test_golden_real.py` 双重 assert 校验。

## 为什么需要 `gains`

二值相关性（相关/不相关）下，nDCG **看不出**「第 1 名是次级相关」与「第 1 名完全无关」的区别
（两者都被记为 0 贡献）。分级标注让 nDCG 能反映这一差别 —— 见
`eval/test_eval.py::test_ndcg_graded_distinguishes_secondary_hits`。

## 负样本为什么不计检索指标

没有 gold 时 recall/nDCG 无意义。`run_case` 对负样本**不产出任何指标键**，
而 `summarize_results` 只平均"实际存在的指标键"，因此它们**自动不进分母**
（否则会以 0 计入，把报告拉成误导性数字）。
负样本真正的用途在 **L0-3 的答案层判官**：语料无答案时是否如实拒答、有没有编造引用。

## 运行

```bash
python eval/make_golden_real.py                              # 生成 corpus/*.pdf 与 golden_real.jsonl
python eval/run_eval.py --cases cases/golden_real.jsonl --embedder real --profile prod
```

- `--profile prod`（默认）= 对齐线上：CrossEncoder 重排 + jieba 分词 + BGE-M3 稀疏向量；
  `--profile fast` = ScoreReranker + 字符级分词（秒级，**仅供链路自检，数字不可对外**）。
- 报告里的 `profile` 字段会写明口径 —— 拿到两份数字先比 `profile`，避免"两套口径"混用。

## ⚠️ 标注状态

**当前 `gains` 为模型初标（2026-09-15 合成集 / 2026-09-16 真实集），尚未人工复核。**
分级判断带有主观性（尤其 2 与 1 的边界），正式使用前建议由领域人员过一遍，重点看：

1. grade 3 是否确实"单独可回答问题"；
2. 有没有遗漏应当标注的相关段落（漏标会低估检索质量）；
3. 泛问类问题（如 q14「论文常用的评估指标」）会有多个 grade 3，注意别只标一个。
