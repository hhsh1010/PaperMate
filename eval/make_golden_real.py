"""生成**真实文档**检索评测集（L0-2，2026-09-16）

## 为什么要重做一套

原 `cases/golden.jsonl` 是 `make_golden.py` 里**硬编码的内联短句**：
`docs` 字段直接写 `{"transformer": ["循环神经网络处理长序列时存在串行瓶颈。", ...]}`，
`gold` 由人工按列表位置指定（`transformer-c1`）。
它**完全绕过了真实链路** —— PDF 抽取、换行/分栏重排、语义分块、表格与公式文本，
这些都是检索质量真正会出问题的地方；20 条合成短句上的高指标**不代表真实可用**
（`docs/项目不足清单-对照五阶段清单复核.md` §3 A1）。

## 本脚本怎么做（关键：gold 由真实链路产出，不手工指定）

1. 用 PyMuPDF 生成 **3 篇真实 PDF** 到 `eval/corpus/`（中文论文 / 英文论文 / 中英混排长文，
   含章节标题、编号段落、表格行、公式行 —— 都是真实论文里会遇到的文本形态）；
2. 把 PDF 交给**生产同款链路**（`RAGPipeline.add_document` → `DocumentParser` + `Chunker`）
   解析分块，拿到**真实的 chunk id 与文本**；
3. gold **不是手写的**：把每个问题的"答案原文片段"拿去在真实 chunk 集合里**定位**，
   命中哪个块，哪个块就是 gold（gain=3）；再按关键词给同主题块标 gain=2；
4. **定位失败即报错退出**（`SystemExit`）—— 宁可脚本红，也不让评测集悄悄退化成"测不到真实证据"；
5. 另附 **负样本**（语料里确实没有答案的问题，`answerable: false`）。
   ⚠️ 负样本**不参与检索指标聚合**（没有 gold，recall/ndcg 无意义，`summarize_results`
   只平均"存在的指标键"，天然跳过），它们服务于 **L0-3 的答案层判官**（是否如实拒答）。

用法：
  python eval/make_golden_real.py          # 生成 corpus/*.pdf 与 cases/golden_real.jsonl
  python eval/run_eval.py --cases cases/golden_real.jsonl [--embedder real]
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = str(_HERE.parents[0] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pymupdf  # noqa: E402

from lightcore.rag.embedder import FakeHashEmbedder  # noqa: E402
from lightcore.rag.pipeline import RAGPipeline  # noqa: E402

CORPUS_DIR = _HERE / "corpus"
OUT = _HERE / "cases" / "golden_real.jsonl"

PAGE_W, PAGE_H = 595.0, 842.0
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 60.0, 60.0, 60.0
FS_TITLE, FS_H, FS_BODY = 15.0, 12.0, 10.5
LINE_GAP, CHARS_PER_LINE = 1.45, 44

# --------------------------------------------------------------------------------------
# 语料：每篇 = 标题 + 若干"段落"（段落里的句子就是答案的可能出处）
# 刻意混入 表格行 / 公式行 / 英文段 / 编号列表，贴近真实论文的文本形态。
# --------------------------------------------------------------------------------------

CORPUS: dict[str, dict] = {
    "pm-rag": {
        "title": "面向本地文献库的混合检索增强生成系统",
        "paragraphs": [
            "摘要：本文面向本地优先的单用户文献研读场景，提出一套混合检索增强生成方案，"
            "在不依赖外部服务的前提下提升论文问答的事实一致性。",
            "1 引言。学术文献的问答对证据定位要求很高，纯生成模型容易给出文档中并不存在的结论，"
            "因此检索质量直接决定了最终回答的可信度。",
            "2 分块策略。系统按语义边界把正文切分为三百到五百字的块，"
            "过小的块会丢失上下文，过大的块会稀释检索信号。",
            "3 检索通道。系统同时使用三条召回通道：稠密向量检索负责语义相似，"
            "BM25 负责关键词命中，稀疏向量兼顾两者，其中 BM25 使用 jieba 按词切分以适应中文。",
            "4 融合策略。三路召回结果通过 RRF 融合为统一排序，"
            "融合常数取 60，该值越大则排名差异被压得越平缓。",
            "5 重排。融合后的候选交给交叉编码器重排，"
            "重排模型同时读取查询与候选文本，因此精度高于仅比较向量的方式，但耗时显著更长。",
            "6 性能观测。实测重排阶段占单次检索总耗时的百分之九十七，是唯一的性能瓶颈；"
            "把候选数从五十降到二十后，前五名结果完全一致，而重排耗时下降约六成。",
            "表 1 各阶段耗时（单位：毫秒）。编码 47 ；混合召回 58 ；重排 1444 ；合计 1549。",
            "7 缓存。系统对查询结果做容量为 32 的先进先出缓存，"
            "在索引发生变更时整体失效，避免旧结果污染新索引。",
            "8 失败语义。系统严格区分两种情况：检索服务不可用属于系统故障，"
            "而检索结果为空表示文档中确实没有相关内容，两者对用户的提示完全不同。",
            "9 结论。混合检索配合重排可以在本地设备上取得可用的检索质量，"
            "但必须以真实文档构造的评测集来验证，否则指标不具备参考价值。",

            # ↓↓↓ 2026-09-26 追加（T-04-B）：原 11 段一字未改，仅在**末尾追加**段落。
            # 目的：原语料只有 1 个 chunk → precision@5 的分母被压成 1（18 例分母塌缩）。
            # 追加在末尾可保证既有分块与 chunk id 不变（分块按顺序处理），旧 gold 仍可定位。
            "10 评测协议。评测集应当由真实文档生成，并覆盖逐字命中、改写提问、"
            "多证据与负样本四类情形。只测其中一类会让指标看起来很好，却无法反映真实使用中的表现。",
            "11 冗余与去重。多路召回天然会带来重叠结果，若不去重，"
            "同一段文本可能占据前五名中的多个位置，等效于减少了实际可用的证据数量。",
            "12 引用溯源。每个分块都必须保留页码与文档标识，"
            "否则答案无法回溯到出处，用户也无法判断引用是否与原文一致。",
            "13 索引更新。文档新增或删除后应当增量更新索引，"
            "而不是整体重建；整体重建在文献库较大时会产生明显的等待时间，"
            "并且在重建过程中无法提供检索服务。",
            "14 中英混排。学术文献普遍存在中英混排现象，分词器必须同时处理两种语言："
            "按字符切分会让英文单词失去边界，按空格切分则会把整段中文视为一个词。",
            "15 成本模型。检索成本由三个部分构成：查询编码、混合召回与重排。"
            "其中查询编码的耗时与文档规模无关，属于固定成本；"
            "重排的耗时与候选数量成正比，是唯一可以通过裁剪候选显著降低的部分。",
            "16 本地化与隐私。系统被设计为完全本地运行，"
            "文档内容与检索请求均不出本机，这让它在处理未公开稿件时具备可用性；"
            "代价是所有计算都必须在本机设备上完成。",
            "17 失败恢复。检索链路的任何一环失败都应当被明确上报，"
            "而不是静默返回空结果；用户需要区分系统故障与文档中确实没有答案这两种情况。",
            "18 局限性。本方案依赖交叉编码器重排，"
            "因此在候选规模较大时延迟会显著上升；此外对表格与公式的处理仍然有限，"
            "这些内容在抽取阶段容易被压成一维文本而丢失结构。",
            "19 相关工作。与端到端的生成式问答相比，本方案把证据定位与答案生成分离，"
            "牺牲了一定的灵活性，换来的是可核查的引用与更稳定的事实一致性。",
            "20 未来工作。下一步计划引入文档结构感知的分块策略，"
            "并把评测集扩展到表格与图注等结构化内容，使质量评估覆盖更多真实形态。",
            "21 分块边界的实证。按固定长度切分会让跨段落的论证被拆散，"
            "而完全按段落切分又会让短段落稀释检索信号；折中做法是设定目标长度并在句子边界收口，"
            "使绝大多数块落在目标长度的邻域内。",
            "22 融合常数的敏感性。融合常数决定排名差异被压缩的程度："
            "取值过小时，单通道的偶然高分会被放大；取值过大时，各通道的排名差异失去意义。"
            "实测表明在候选规模较小时，结果对这一常数并不敏感。",
            "23 候选裁剪的位置。裁剪发生在重排之前还是之后，决定了省下的是哪一部分成本："
            "放在重排之前可以显著降低延迟，代价是可能丢掉需要深层比对才能识别的证据。",
            "24 长文档的召回衰减。当单篇文档的块数远超候选上限时，"
            "同一文档内部会形成竞争，正确块可能被同文档的相似块挤出前五名，"
            "这也是多证据提问比单证据提问更容易漏检的原因。",
            "25 查询改写的作用与代价。把口语化提问改写为检索式可以提升召回，"
            "但改写本身会引入一次模型调用，并且可能偏离用户原意；"
            "在问题已经足够具体时，改写的收益通常不足以抵消其成本。",
            "26 向量维度与存储。稠密向量的维度直接决定索引体积与检索时的内存占用，"
            "在单机部署中需要在召回质量与常驻内存之间取舍。",
            "27 索引一致性。文档被删除后，其向量与分块必须一并清理，"
            "否则检索会命中已经不存在的内容，表现为引用指向空文档。",
            "28 检索与生成的边界。检索只负责提供证据，生成只负责组织语言；"
            "把事实判断交给生成模型会引入无法核查的内容，因此系统提示词被明确限定为只能依据资料回答。",
            "29 可复现性。同一份评测集在相同配置下重跑，检索结果应当完全一致；"
            "若出现差异，说明某处引入了不确定性，而这类不确定性会让跨版本的指标比较失去意义。",
            "30 结束语。上述工程取舍的共同点是：先用可复现的评测确认问题确实存在，"
            "再投入改动，并在改动后重新测量同一份评测集，而不是凭直觉调整参数。",
            "31 阈值与度量的配套。任何阈值都只在其度量口径下有意义："
            "更换估算方式之后，原有的阈值会静默地改变含义，因此二者必须成对调整并同时记录。",
            "32 负样本的价值。只统计找到了多少证据会掩盖一个方向的问题，"
            "即系统在资料不足时是否如实说明；负样本的存在使这一维度可被测量，"
            "没有它，答案层的安全边界就没有证据。",
            "33 判官与生成模型的分离。用同一模型既生成答案又评判答案，"
            "会让评判倾向于认可自己的表述，因此评判应当由不同来源的模型完成，"
            "并用人工标注校验评判本身是否可靠。",
            "34 一致性指标的用法。一致性指标衡量的是超出随机的一致程度，"
            "当两侧判定都没有区分度时该指标无定义；这时应当报告错误率及其置信上界，"
            "而不是把算不出的指标写成一个数字。",
            "35 样本设计的先行条件。在给指标设定目标之前，"
            "必须先确认样本中确实存在可能被判错的案例，否则无论系统表现如何，"
            "得到的都只会是一个没有分辨力的结果。",
            "36 指标上限的核算。精度类指标的上限由标注中的相关结果数量与截断位次共同决定；"
            "当每条提问只有一个相关块时，截断位次越大，精度上限越低，"
            "此时应当改用首中率或排序质量类的指标来反映真实表现。",
            "37 分母塌缩。当候选池的规模小于截断位次时，精度的分母会随之变小，"
            "使指标看起来偏高；因此评测语料的规模必须保证候选池大于截断位次，"
            "否则指标无法与更大语料下的结果比较。",
            "38 结束语（补充）。可复现、可审计、可比较，是评测体系的三个最低要求；"
            "三者缺一时，指标数字本身就会变成一种误导。",
            "39 评测与生产的同源。评测所用的检索路径必须与生产一致，"
            "否则测的是另一套系统；组合层、包装层与运行期开关都要在评测中显式走同一条路径。",
            "40 改动前后的对照方法。评估一项改动时，两侧应当只在被考察的那一项上不同，"
            "其余条件保持一致；同时改动多项会让结论无法归因，"
            "即便指标变好也无法判断是哪一项起了作用。",
            "41 记录与留痕。每一次评测的数值、口径与代码版本都应写入报告，"
            "使事后可以回答这个数字是在什么条件下得到的；缺少口径的数字无法与后续版本比较。",
            "42 结束语（终）。本文所述的方法并不追求更高的指标数值，"
            "而是追求每一个数字都能被追溯、被复核、被比较。",
        ],
    },
    "agent-loop": {
        "title": "A Minimal Reasoning-Acting Loop for Tool Use",
        "paragraphs": [
            "Abstract. This paper describes a minimal reasoning-acting loop that alternates "
            "between deliberation and tool invocation without relying on external orchestration frameworks.",
            "1 Introduction. Most agent frameworks bolt a planner on top of a chat model. "
            "We argue that a single loop with strict resource guards is easier to reason about.",
            "2 States. The loop exposes five observable states: think, act, observe, finish and abort, "
            "and every transition is emitted as an event so that the caller can render progress.",
            "3 Guards. Four guards keep the loop bounded: a maximum iteration count, "
            "a maximum number of consecutive tool failures, a step budget without progress, "
            "and a per-turn monetary cost ceiling.",
            "4 Tool timeout. A tool that hangs would otherwise stall the whole turn, "
            "so each invocation is wrapped in a timeout; the timeout is deliberately loose "
            "because legitimate tools may take tens of seconds.",
            "5 Cancellation. User cancellation must not be recorded as a tool failure, "
            "so the loop only catches timeout errors and lets cancellation propagate untouched.",
            "6 Finalization. When the budget is exhausted the loop still produces an answer: "
            "it calls the model with no tools at all, which structurally guarantees convergence.",
            "Table 1 Default configuration. iterations 12 ; tool failures 3 ; token budget 8000 ; timeout 120 s.",
            "7 Self reflection. On an aborted turn only, the loop may ask the model to critique "
            "its own draft and regenerate once with the critique attached.",
            "8 Conclusion. Bounded loops with explicit failure semantics are preferable to "
            "unbounded autonomy, especially for single-user local deployments.",

            # ↓↓↓ 2026-09-26 追加（T-04-B）：原段落一字未改，仅在末尾追加
            "9 Observability. Every iteration emits a structured event with the current step index, "
            "the tool that is about to run and the reason the loop selected it. Without these events "
            "a stalled turn is indistinguishable from a slow one.",
            "10 Context budget. The prompt sent to the model grows with every tool result, "
            "so the loop projects the next request size before issuing it and drops the oldest "
            "tool outputs when the projection crosses a trigger line.",
            "11 Token accounting. Character counts are a poor proxy for tokens because the ratio "
            "differs by content type; the loop therefore uses the provider reported usage whenever "
            "it is available and only falls back to estimation for the first request of a turn.",
            "12 Tool result shaping. A tool may return far more text than the model needs. "
            "Truncation is applied to the payload rather than by removing messages, "
            "because removing a message would break the pairing between a call and its result.",
            "13 Idempotence. Tools that mutate state must be safe to retry, "
            "since a timeout gives the loop no way to know whether the effect already happened. "
            "Read-only tools should be preferred whenever the task allows it.",
            "14 Failure taxonomy. The loop distinguishes a tool that failed, a tool that timed out "
            "and a tool that returned nothing useful. Each of these leads to a different next step, "
            "and conflating them produces loops that retry hopelessly.",
            "15 Evaluation. Loop behaviour is exercised with scripted tool responses so that the "
            "control flow is tested deterministically, while the token accounting is measured "
            "against a real provider to keep the two layers honest.",
            "16 Limitations. The loop assumes a single writer; concurrent turns on the same session "
            "would make the step budget and the context projection ambiguous.",
            "17 Prompt stability. The loop keeps the system prompt fixed across iterations and only "
            "appends observations, so that any change in behaviour can be attributed to the new "
            "evidence rather than to a rewritten instruction.",
            "18 Cost ceiling. A per-turn monetary ceiling is enforced before each request; "
            "when it is reached the loop finalises instead of continuing, "
            "which makes the worst case predictable for the caller.",
            "19 Progress detection. Counting iterations is not the same as counting progress. "
            "The loop tracks steps that produced no new information and stops when they accumulate, "
            "because a loop that keeps trying without learning is indistinguishable from a hang.",
            "20 Tool selection. Tools are described to the model with the same names they are "
            "invoked by; a mismatch between the two is a silent failure that only shows up "
            "as the model repeatedly asking for a tool that cannot be resolved.",
            "21 Streaming. Intermediate tokens are streamed to the caller while tools run, "
            "so that a long turn does not look like a stalled one, "
            "but the stream must never leak the raw tool output.",
        ],
    },
    "mixed-long": {
        "title": "长文改稿中的格式保真与段落寻址",
        "paragraphs": [
            "摘要：针对 Word 长文的机器辅助改稿，本文讨论段落级寻址、修订留痕与格式保真三件事，"
            "并给出在本地实现的工程取舍。",
            "1 段落寻址。系统为每个正文段落分配稳定编号，写作 P 加序号，"
            "模型只能通过这些编号提出修改，避免出现无法定位的自由文本。",
            "2 修订留痕。所有改动以插入与删除两种修订形式写入文档，"
            "用户在 Word 中可以直接接受或拒绝，而不必对照两份文件。",
            "3 格式继承。最初的实现把整段原文标记为删除、再追加一段新文本，"
            "导致整段退化为默认字体与字号；修正后改为逐段运行块对齐，"
            "把变化部分单独包裹为修订，未变化部分原样保留。",
            "4 Rules for nested structures. When a paragraph contains nested elements such as "
            "insertions inside hyperlinks, the implementation falls back to whole paragraph "
            "replacement, because partial rewriting there would corrupt the markup.",
            "5 表格与页码。表格内的段落使用独立编号前缀，"
            "而缺少物理页码的文档在引用核验时只报可疑项，不报弱提示，避免提示疲劳。",
            "公式 1 保真度定义：保真度等于未被改动的运行块数量除以运行块总数。",
            "表 2 操作类型与语义。替换 replace ；追加 append ；后插 insert_after ；删除 delete 。",
            "6 原子写入。文档保存采用先写临时文件再原子替换的方式，"
            "并保留最近五份备份，防止写入中断导致文件损坏。",
            "7 结论。把改稿约束在段落级、并把格式保真作为一等目标，"
            "可以让机器改稿的结果直接进入人工审阅流程。",

            # ↓↓↓ 2026-09-26 追加（T-04-B）：原段落一字未改，仅在末尾追加
            "8 无效回复机制。模型有时会返回格式正确但不包含任何修改意见的内容，"
            "这类回复必须被识别并记为空转，而不是当作一次成功的改稿计入进度。",
            "9 差异对照。系统为用户同时提供图文视图与结构视图："
            "前者用于阅读成稿效果，后者用于逐段核对改动来源，两者共享同一份底层数据。",
            "10 批量改写的边界。一次性提交过多段落会让模型的注意力分散，"
            "实测表明按章节分批提交时保真度更稳定，代价是需要保存中间状态。",
            "11 样式表处理。文档样式表与正文是分离的，若替换段落时未继承原段落的样式引用，"
            "即使文本正确也会在渲染时出现字体跳变，因此样式引用必须随运行块一并处理。",
            "12 图片与题注。图片本身不参与文本改写，但其题注编号需要与正文引用保持一致；"
            "改写正文后应当重新校验编号，否则会出现图文不一致。",
            "13 修订冲突。当同一段落被连续两次改写时，第二次改写必须基于第一次的结果，"
            "而不是原始文本，否则前一版的修订会被静默覆盖。",
            "14 引用核验。系统对引用逐一核对页码与段落是否存在，"
            "对无法核验的引用单独列出可疑项交人工确认，不做自动删除。",
            "15 性能取舍。整篇文档的运行块扫描是最耗时的环节，"
            "因此在改写前先计算受影响范围，只扫描相关段落，可以显著降低等待时间。",
            "16 局限性。当前实现依赖段落编号的稳定性，"
            "若用户在外部编辑文档中插入或删除段落，编号会整体错位，需要重新建立寻址。",
            "17 修订的可见性。插入与删除两种修订在 Word 中呈现方式不同，"
            "删除内容默认以删除线显示而不消失，这让用户能够确认被移除的原文，"
            "因此在实现中不应当用真正的删除操作来替代删除修订。",
            "18 段落合并与拆分。改稿请求有时要求合并两段或把一段拆开，"
            "这类操作会改变后续所有段落的编号，因此必须在完成后重新编号并刷新寻址表，"
            "否则后续引用会指向错误的段落。",
            "19 中英混排的处理。文档中存在中英混排时，字数统计与显示宽度并不一致，"
            "若按字符数限制改写长度，英文段落会被切得过短，"
            "因此长度约束应当按语言分别设定。",
            "20 脚注与尾注。脚注编号与正文引用相互依赖，"
            "改写正文后若不同步调整脚注顺序，会出现编号跳号，故脚注被排除在自动改写范围之外。",
            "21 审阅轮次。改稿通常不是一次性完成的，系统保存每一轮的修订与批注，"
            "使得用户可以回到任意一轮继续，而不是只能接受或拒绝最终结果。",
            "22 段落级最小改动原则。改写应当只触碰必要的运行块，"
            "把整段重写的做法虽然实现简单，但会把作者原有的措辞习惯一并抹掉，"
            "这也是格式保真指标被单独提出的原因。",
            "23 空转识别与提示。当模型连续返回不含任何修改的回复时，"
            "系统给出明确的提示而不是继续重试，因为这类空转通常意味着请求本身缺少可操作的指令。",
            "24 模板与样式继承。使用模板创建的文档带有预设样式，"
            "改写时若直接套用正文默认样式，会让标题与正文的层级关系丢失，"
            "因此必须读取并沿用原段落的样式标识。",
            "25 结束语。改稿系统的价值不在于改得更多，而在于让每一处改动都可定位、可撤销、可核对，"
            "这三点决定了它能否真正进入人工审阅流程。",
            "26 编号稳定性与重编号策略。文档结构变化后，最稳妥的做法是保留旧编号到新编号的映射，"
            "使既有批注仍能指向正确位置，而不是简单地从当前顺序重新编号。",
            "27 批注与修订的区分。批注用于提出疑问，修订用于实际改动，"
            "两者在文档中承担不同职责；把疑问直接写成修订会让用户误以为内容已被修改。",
            "28 段落属性的保留。段落具有对齐、缩进、行距等属性，"
            "改写时若只替换文本而不复制属性，视觉上会出现突兀的变化，"
            "因此属性必须与运行块一同处理。",
            "29 超长段落的处理。超过一定长度的段落会超出模型的稳定处理范围，"
            "实践中按语义切分为若干子段分别处理，再合并结果并保持原有段落边界不变。",
            "30 表格内改写的限制。表格单元格内的文本改写会牵动列宽与对齐，"
            "因此默认只处理纯文本替换，不调整表格结构，结构层面的调整交由人工完成。",
            "31 文档打开状态的冲突。文档被外部程序占用时写入会失败，"
            "系统检测这一状态并提示用户关闭，而不是反复重试或静默放弃。",
            "32 备份与回滚。每次写入前生成备份，并允许用户回滚到任意一次备份；"
            "原子写入只能保证文件不损坏，不能保证语义上符合预期，因此两者都需要。",
            "33 结束语（补充）。段落寻址、修订留痕与格式保真三者互相依赖："
            "缺少任何一个，机器改稿的结果都无法直接进入人工审阅。",
            "34 人工复核的定位。机器改稿的输出应当被定位为候选，"
            "而不是结论；系统的职责是降低人工核对的成本，而不是替代判断。",
            "35 请求的可操作性。改稿请求若只给出模糊方向，模型难以产出可用结果；"
            "实践中要求请求至少指明目标段落与期望变化，否则宁可提示用户补充，也不猜测意图。",
            "36 术语一致性。文档中的专业术语在改写时不应被同义替换，"
            "因此需要维护术语表并在改写时保护其中的词条，避免引入不一致的表述。",
            "37 结束语（终）。把改稿约束在可定位、可撤销、可核对的范围内，"
            "机器产出的每一处改动才具备进入正式审阅流程的资格。",
            "38 修订密度的控制。一份文档中修订过于密集会让人工审阅难以进行，"
            "因此系统对单次提交的段落数量设定上限，并提示分批进行。",
            "39 与检索的配合。改稿请求常常需要先定位证据所在的段落，"
            "此时检索提供候选段落，模型再在此范围内提出修改，"
            "把开放式的改写收敛为有依据的局部改动。",
            "40 输出的可核查性。每一处改动都应附带改动理由与依据段落，"
            "缺少理由的改动即便文本更通顺，也会增加人工核对的成本。",
            "41 结束语（终补）。可定位、可撤销、可核对，"
            "既是改稿系统的设计约束，也是评价其输出是否可用的三条判据。",
        ],
    },
}

# --------------------------------------------------------------------------------------
# 问答标注：(id, query, doc_id, 答案原文片段（必须在正文里逐字出现）, 参考要点, 关键词)
# 关键词命中但非 gold 的块 → gain=2（强相关），用于 nDCG 分级相关性。
# --------------------------------------------------------------------------------------

QA: list[tuple[str, str, str, str, str, list[str]]] = [
    ("r01", "系统把正文切成多大的块", "pm-rag",
     "三百到五百字的块", "分块大小为 300–500 字", ["分块", "块"]),
    ("r02", "BM25 中文是怎么分词的", "pm-rag",
     "BM25 使用 jieba 按词切分以适应中文", "BM25 用 jieba 按词切分", ["BM25", "jieba"]),
    ("r03", "RRF 的融合常数取多少", "pm-rag",
     "融合常数取 60", "RRF k=60", ["RRF", "融合"]),
    ("r04", "重排为什么比向量检索更准", "pm-rag",
     "重排模型同时读取查询与候选文本", "交叉编码器同时看 query 与候选", ["重排"]),
    ("r05", "单次检索的性能瓶颈在哪里", "pm-rag",
     "重排阶段占单次检索总耗时的百分之九十七", "重排占 97% 耗时", ["重排", "耗时"]),
    ("r06", "为什么检索结果为空和检索失败要分开报", "pm-rag",
     "检索服务不可用属于系统故障", "系统故障 ≠ 空结果", ["失败", "检索"]),
    ("r07", "循环有哪几个状态", "agent-loop",
     "think, act, observe, finish and abort", "五个状态", ["states", "loop"]),
    ("r08", "工具调用超时值为什么设得比较宽松", "agent-loop",
     "the timeout is deliberately loose", "避免误伤正常慢调用", ["timeout"]),
    ("r09", "预算耗尽后还会给出回答吗", "agent-loop",
     "it calls the model with no tools at all", "tools=None 机制上保证收敛", ["finalization", "budget"]),
    ("r10", "段落寻址用什么形式", "mixed-long",
     "写作 P 加序号", "P{i} 段落编号", ["段落", "寻址"]),
    ("r11", "为什么早期版本会把整段格式改掉", "mixed-long",
     "导致整段退化为默认字体与字号", "整段退化为默认格式", ["格式", "修订"]),
    ("r12", "保真度是怎么定义的", "mixed-long",
     "保真度等于未被改动的运行块数量除以运行块总数", "未改动 run / 总 run", ["保真", "运行块"]),
]

#: 改写型难例（id, query, doc_id, 答案原文片段, 参考要点, 关键词）
#: 与 QA 的区别：**query 刻意不与答案共享关键词**（同义改写 / 跨语言），
#: 词法通道（BM25）帮不上忙，必须靠稠密语义 —— 这是"同词提问"测不出来的能力。
PARAPHRASE_QA: list[tuple[str, str, str, str, str, list[str]]] = [
    ("p01", "服务自己出毛病，和文档里根本没写，给用户的提示一样吗", "pm-rag",
     "检索服务不可用属于系统故障", "系统故障与空结果必须区分", ["故障"]),
    ("p02", "如果某个工具一直不返回结果，整个流程会怎样", "agent-loop",
     "so each invocation is wrapped in a timeout", "跨语言：中文问英文文档", ["timeout"]),
    ("p03", "改稿时怎么保证不会把文件写坏", "mixed-long",
     "文档保存采用先写临时文件再原子替换的方式", "原子写入 + 备份", ["写入"]),
]

#: 多 gold 用例：(id, query, 答案所在文档, 干扰文档, [答案片段1, 答案片段2, ...], 参考要点, 关键词)
#: 用途：一次提问需要**两处证据**才能回答 → recall@k 可能 < 1，指标才有分辨力
#: （单 gold 用例下 recall 恒为 0 或 1，饱和严重）。
MULTI_GOLD_QA: list[tuple[str, str, str, list[str], list[str], str, list[str]]] = [
    # T-02（2026-09-25）：原 g01/g02 被移除。原因：multi_gold 的口径要求"每条 ≥2 个 gold chunk"，
    # 而 pm-rag 与 mixed-long **各只有 1 个 chunk** → 结构上不可能满足（见 test_golden_real.py 的
    # test_hard_case_kinds_present）。这两条已原样归档在 cases/golden_real_25cases.jsonl.bak-20260925。
    # 替代用例（g03~g24）建在多 chunk 文档上，见 eval/corpus_specs_t02.py。
]

#: 多文档用例：(id, query, 答案所在文档, 同时加载的干扰文档, 答案原文片段, 参考要点, 关键词)
#: 用途：验证"同时挂多篇文献时，证据没有被干扰文档挤掉"（单文档用例测不出这一类退化）。
MULTI_QA: list[tuple[str, str, str, list[str], str, str, list[str]]] = [
    ("m01", "分块大小是怎么定的", "pm-rag", ["agent-loop"],
     "三百到五百字的块", "干扰文档在场时仍应命中 pm-rag 的分块段", ["分块"]),
    ("m02", "段落编号是怎么分配的", "mixed-long", ["pm-rag"],
     "写作 P 加序号", "干扰文档在场时仍应命中 mixed-long 的寻址段", ["段落"]),
    ("m03", "What happens when the budget is exhausted?", "agent-loop", ["mixed-long"],
     "it calls the model with no tools at all", "英文查询 + 中文干扰文档", ["budget", "finalization"]),
]

#: 负样本：语料中**确实没有**答案的问题（用于 L0-3 判"是否如实拒答"）
NEGATIVES: list[tuple[str, str]] = [
    ("n01", "这套系统的许可证费用是多少"),
    ("n02", "作者在哪个会议上发表了这篇文章"),
    ("n03", "论文里报告的 GPU 型号是什么"),
    ("n04", "What is the exact learning rate used in pretraining?"),
    ("n05", "实验用的数据集一共有多少条样本"),
]


# --------------------------------------------------------------------------------------
# T-02 扩容（2026-09-25）：合并新增语料与配额扩样
#
# **为什么不直接改上面的列表**：既有 25 条用例的 `gold` 引用 3 篇原文的 chunk id，
# 正文改一个字就可能让 gold 失效（`locate()` 会 FATAL，但"改没改过"这件事本身不该靠运气）。
# 因此新增内容独立成 `corpus_specs_t02.py`，这里只做**合并** ——
# 原文不动、增量可单独审阅、也便于分批写入。
#
# ⚠️ 两处必须知道的口径后果：
# 1. 合并后**用例集指纹变化** → 既有 `eval/report_real.json` 立刻被视为过期
#    （`tests/test_eval_baseline.py` 的指纹新鲜度门禁会报红）→ 必须重跑 `run_eval`；
# 2. 负样本的 `corpus` 取 `list(CORPUS)`（全部文档）→ 合并后负样本的"不可答"是针对
#    **整个扩容后的语料**而言，因此新负样本必须对所有文档都不可答。
# --------------------------------------------------------------------------------------
from corpus_specs_t02 import (  # noqa: E402  同目录模块（脚本/测试都在 eval/ 下运行）
    CORPUS_T02,
    MULTI_DOC_T02,
    MULTI_GOLD_T02,
    NEGATIVES_T02,
    PARAPHRASE_T02,
    VERBATIM_T02,
)

CORPUS.update(CORPUS_T02)
QA.extend(VERBATIM_T02)
PARAPHRASE_QA.extend(PARAPHRASE_T02)
MULTI_GOLD_QA.extend(MULTI_GOLD_T02)
MULTI_QA.extend(MULTI_DOC_T02)
NEGATIVES.extend(NEGATIVES_T02)


def wrap(text: str, per_line: int) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= per_line:
            lines.append(cur)
            cur = ""
    if cur:
        lines.append(cur)
    return lines


def write_pdf(path: Path, title: str, paragraphs: list[str]) -> None:
    """把标题与段落写成**真实 PDF 文件**（多页），供生产链路解析。

    ⚠️ 中文字体用 PyMuPDF **内置** 的 `china-s`（简体），不要 `insert_font(fontfile=simhei.ttf)`：
    simhei 本体 9.7MB，嵌入后单篇 PDF 就是 9.7MB（3 篇近 30MB）；
    内置字体的文本走 CMap，文件只有 KB 级，且 `get_text()` 抽取结果一致（实测对比过）。
    """
    doc = pymupdf.open()
    state = {"page": doc.new_page(width=PAGE_W, height=PAGE_H), "y": MARGIN_TOP}

    def new_page() -> None:
        state["page"] = doc.new_page(width=PAGE_W, height=PAGE_H)
        state["y"] = MARGIN_TOP

    def put(text: str, size: float, gap: float) -> None:
        """写一行；触底换页。中文/英文统一用内置字体（无外部依赖）。"""
        if state["y"] > PAGE_H - MARGIN_BOTTOM:
            new_page()
        state["page"].insert_text((MARGIN_X, state["y"]), text, fontsize=size, fontname="china-s")
        state["y"] += size * gap

    put(title, FS_TITLE, LINE_GAP)
    state["y"] += 8
    for para in paragraphs:
        for line in wrap(para, CHARS_PER_LINE):
            put(line, FS_BODY, LINE_GAP)
        state["y"] += 4  # 段间距
    doc.save(str(path))
    doc.close()


def _norm(s: str) -> str:
    """归一化：去掉全部空白字符。

    必要性：PDF 抽取会把长句按行折断并插入换行，而分块与引用里是连续文本；
    不做归一化则"答案片段"永远匹配不上（这是"合成集"从未暴露的问题）。
    """
    return re.sub(r"\s+", "", s)


def build_corpus(out_dir: Path | None = None) -> dict[str, list]:
    """生成 PDF，并用生产同款链路解析出真实 chunk（返回 doc_id → chunk 列表）。

    ★ `out_dir` 是给**测试**用的（2026-10-05 修）：此前固定写 `CORPUS_DIR`，
    而 PyMuPDF 每次 `doc.save()` 都会刷新 PDF 的 `/CreationDate` 与 `/ID`
    ⇒ **跑一次测试就把仓库里 11 个语料 PDF 改写**（大小相同、约 57 字节不同），
    于是 `git status` 每次跑完门禁都变脏，极易把噪声误提交。
    默认仍是 `CORPUS_DIR`（生成脚本 `main()` 的行为不变）。
    """
    out_dir = out_dir or CORPUS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_by_doc: dict[str, list] = {}
    for doc_id, spec in CORPUS.items():
        pdf = out_dir / f"{doc_id}.pdf"
        write_pdf(pdf, spec["title"], spec["paragraphs"])
        # 真实链路：与线上完全同一套 DocumentParser + Chunker（RAGPipeline 的默认装配）
        pipe = RAGPipeline(embedder=FakeHashEmbedder())
        n = pipe.add_document(pdf, doc_id=doc_id)
        chunks = list(pipe._chunks)  # noqa: SLF001  管线未暴露公开访问器；下方有数量断言兜底
        if n != len(chunks) or n == 0:
            raise SystemExit(f"[FATAL] {doc_id}: add_document 返回 {n} 与内部块数 {len(chunks)} 不一致")
        chunks_by_doc[doc_id] = chunks
        print(f"  {doc_id}.pdf  {pdf.stat().st_size:>7} 字节  →  {n} 个真实 chunk")
    return chunks_by_doc


def locate(chunks: list, span: str) -> list[str]:
    """在真实 chunk 里定位答案原文片段，返回命中的 chunk id 列表。"""
    needle = _norm(span)
    return [c.id for c in chunks if needle and needle in _norm(c.text)]


def main() -> None:
    # T-02（2026-09-25）：支持 `--out`，便于**先验证、后替换**——
    # 直接覆盖权威集 `cases/golden_real.jsonl` 会立刻让 `report_real.json` 的指纹过期、
    # 一键门禁转红；扩容批次应先在临时文件上验证（定位全绿、分块规模合理）再切换。
    import argparse

    global OUT

    ap = argparse.ArgumentParser(description="生成/更新真实文档评测集")
    ap.add_argument("--out", default=str(OUT),
                    help=f"输出路径（默认 {OUT.relative_to(_HERE.parent)}）")
    args = ap.parse_args()

    OUT = Path(args.out)

    print("① 生成真实 PDF 并走生产链路分块：")
    chunks_by_doc = build_corpus()

    print("② 定位答案片段（gold 由真实 chunk 产出，非手工指定）：")
    rows: list[dict] = []
    failures: list[str] = []

    def make_row(cid: str, query: str, corpus: list[str], primary: str, spans: list[str],
                 ref: str, keywords: list[str], kind: str) -> dict | None:
        """spans 是**多个**答案片段：每个都必须能定位到真实 chunk（否则整条用例作废）。"""
        chunks = chunks_by_doc[primary]
        gold: list[str] = []
        for span in spans:
            hit = locate(chunks, span)
            if not hit:
                failures.append(f"{cid}: 在 {primary} 的真实 chunk 中找不到答案片段「{span}」")
                return None
            for h in hit:
                if h not in gold:
                    gold.append(h)
        gains = {g: 3 for g in gold}
        for c in chunks:  # 同主题（命中关键词）但非 gold → gain 2
            if c.id not in gains and any(k and k in c.text for k in keywords):
                gains[c.id] = 2
        label = f"[{'+'.join(corpus)}]"
        print(f"  {cid}  gold={gold}  {label}  {kind}  分级={sorted(gains.values(), reverse=True)}")
        return {
            "id": cid, "query": query, "corpus": corpus, "doc_id": primary,
            "answer_sentence": spans[0] if len(spans) == 1 else spans,
            "answerable": True, "kind": kind,
            "gold": gold, "gains": gains, "reference": ref,
        }

    for cid, query, doc_id, span, ref, keywords in QA:
        row = make_row(cid, query, [doc_id], doc_id, [span], ref, keywords, "verbatim")
        if row:
            rows.append(row)

    for cid, query, doc_id, span, ref, keywords in PARAPHRASE_QA:
        row = make_row(cid, query, [doc_id], doc_id, [span], ref, keywords, "paraphrase")
        if row:
            rows.append(row)

    for cid, query, primary, others, spans, ref, keywords in MULTI_GOLD_QA:
        row = make_row(cid, query, [primary, *others], primary, spans, ref, keywords, "multi_gold")
        if row:
            rows.append(row)

    for cid, query, primary, others, span, ref, keywords in MULTI_QA:
        row = make_row(cid, query, [primary, *others], primary, [span], ref, keywords, "multi_doc")
        if row:
            rows.append(row)

    for nid, query in NEGATIVES:
        rows.append({
            "id": nid,
            "query": query,
            "corpus": list(CORPUS),
            "doc_id": "",
            "answer_sentence": "",
            "answerable": False,
            "kind": "negative",
            "gold": [],
            "gains": {},
            "reference": "（语料中不存在答案，期望如实拒答）",
        })
        print(f"  {nid}  负样本（语料无答案）  [{'/'.join(CORPUS)}]")

    if failures:
        raise SystemExit("[FATAL] 以下用例的答案片段未能定位到真实 chunk（评测集会失去意义）：\n  - "
                         + "\n  - ".join(failures))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()[:12]
    by_kind: dict[str, int] = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    print(f"\n③ 已写入 {OUT}")
    print(f"   共 {len(rows)} 条：" + " / ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
    print(f"   sha256[:12]={digest} · 语料 {CORPUS_DIR}（{len(CORPUS)} 篇真实 PDF）")


if __name__ == "__main__":
    main()
