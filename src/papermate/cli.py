"""PaperMate · 最小命令行对话（M4 应用层前置验证入口）

把 M1-M3（Agent 循环 / 工具 / RAG）串成可在终端逐轮对话的入口：
- 默认 FakeLLM（无 key），演示 ReAct 检索→回答轨迹；
- --model deepseek/openai/qwen：在 .env 配置对应 key 后接真实模型自由对话；
- --doc path.txt：加载论文纯文本构建 RAG 管线，retrieve 走真实检索（--embedder real 用 BGE-M3）。

已接入 M2 记忆（MemoryManager + JSON 持久化，跨会话召回）与 M1.2 联网能力
（大模型原生联网搜索，需出网）。
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import os
import sys
from pathlib import Path

from .prompts import SYSTEM_PROMPT

# 防御性：确保 src 在 sys.path（部分调用方式下 -m 已自动加入，此处兜底）
_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from lightcore.core.loop import AgentLoop, LoopConfig  # noqa: E402
from lightcore.llm.base import ChatMessage, Role, ToolCall  # noqa: E402
from lightcore.llm.fake import FakeLLM  # noqa: E402
from lightcore.llm.router import PRESET_ENDPOINTS, ModelRouter  # noqa: E402
from lightcore.memory.manager import MemoryManager  # noqa: E402
from lightcore.memory.store import JSONFileMemoryStore  # noqa: E402
from lightcore.rag.embedder import FakeHashEmbedder  # noqa: E402
from lightcore.tools.builtin.basic import register_builtin_tools  # noqa: E402
from lightcore.tools.spec import Permission, ToolExecutor, ToolRegistry  # noqa: E402

try:  # rich 已随依赖安装；缺失时降级为纯 print
    from rich.console import Console
    from rich.markdown import Markdown

    _console = Console()
    _HAS_RICH = True
except Exception:  # noqa: BLE001  # pragma: no cover  # 可选依赖 rich 缺失则降级纯文本
    _HAS_RICH = False


class CliFakeLLM(FakeLLM):
    """CLI 用的 FakeLLM：用用户真实问题作为检索 query，并把工具结果整理成可读摘要。

    仅用于无 key 的本地演示，不影响全局 FakeLLM 剧本（spike_s1 等测试不受影响）。
    """

    def __init__(self, top_k: int = 3, model_name: str = "fake-cli") -> None:
        super().__init__(model_name)
        self.top_k = top_k

    def _decide(self, last_content: str, messages: list[ChatMessage]) -> ChatMessage:
        if last_content.startswith("RESULT:"):
            payload = _parse_result(last_content)
            results = payload.get("results", []) if isinstance(payload, dict) else []
            if not results:
                return ChatMessage(role=Role.ASSISTANT, content="（未检索到相关段落）")
            lines = "\n".join(f"[{r.get('id', '?')}] {r.get('text', '')}" for r in results)
            src = payload.get("source", "")
            return ChatMessage(role=Role.ASSISTANT, content=f"根据检索内容（source={src}）：\n{lines}")
        user_q = next((m.content for m in reversed(messages) if m.role == Role.USER), last_content)
        return ChatMessage(
            role=Role.ASSISTANT,
            content="我先检索一下相关段落。",
            tool_calls=[ToolCall(id="call_1", name="retrieve",
                                 arguments={"query": user_q, "top_k": self.top_k})],
        )


def _parse_result(last_content: str):
    try:
        return ast.literal_eval(last_content[len("RESULT:"):].strip())
    except Exception:  # noqa: BLE001  # ast 解析失败降级返回空结果
        return {}


def build_agent(
    model: str,
    doc: str | None = None,
    embedder_kind: str = "fake",
    top_k: int = 3,
    system_prompt: str = SYSTEM_PROMPT,
) -> AgentLoop:
    """组装一个可直接对话的 AgentLoop。"""
    router = ModelRouter.from_presets(prefer=model if model in PRESET_ENDPOINTS else "deepseek", fake=True)
    if model == "fake":
        llm = CliFakeLLM(top_k=top_k)
    else:
        llm = router.get(model)  # 真实端点（需 key，否则首次调用报错）

    registry = ToolRegistry()
    rag_pipeline = None
    if doc:
        from lightcore.rag.embedder import BgeM3Embedder, FakeHashEmbedder
        from lightcore.rag.pipeline import RAGPipeline

        emb = BgeM3Embedder() if embedder_kind == "real" else FakeHashEmbedder()
        rag_pipeline = RAGPipeline(embedder=emb)
        n = rag_pipeline.add_document(doc)
        print(f"[系统] 已载入文档，分块 {n} 条（embedder={embedder_kind}）。")
    # 内置工具始终注册（retrieve 在无文档时回退 mock），否则无 --doc 运行时 FakeLLM 调用的 retrieve 会 KeyError
    register_builtin_tools(registry, rag_pipeline=rag_pipeline)
    executor = ToolExecutor(registry, permission_enabled={
        Permission.SAFE: True, Permission.NETWORK: True, Permission.SANDBOX: True, Permission.ADMIN: False,
    })
    return AgentLoop(llm=llm, tools=executor, system_prompt=system_prompt,
                     config=LoopConfig(temperature=0.3))


def _brief(out, limit: int = 100) -> str:
    if isinstance(out, dict):
        results = out.get("results", [])
        if results:
            txt = " | ".join(str(r.get("text", ""))[:40] for r in results[:3])
            return f"命中 {len(results)} 段: {txt}…"
        return str(out)[:limit]
    return str(out)[:limit]


def _render_event(ev, streaming: bool = False) -> None:
    t = ev.type
    if t == "thinking":
        if _HAS_RICH:
            _console.print(f"[dim]🤔 思考中… (step {ev.data.get('step')})[/dim]")
        else:
            print(f"🤔 思考中… (step {ev.data.get('step')})")
    elif t == "tool_call":
        if _HAS_RICH:
            _console.print(f"[cyan]🔧 调用工具[/cyan] [bold]{ev.data['name']}[/bold]({ev.data.get('arguments', {})})")
        else:
            print(f"🔧 调用工具 {ev.data['name']}({ev.data.get('arguments', {})})")
    elif t == "tool_result":
        if _HAS_RICH:
            _console.print(f"[green]↩️ 工具结果[/green] {_brief(ev.data.get('output'))}")
        else:
            print(f"↩️ 工具结果 {_brief(ev.data.get('output'))}")
    elif t == "message":
        if ev.data.get("delta") is not None:
            # 增量渲染：直接输出 delta，不换行（逐字效果）
            if _HAS_RICH:
                _console.out(ev.data["delta"], end="")
            else:
                print(ev.data["delta"], end="", flush=True)
        elif not streaming:
            # 非流式模式下的完整回答
            content = ev.data.get("content", "")
            if _HAS_RICH:
                _console.print(Markdown(content))
            else:
                print(content)
        # 流式模式下收尾的完整 message 事件（content 存在）跳过，已在 delta 中展示
    elif t == "error":
        reason = ev.data.get("reason", "未知错误")
        if _HAS_RICH:
            _console.print(f"[red]⚠️ {reason}[/red]")
        else:
            print(f"⚠️ {reason}")
    # done 事件不单独渲染


def _rebuild_history(history, user_input, events):
    """与 core/loop.py::AgentLoop.run 一致的 history 重建逻辑。"""
    new_history = list(history or [])
    new_history.append(ChatMessage(role=Role.USER, content=user_input))
    for e in events:
        if e.type == "tool_call":
            new_history.append(ChatMessage(
                role=Role.ASSISTANT, content=e.data.get("thought", ""),
                tool_calls=[ToolCall(id=e.data["call_id"], name=e.data["name"], arguments=e.data["arguments"])]))
        elif e.type == "tool_result":
            new_history.append(ChatMessage(role=Role.TOOL, content=f"RESULT: {e.data.get('output')}",
                                           tool_call_id=e.data["call_id"]))
        elif e.type == "message" and e.data.get("content"):
            new_history.append(ChatMessage(role=Role.ASSISTANT, content=e.data.get("content", "")))
    return new_history


async def _turn(agent: AgentLoop, user_input: str, history, stream: bool, memory) -> tuple[str, list]:
    # 跨会话记忆：把与当前问题相关的长期记忆作为 SYSTEM 注记前缀注入（不写入持久历史）
    recalled = memory.recall_long(user_input, k=5)
    note = None
    if recalled:
        note = ChatMessage(role=Role.SYSTEM, content="【长期记忆】\n" + "\n".join(f"- {it.text}" for it in recalled))
    messages = ([note] if note else []) + list(history or [])
    events = []
    async for ev in agent.run_stream(user_input, messages, stream_tokens=stream):
        events.append(ev)
        if stream:
            _render_event(ev, streaming=True)
    final = next((e for e in reversed(events) if e.type == "message" and e.data.get("content")), None)
    final_text = final.data.get("content", "") if final else ""
    if not stream:  # 非流式：仅打印最终回答
        if final_text:
            if _HAS_RICH:
                _console.print(Markdown(final_text))
            else:
                print(final_text)
    # 记忆：轮末反思写入（跨会话持久化）
    if final_text:
        await memory.reflect(agent.llm, conversation=[
            ChatMessage(role=Role.USER, content=user_input),
            ChatMessage(role=Role.ASSISTANT, content=final_text),
        ])
    return final_text, _rebuild_history(history, user_input, events)


async def _repl(agent: AgentLoop, stream: bool, memory) -> None:
    banner = "PaperMate 最小对话已启动（输入 exit / quit 退出；记忆跨会话持久化已开启）。"
    if _HAS_RICH:
        _console.print(f"[bold]{banner}[/bold]")
    else:
        print(banner)
    history = []
    while True:
        try:
            user_input = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("再见。")
            break
        final_text, history = await _turn(agent, user_input, history, stream, memory)


def main() -> None:
    parser = argparse.ArgumentParser(description="PaperMate 最小命令行对话")
    parser.add_argument("--model", default="fake",
                        choices=["fake", "deepseek", "openai", "qwen"],
                        help="LLM 后端（fake 无需 key；其余需在 .env 配置对应 key）")
    parser.add_argument("--doc", default=None, help="论文纯文本路径，构建 RAG 管线并接入 retrieve")
    parser.add_argument("--embedder", default="fake", choices=["fake", "real"],
                        help="RAG 嵌入器：fake=哈希占位(秒级) / real=BGE-M3(需已下载权重)")
    parser.add_argument("--top-k", type=int, default=3, help="retrieve 返回条数")
    parser.add_argument("--no-stream", action="store_true", help="关闭流式，仅输出最终回答")
    parser.add_argument("--system", default=None, help="自定义系统提示词")
    parser.add_argument("--reindex", action="store_true",
                        help="重建所有文档的检索索引后退出（切换 embedding 模式后使用，不进入对话）")
    args = parser.parse_args()

    # 重建索引模式：按当前生效配置重建所有文档管线缓存并退出（不进入对话）
    if args.reindex:
        from sqlalchemy.orm import sessionmaker

        from papermate.config import get_settings
        from papermate.models.db import init_db, make_engine
        from papermate.services.document_service import DocumentService
        settings = get_settings()
        engine = make_engine(settings.db_url)
        init_db(engine)
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        doc_svc = DocumentService(SessionLocal, settings)
        eff = doc_svc._rag_rt()
        print(f"[生效配置] rag_embedder={eff['rag_embedder']} "
              f"use_bge_sparse={eff['use_bge_sparse']} use_jieba={eff['use_jieba']}")
        n = asyncio.run(doc_svc.reindex_all())
        print(f"[完成] 已重建索引文档数：{n}")
        return

    doc_text = None
    if args.doc:
        try:
            doc_text = Path(args.doc).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"[错误] 找不到文档文件: {args.doc}")
            sys.exit(2)
    agent = build_agent(model=args.model, doc=doc_text, embedder_kind=args.embedder,
                        top_k=args.top_k, system_prompt=args.system or SYSTEM_PROMPT)
    # 跨会话记忆管理器（持久化到 data/memories.json）
    _mem_path = Path(os.environ.get("PAPERMATE_DATA_DIR", "data")) / "memories.json"
    _mem_path.parent.mkdir(parents=True, exist_ok=True)
    memory = MemoryManager(FakeHashEmbedder(), store=JSONFileMemoryStore(_mem_path))
    try:
        asyncio.run(_repl(agent, stream=not args.no_stream, memory=memory))
    except KeyboardInterrupt:
        print("\n再见。")


if __name__ == "__main__":
    main()
