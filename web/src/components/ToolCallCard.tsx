import { useState } from "react";
import type { ToolCallView } from "../lib/types";
import { Icons } from "../lib/icons";

function pretty(v: unknown): string {
  try {
    return typeof v === "string" ? v : JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

/** 工具调用卡片：折叠展示参数与结果（docs/04 §4.4 · ToolCallCard） */
export function ToolCallCard({ tool }: { tool: ToolCallView }) {
  const [open, setOpen] = useState(false);
  const icon =
    tool.state === "running" ? (
      <Icons.clock size={14} />
    ) : tool.state === "ok" ? (
      <Icons.checkCircle size={14} />
    ) : (
      <Icons.xCircle size={14} />
    );

  return (
    <div className={`tool-call tool-${tool.state} ${open ? "open" : ""}`}>
      <button className="tool-head" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        <span>{icon}</span>
        <span className="tool-name">{tool.name}</span>
        <span className="tool-state">
          {tool.state === "running" ? "执行中" : tool.state === "ok" ? "完成" : "失败"}
        </span>
        <span className="tool-chevron" aria-hidden="true"><Icons.chevronRight size={14} /></span>
      </button>
      {/* 常挂载 + class 控制高度：收起时平滑过渡（原来条件挂载会"啪"地消失）*/}
      <div className={`collapse${open ? " open" : ""}`}>
        <div className="collapse-inner">
        <div className="tool-body">
          <div>
            <strong>参数</strong>
            <pre>{pretty(tool.args)}</pre>
          </div>
          {tool.state !== "running" && (
            <div>
              <strong>{tool.state === "ok" ? "结果" : "错误"}</strong>
              <pre>{pretty(tool.result ?? tool.error)}</pre>
            </div>
          )}
        </div>
        </div>
      </div>
    </div>
  );
}
