import { useEffect, useState } from "react";
import type { DocumentOut, SessionOut } from "../lib/types";
import { api } from "../api/client";
import { timeAgo } from "../lib/time";
import { Icons } from "../lib/icons";
import { toErrorMessage } from "../lib/error";

const STATUS_LABEL: Record<string, string> = {
  uploaded: "已上传",
  parsing: "解析中",
  chunking: "分块中",
  embedding: "向量化",
  ready: "就绪",
  failed: "失败",
};

/**
 * F9：研究首页（聚合「最近会话 / 最近文献 / 快捷研读」），填补已删首页的价值真空。
 * 按产品决策**不设为默认落地页**（默认仍是 chat），作为侧栏「首页」总览入口。
 */
export function HomePage({
  onOpenSession,
  onNavigate,
  onNewChat,
  onCreateWorkspace,
}: {
  onOpenSession: (id: number) => void;
  onNavigate: (v: "docs" | "chat" | "eval") => void;
  onNewChat: () => void;
  onCreateWorkspace: () => void;
}) {
  const [sessions, setSessions] = useState<SessionOut[]>([]);
  const [docs, setDocs] = useState<DocumentOut[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    Promise.all([api.listSessions(), api.listDocuments()])
      .then(([ss, ds]) => {
        if (!alive) return;
        setSessions(ss);
        setDocs(ds);
      })
      .catch((e: unknown) => {
        if (alive) setError(toErrorMessage(e) || "加载数据失败");
      });
    return () => {
      alive = false;
    };
  }, []);

  // 后端已按 updated_at 倒序返回会话；文献按 created_at 倒序取最近
  const recentSessions = sessions.slice(0, 5);
  const recentDocs = [...docs]
    .sort((a, b) => (Date.parse(b.created_at ?? "") || 0) - (Date.parse(a.created_at ?? "") || 0))
    .slice(0, 5);

  return (
    <div className="home-page">
      <header className="home-hero">
        <h1>欢迎回来</h1>
        <p className="muted">上传论文、开始研读，或从上次的进度继续。</p>
        <div className="home-actions">
          <button className="home-action primary" onClick={onNewChat}><Icons.sparkles size={14} /> 新建对话
          </button>
          <button className="home-action" onClick={() => onNavigate("docs")}><Icons.fileText size={13} /> 上传 / 管理文献
          </button>
          <button className="home-action" onClick={() => onNavigate("eval")}><Icons.chart size={14} /> 忠实度看板
          </button>
          <button className="home-action workspace" onClick={onCreateWorkspace}><Icons.layers size={14} /> 新建工作间
          </button>
        </div>
      </header>

      <div className="home-stats">
        <div className="stat-tile">
          <span className="stat-num">{docs.length}</span>
          <span className="stat-label">文献总数</span>
        </div>
        <div className="stat-tile">
          <span className="stat-num">{sessions.length}</span>
          <span className="stat-label">会话总数</span>
        </div>
        <div className="stat-tile">
          <span className="stat-num">{recentSessions.length}</span>
          <span className="stat-label">最近活跃</span>
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}

      <div className="home-grid">
        <section className="home-card">
          <div className="home-card-head">
            <span className="usage-table-title">最近会话</span>
            <button className="home-more" onClick={() => onNavigate("chat")}>
              全部
              <Icons.arrowRight size={13} />
            </button>
          </div>
          {recentSessions.length === 0 ? (
            <div className="trend-empty">还没有会话，点上方「新建对话」开始</div>
          ) : (
            <ul className="home-list">
              {recentSessions.map((s) => (
                <li
                  key={s.id}
                  className="home-item clickable"
                  role="button"
                  tabIndex={0}
                  onClick={() => onOpenSession(s.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onOpenSession(s.id);
                    }
                  }}
                >
                  <span className="home-item-title">{s.title}</span>
                  <span className="home-item-time">{timeAgo(s.updated_at ?? s.created_at)}</span>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="home-card">
          <div className="home-card-head">
            <span className="usage-table-title">最近文献</span>
            <button className="home-more" onClick={() => onNavigate("docs")}>
              全部
              <Icons.arrowRight size={13} />
            </button>
          </div>
          {recentDocs.length === 0 ? (
            <div className="trend-empty">文献库还是空的，先到「文献」上传论文</div>
          ) : (
            <ul className="home-list">
              {recentDocs.map((d) => (
                <li
                  key={d.id}
                  className="home-item"
                  role="button"
                  tabIndex={0}
                  onClick={() => onNavigate("docs")}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onNavigate("docs");
                    }
                  }}
                >
                  <span className="home-item-title">{d.filename}</span>
                  <span className={`home-item-status ${d.status}`}>{STATUS_LABEL[d.status] ?? d.status}</span>
                  <span className="home-item-time">{timeAgo(d.created_at)}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}
