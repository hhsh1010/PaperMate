import { useState } from "react";
import { Icons } from "../lib/icons";

/**
 * 登录 / 注册页（画布 14:242）
 * 结构对齐画布：浅灰底居中 1080×720 白卡（圆角 24）——
 * 左品牌区（蓝底 480：品牌头部 + 36px 标题 + 副标题 + 亮点列表），
 * 右表单区（600：26px 表单头 + Tab 切换 + 邮箱/密码 + 记住我 + 登录按钮
 * + 分隔线 + 第三方登录 + API Token 登录 + 表单底部）。
 * ⚠️ 后端暂无 auth 接口：登录仅做本地表单校验后进入工作台（TODO: 接入 /api/v1/auth/login）。
 */
export function LoginPage({
  onNavigate,
  onLogin,
}: {
  onNavigate: (view: string) => void;
  onLogin: () => void;
}) {
  const [tab, setTab] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [remember, setRemember] = useState(true);
  const [token, setToken] = useState("");
  const [msg, setMsg] = useState<string | null>(null);

  const submit = () => {
    // TODO(auth): 后端 auth 接口落地后改为 await api.login(...)
    if (tab === "login" && (!email || !password)) {
      setMsg("请填写邮箱与密码");
      return;
    }
    setMsg(null);
    onLogin();
  };

  return (
    <div className="login-page">
      <div className="login-card">
        {/* 左 · 品牌区（蓝底 480） */}
        <aside className="login-brand">
          <div className="login-brand-head">
            <span className="login-brand-icon">
              <Icons.bot size={22} />
            </span>
            <span className="login-brand-name">论文智友 · PaperMate</span>
          </div>

          <h2 className="login-brand-title">
            欢迎回到
            <br />
            论文智友
          </h2>
          <p className="login-brand-sub">
            让每一篇论文，都被充分理解。
            <br />
            深度问答 · 引用溯源 · 长期记忆。
          </p>

          <div className="login-brand-spacer" />

          <ul className="login-brand-points">
            <li>带引用溯源的多轮问答</li>
            <li>单篇总结 / 多篇对比 / 相关工作</li>
            <li>推理轨迹与用量成本全透明</li>
          </ul>
        </aside>

        {/* 右 · 表单区（600） */}
        <section className="login-form">
          <h3 className="login-form-title">登录你的账户</h3>
          <p className="login-form-sub">继续你的论文智读之旅</p>

          <div className="login-tabs" role="tablist">
            <button
              role="tab"
              aria-selected={tab === "login"}
              className={`login-tab ${tab === "login" ? "active" : ""}`}
              onClick={() => setTab("login")}
            >
              登录
            </button>
            <button
              role="tab"
              aria-selected={tab === "register"}
              className={`login-tab ${tab === "register" ? "active" : ""}`}
              onClick={() => setTab("register")}
            >
              注册
            </button>
          </div>

          <div className="login-fields">
            <label className="settings-field">
              <span className="settings-label">邮箱</span>
              <input
                className="settings-control"
                type="email"
                autoComplete="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </label>
            <label className="settings-field">
              <span className="settings-label">密码</span>
              <input
                className="settings-control"
                type="password"
                autoComplete={tab === "login" ? "current-password" : "new-password"}
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </label>

            <label className="login-remember">
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
              />
              <span>记住我</span>
            </label>
          </div>

          {msg && <div className="error-box">{msg}</div>}

          <button className="login-submit-btn" onClick={submit}>
            {tab === "login" ? "登录" : "注册"}
          </button>

          <div className="login-divider">
            <span>或</span>
          </div>

          <div className="login-third-party">
            <button className="login-third-btn" title="微信登录（未接入）">微</button>
            <button className="login-third-btn" title="QQ 登录（未接入）">Q</button>
            <button className="login-third-btn" title="GitHub 登录（未接入）">
              <Icons.bot size={16} />
            </button>
          </div>

          <button className="login-token-btn" title="API Token 登录（后端 auth 未开放）">
            <span>使用 API Token 登录</span>
            <span className="login-token-arrow"><Icons.arrowRight size={14} /></span>
          </button>

          <p className="login-hint">
            {tab === "login" ? "还没有账号？" : "已有账号？"}
            <button className="linklike" onClick={() => setTab(tab === "login" ? "register" : "login")}>
              {tab === "login" ? "去注册" : "去登录"}
            </button>
            · 或 <button className="linklike" onClick={() => onNavigate("chat")}>先去问答</button>
          </p>
          <p className="login-note muted">提示：后端认证接口暂未开放，本页为前端入口占位，登录后进入工作台。</p>
        </section>
      </div>
    </div>
  );
}
