// 轻量 i18n：核心 UI 文案中英切换（localStorage 持久化）。
// 范围说明：覆盖导航/页头/设置页等高频界面文案；对话内容与文档正文不参与翻译。
import { useCallback, useEffect, useState } from "react";

export type Lang = "zh" | "en";

const DICT: Record<string, { zh: string; en: string }> = {
  "nav.home": { zh: "首页", en: "Home" },
  "nav.docs": { zh: "文献", en: "My Library" },
  "nav.chat": { zh: "问答", en: "Chat" },
  "nav.usage": { zh: "用量", en: "Usage" },
  "nav.settings": { zh: "设置", en: "Settings" },
  "nav.about": { zh: "关于", en: "About" },
  "settings.title": { zh: "设置", en: "Settings" },
  "settings.subtitle": { zh: "模型、功能开关与本地数据管理", en: "Models, features and local data" },
  "settings.models": { zh: "模型与厂商", en: "Models & Providers" },
  "settings.features": { zh: "功能模块", en: "Feature Toggles" },
  "settings.rules": { zh: "本地长期记忆（Agent 规则）", en: "Local Long-term Memory (Agent Rules)" },
  "settings.rules.hint": { zh: "设定希望 Agent 始终遵循的规则/偏好，逐行一条，直接影响所有对话", en: "Rules the agent must always follow, one per line; affects all conversations" },
  "settings.data": { zh: "本地数据管理", en: "Local Data" },
  "settings.security": { zh: "安全中心", en: "Security Center" },
  "settings.balances": { zh: "账户余额", en: "Account Balances" },
  "settings.test": { zh: "测试连接", en: "Test Connection" },
  "settings.testing": { zh: "测试中…", en: "Testing…" },
  "settings.save": { zh: "保存设置", en: "Save Settings" },
  "settings.saved": { zh: "已保存", en: "Saved" },
  // ★ 2026-10-22：原名「全局默认模型」语义不清 —— 它其实承担**两件事**：
  //   ① 新建会话的初始模型；② **无会话的功能**（一键总结 / 标题翻译）唯一可用的模型。
  //   会话内的对话与提示词增强**永远优先用会话自己的模型**，不吃这个值。
  "settings.defaultModel": {
    zh: "默认模型（新会话 + 后台服务）",
    en: "Default model (new sessions + background services)",
  },
  "settings.apiKey": { zh: "API Key", en: "API Key" },
  "settings.baseUrl": { zh: "Base URL", en: "Base URL" },
  "settings.model": { zh: "模型", en: "Model" },
  "settings.lang": { zh: "界面语言", en: "Language" },
  "settings.path": { zh: "本地存储路径", en: "Local Data Path" },
  "settings.path.hint": { zh: "数据目录（当前只读展示；通过 PAPERMATE_DATA_DIR 环境变量配置）", en: "Data directory (read-only; configure via PAPERMATE_DATA_DIR env)" },
  "settings.tools": { zh: "可用工具", en: "Available Tools" },
  "settings.skills": { zh: "可用技能", en: "Available Skills" },
  "settings.sandbox": { zh: "沙箱安全配置", en: "Sandbox Security" },
  "common.enabled": { zh: "启用", en: "On" },
  "common.disabled": { zh: "关闭", en: "Off" },
  "common.confirm": { zh: "确认", en: "OK" },
};

export function useLang(): { lang: Lang; setLang: (l: Lang) => void; t: (key: string) => string } {
  const [lang, setLangState] = useState<Lang>(() => {
    // N10 修复（2026-09-10）：读取侧原先既未校验合法性也未包 try——
    // ①存储污染（如被改成 "ZH"）导致语言落空分支；②隐私模式下 localStorage 访问抛错
    // 会让 useState 初始化器直接崩掉整棵组件树（写入侧下方 effect 本就有 try）。
    try {
      const raw = typeof localStorage === "undefined" ? null : localStorage.getItem("papermate.lang");
      return raw === "zh" || raw === "en" ? raw : "zh";
    } catch {
      return "zh"; // 存储不可用（隐私模式/配额限制）：回落默认语言，不阻断渲染
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem("papermate.lang", lang);
    } catch {
      /* 忽略存储失败 */
    }
  }, [lang]);

  const setLang = useCallback((l: Lang) => setLangState(l), []);
  const t = useCallback(
    (key: string) => {
      const item = DICT[key];
      if (!item) return key;
      return lang === "zh" ? item.zh : item.en;
    },
    [lang],
  );

  return { lang, setLang, t };
}
