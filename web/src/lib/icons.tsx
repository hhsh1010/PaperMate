/**
 * 内联 SVG 图标（线性 stroke 风格，对标 Linear / Vercel / Lucide）。
 * - 统一 1.5px 描边（旧版 2px 偏粗显老旧）；圆头端点，视觉更克制精致
 * - 独立文件：App 与各页面共享，避免 App ↔ 页面循环依赖
 * - 覆盖全站场景：导航 / 输入区 / 状态 / 文档 / 技能 / 编辑操作
 */

export type IconProps = { size?: number; strokeWidth?: number };

const icon = (path: React.ReactNode) =>
  function Icon({ size = 18, strokeWidth = 1.5 }: IconProps) {
    return (
      <svg
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        {path}
      </svg>
    );
  };

export const Icons = {
  /* ---------------- 导航 ---------------- */
  chat: icon(<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />),
  docs: icon(
    <>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6" />
      <path d="M16 13H8" />
      <path d="M16 17H8" />
    </>,
  ),
  usage: icon(
    <>
      <path d="M3 3v18h18" />
      <path d="M7 15v-3" />
      <path d="M12 15V8" />
      <path d="M17 15v-5" />
    </>,
  ),
  settings: icon(
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </>,
  ),
  home: icon(
    <>
      <path d="m3 10 9-7 9 7v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <path d="M9 22V12h6v10" />
    </>,
  ),
  upload: icon(
    <>
      <path d="M12 16V5" />
      <path d="m8 9 4-4 4 4" />
      <path d="M5 19h14" />
    </>,
  ),
  menu: icon(<path d="M4 6h16M4 12h16M4 18h16" />),
  moon: icon(<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />),
  sun: icon(
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
    </>,
  ),
  bot: icon(
    <>
      <rect x="4" y="8" width="16" height="12" rx="2" />
      <path d="M12 8V4M9 4h6" />
      <path d="M8 14h.01M16 14h.01" />
      <path d="M9 17h6" />
    </>,
  ),
  user: icon(
    <>
      <circle cx="12" cy="8" r="4" />
      <path d="M4 21c0-4 4-6 8-6s8 2 8 6" />
    </>,
  ),

  /* ---------------- 方向与操作 ---------------- */
  plus: icon(<path d="M12 5v14M5 12h14" />),
  plusIcon: icon(<path d="M12 5v14M5 12h14" />),
  x: icon(<path d="M18 6 6 18M6 6l12 12" />),
  check: icon(<path d="M20 6 9 17l-5-5" />),
  checkCircle: icon(
    <>
      <path d="M21.8 10A10 10 0 1 1 17 3.34" />
      <path d="m22 4-10 10.01-3-3" />
    </>,
  ),
  chevronRight: icon(<path d="m9 18 6-6-6-6" />),
  chevronLeft: icon(<path d="m15 18-6-6 6-6" />),
  chevronDown: icon(<path d="m6 9 6 6 6-6" />),
  chevronUp: icon(<path d="m18 15-6-6-6 6" />),
  arrowRight: icon(
    <>
      <path d="M5 12h14" />
      <path d="m12 5 7 7-7 7" />
    </>,
  ),
  arrowLeft: icon(
    <>
      <path d="M19 12H5" />
      <path d="m12 19-7-7 7-7" />
    </>,
  ),
  arrowDown: icon(
    <>
      <path d="M12 5v14" />
      <path d="m19 12-7 7-7-7" />
    </>,
  ),
  arrowUpDown: icon(
    <>
      <path d="m8 3-4 4 4 4" />
      <path d="M4 7h16" />
      <path d="m16 21 4-4-4-4" />
      <path d="M20 17H4" />
    </>,
  ),

  /* ---------------- 状态与提示 ---------------- */
  alert: icon(
    <>
      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3" />
      <path d="M12 9v4" />
      <path d="M12 17h.01" />
    </>,
  ),
  xCircle: icon(
    <>
      <circle cx="12" cy="12" r="10" />
      <path d="m15 9-6 6M9 9l6 6" />
    </>,
  ),
  info: icon(
    <>
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4" />
      <path d="M12 8h.01" />
    </>,
  ),
  clock: icon(
    <>
      <circle cx="12" cy="12" r="10" />
      <path d="M12 6v6l4 2" />
    </>,
  ),

  /* ---------------- 文档与内容 ---------------- */
  file: icon(
    <>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6" />
    </>,
  ),
  fileText: icon(
    <>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6" />
      <path d="M9 13h6M9 17h6" />
    </>,
  ),
  book: icon(
    <>
      <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" />
    </>,
  ),
  folder: icon(
    <path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z" />,
  ),
  image: icon(
    <>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="9" cy="9" r="2" />
      <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21" />
    </>,
  ),
  link: icon(
    <>
      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
    </>,
  ),
  externalLink: icon(
    <>
      <path d="M15 3h6v6" />
      <path d="M10 14 21 3" />
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
    </>,
  ),

  /* ---------------- 技能与 AI ---------------- */
  zap: icon(<path d="M4 14h7l-1 8 10-12h-7l1-8z" />),
  sparkles: icon(
    <>
      <path d="m9.94 15.5-1.44-5.56L2.94 8.5l6.56-1.44L10.94 1.5l1.44 5.56 6.56 1.44-6.56 1.44z" />
      <path d="M19 21l-.7-2.3L16 18l2.3-.7L19 15l.7 2.3L22 18l-2.3.7z" />
    </>,
  ),
  brain: icon(
    <>
      <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z" />
      <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z" />
      <path d="M12 5v13" />
    </>,
  ),
  terminal: icon(
    <>
      <path d="m4 17 6-6-6-6" />
      <path d="M12 19h8" />
    </>,
  ),
  cpu: icon(
    <>
      <rect x="4" y="4" width="16" height="16" rx="2" />
      <rect x="9" y="9" width="6" height="6" rx="1" />
      <path d="M15 2v2M9 2v2M15 20v2M9 20v2M2 15h2M2 9h2M20 15h2M20 9h2" />
    </>,
  ),

  /* ---------------- 数据与视图 ---------------- */
  chart: icon(
    <>
      <path d="M3 3v16a2 2 0 0 0 2 2h16" />
      <path d="M18 17V9" />
      <path d="M13 17V5" />
      <path d="M8 17v-3" />
    </>,
  ),
  grid: icon(
    <>
      <rect x="3" y="3" width="7" height="7" rx="1" />
      <rect x="14" y="3" width="7" height="7" rx="1" />
      <rect x="14" y="14" width="7" height="7" rx="1" />
      <rect x="3" y="14" width="7" height="7" rx="1" />
    </>,
  ),
  list: icon(
    <>
      <path d="M8 6h13M8 12h13M8 18h13" />
      <path d="M3 6h.01M3 12h.01M3 18h.01" />
    </>,
  ),
  filter: icon(<path d="M22 3H2l8 9.46V19l4 2v-8.54L22 3z" />),
  layers: icon(
    <>
      <path d="m12 2 9 5-9 5-9-5z" />
      <path d="m3 12 9 5 9-5" />
      <path d="m3 17 9 5 9-5" />
    </>,
  ),

  /* ---------------- 编辑与其它 ---------------- */
  search: icon(
    <>
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </>,
  ),
  send: icon(
    <>
      <path d="M22 2 11 13" />
      <path d="M22 2 15 22l-4-9-9-4z" />
    </>,
  ),
  stop: icon(<rect x="6" y="6" width="12" height="12" rx="2" />),
  copy: icon(
    <>
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </>,
  ),
  download: icon(
    <>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="m7 10 5 5 5-5" />
      <path d="M12 15V3" />
    </>,
  ),
  trash: icon(
    <>
      <path d="M3 6h18" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    </>,
  ),
  /** 铅笔：会话/工作间**重命名**入口（2026-10-02）。 */
  pencil: icon(
    <>
      <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
      <path d="m15 5 4 4" />
    </>,
  ),
  /** 图钉：侧栏**置顶**（2026-10-02）。 */
  pin: icon(
    <>
      <path d="M12 17v5" />
      <path d="M9 10.8V4h6v6.8l2 3.2H7z" />
    </>,
  ),
  refresh: icon(
    <>
      <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
      <path d="M21 3v5h-5" />
      <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
      <path d="M8 16H3v5" />
    </>,
  ),
  globe: icon(
    <>
      <circle cx="12" cy="12" r="10" />
      <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20" />
      <path d="M2 12h20" />
    </>,
  ),
  lock: icon(
    <>
      <rect x="3" y="11" width="18" height="11" rx="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </>,
  ),
  eye: icon(
    <>
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z" />
      <circle cx="12" cy="12" r="3" />
    </>,
  ),
  smile: icon(
    <>
      <circle cx="12" cy="12" r="10" />
      <path d="M8 14s1.5 2 4 2 4-2 4-2" />
      <path d="M9 9h.01M15 9h.01" />
    </>,
  ),
  star: icon(
    <path d="m12 2 3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />,
  ),
  thumbUp: icon(
    <path d="M7 10v11H4a1 1 0 0 1-1-1v-9a1 1 0 0 1 1-1h3zm0 0 4.2-7.1a1.6 1.6 0 0 1 2.9 1.2L13 9h4.6a2 2 0 0 1 2 2.4l-1.3 6.4a2 2 0 0 1-2 1.6H7" />,
  ),
  thumbDown: icon(
    <path d="M7 14V3H4a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h3zm0 0 4.2 7.1a1.6 1.6 0 0 0 2.9-1.2L13 15h4.6a2 2 0 0 0 2-2.4l-1.3-6.4a2 2 0 0 0-2-1.6H7" />,
  ),
  more: icon(
    <>
      <circle cx="12" cy="12" r="1" />
      <circle cx="19" cy="12" r="1" />
      <circle cx="5" cy="12" r="1" />
    </>,
  ),
  // —— 面板布局（预览放大 / 还原 / 收起右侧栏）——
  maximize: icon(
    <>
      <path d="M9 3H5a2 2 0 0 0-2 2v4" />
      <path d="M15 3h4a2 2 0 0 1 2 2v4" />
      <path d="M9 21H5a2 2 0 0 1-2-2v-4" />
      <path d="M15 21h4a2 2 0 0 0 2-2v-4" />
    </>,
  ),
  minimize: icon(
    <>
      <path d="M4 9h4V5" />
      <path d="M20 9h-4V5" />
      <path d="M4 15h4v4" />
      <path d="M20 15h-4v4" />
    </>,
  ),
  panelRightClose: icon(
    <>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M15 3v18" />
      <path d="m9 9 3 3-3 3" />
    </>,
  ),
  panelRightOpen: icon(
    <>
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M15 3v18" />
      <path d="m12 9-3 3 3 3" />
    </>,
  ),
};
