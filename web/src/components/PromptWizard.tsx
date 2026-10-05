import { useEffect } from "react";
import { PROMPT_TEMPLATE_GROUPS, PROMPT_TEMPLATE_TIP } from "../lib/promptTemplates";
import { Icons } from "../lib/icons";

/** 分组名 → 图标（数据层保持纯文本，图标在视图层映射） */
const GROUP_ICON: Record<string, () => React.ReactElement> = {
  读懂论文: () => <Icons.book size={13} />,
  总结梳理: () => <Icons.list size={13} />,
  对比评估: () => <Icons.layers size={13} />,
  方法复现: () => <Icons.terminal size={13} />,
  写作表达: () => <Icons.fileText size={13} />,
};

/**
 * 提问模板面板（2026-09-01 重新定位）
 *
 * 原「SkillWizard」按 category 分组罗列技能命令，与「技能命令」面板**功能重复**
 * （同一份 skills 数据、同一套插入逻辑、同一结果）。
 *
 * 现改为论文通用的**提问 / 需求模板**，服务「不知道该怎么形容自己的问题、
 * 不知道如何向 Agent 有效提问」的用户：点选后把模板原文填入输入框供二次编辑，
 * 不再插入 `/命令`。由此两个入口定位彻底分离：
 *   · 技能命令 = 调用系统能力（我知道要做什么）
 *   · 提问模板 = 获得提问范例（我不知道该怎么问）
 */
export function PromptWizard({
  onPick,
  onClose,
}: {
  /** 收到的是模板原文（自然语言问句），不是 /命令 */
  onPick: (template: string) => void;
  onClose: () => void;
}) {
  // 与技能面板一致：Esc 收起
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="wizard-panel">
      <div className="skill-panel-head">
        <span className="skill-panel-title">
          <Icons.sparkles size={13} />
          选一个提问模板
        </span>
        <button
          type="button"
          className="skill-panel-close"
          onClick={onClose}
          aria-label="收起提问模板"
          title="收起提问模板"
        >
          <Icons.x size={13} />
        </button>
      </div>
      <div className="wizard-body">
        <p className="wizard-tip">{PROMPT_TEMPLATE_TIP}</p>
        {PROMPT_TEMPLATE_GROUPS.map((g) => {
          const GroupIcon = GROUP_ICON[g.category];
          return (
            <div className="wizard-group" key={g.category}>
              <div
                className="wizard-group-title"
                style={{ color: g.color }}
              >
                <span className="wg-icon" aria-hidden="true">
                  {GroupIcon ? <GroupIcon /> : <Icons.chat size={13} />}
                </span>
                {g.category}
              </div>
              <div className="wizard-chips">
                {g.items.map((t) => (
                  <button
                    key={t.id}
                    type="button"
                    className="wizard-chip"
                    style={{ borderLeftColor: g.color }}
                    title={t.template}
                    onClick={() => {
                      onPick(t.template);
                      onClose();
                    }}
                  >
                    <span className="wizard-chip-icon" aria-hidden="true">
                      <Icons.chat size={13} />
                    </span>
                    <span className="wizard-chip-text">
                      <span className="wizard-chip-cmd">{t.label}</span>
                      <span className="wizard-chip-desc">{t.template}</span>
                    </span>
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
