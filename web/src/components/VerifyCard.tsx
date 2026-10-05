import type { VerifyResult } from "../lib/types";
import { Icons } from "../lib/icons";

/** P1-2.1 回复逐条核验卡：只读展示核验结论，绝不改动回复文本。
 * - status=ok      ：折叠显示「已核验 ✓」（含 L2 时注明语义核验通过）
 * - status=issues  ：展开显示存疑项（severity=warn/suspect），可逐条查看页码与原因
 * - status=skipped ：提示核验降级（如 L2 超时/解析失败）
 */
export function VerifyCard({ verify }: { verify: VerifyResult }) {
  const hasIssues = verify.issues.length > 0;
  const l2Tag = verify.l2 ? ` · L2 ${verify.l2 === "ok" ? "通过" : verify.l2 === "skipped" ? "降级" : "发现问题"}` : "";
  return (
    <details className={`verify-card ${hasIssues ? "has-issues" : ""}`} open={hasIssues}>
      <summary>
        {hasIssues ? (
          <span className="verify-title"><Icons.alert size={13} /> 回复核验发现 {verify.issues.length} 处存疑</span>
        ) : (
          <span className="verify-title"><Icons.check size={13} /> 回复已核验（L{verify.level}{l2Tag}）</span>
        )}
      </summary>
      {verify.status === "skipped" && !hasIssues && (
        <p className="verify-note">本次核验降级跳过（LLM 语义核验超时或不可用），引用未发现异常。</p>
      )}
      {hasIssues && (
        <ul className="verify-issues">
          {verify.issues.map((it, i) => (
            <li key={i}>
              <span className={`verify-sev ${it.severity}`}>
                {it.severity === "suspect" ? "存疑" : "提醒"}
              </span>
              <span className="verify-scope">
                {it.scope === "citation" ? "引用" : "回复"}
                {it.page ? ` · p.${it.page}` : ""}
              </span>
              <span className="verify-reason">{it.reason}</span>
              {it.text_snippet && <span className="verify-snippet">“{it.text_snippet}…”</span>}
            </li>
          ))}
        </ul>
      )}
      {!hasIssues && verify.l2 === "ok" && (
        <p className="verify-note">引用文本与页码核对一致，语义核验通过。</p>
      )}
      {!hasIssues && !verify.l2 && (
        <p className="verify-note">全部引用的文本与页码核对一致（确定性核验）。</p>
      )}
    </details>
  );
}
