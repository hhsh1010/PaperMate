// useBackendStatus：轮询 /api/v1/healthz 检测后端在线状态（顶栏状态灯用）。
// 后端不可达时前端不再"静默空白"，而是给出可见的离线指示（2026-08-20 像素匠）。
import { useEffect, useRef, useState } from "react";

export type BackendStatus = "online" | "offline" | "unknown";

export function useBackendStatus(intervalMs = 6000): BackendStatus {
  const [status, setStatus] = useState<BackendStatus>("unknown");
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      try {
        const res = await fetch("/api/v1/healthz", { signal: AbortSignal.timeout(2500) });
        if (cancelled) return;
        setStatus(res.ok ? "online" : "offline");
      } catch {
        if (!cancelled) setStatus("offline");
      }
    };

    check();
    timerRef.current = window.setInterval(check, intervalMs);
    return () => {
      cancelled = true;
      if (timerRef.current != null) window.clearInterval(timerRef.current);
    };
  }, [intervalMs]);

  return status;
}
