#!/usr/bin/env bash
# 一次性实验：定位 09-19 报告（均值 10236.9ms）相对今日 CPU 基线（1792.2ms）的 5.7× 残差
#   臂 A：1 个 CPU 竞争进程（16 线程 matmul）并行   → 测「CPU 争用」
#   臂 B：3 个竞争进程并行                          → 测争用的量级上限
#   臂 C：单线程（OMP_NUM_THREADS=1）独自运行       → 测「线程数」轴
# 判据（先定后测）：若任一无争用臂 / 争用臂达到 ~5.7×，则残差被解释；否则拒绝该假设。
set -u
cd /d/harness-space/agent-project
export PAPERMATE_DEVICE=cpu
PY=.venv/Scripts/python.exe

burn() {
  $PY -c "
import torch, time
torch.set_num_threads(16)
a = torch.randn(1024, 1024); b = torch.randn(1024, 1024)
t0 = time.time()
while time.time() - t0 < $1:
    c = a @ b
"
}

summarize() {
  $PY -c "
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
lat = sorted(x['latency_ms'] for x in d['detail'] if x.get('latency_ms') is not None)
n = len(lat)
print('    %s -> n=%d min=%.1f p50=%.1f mean=%.1f max=%.1f' % (sys.argv[1], n, lat[0], lat[n // 2], sum(lat) / n, lat[-1]))
" "$1"
}

echo "=== 0. 线程数自检（确认 OMP_NUM_THREADS 是否真的生效）==="
$PY -c "import torch, os; print('  默认           -> torch threads =', torch.get_num_threads(), '| cpu_count =', os.cpu_count())"
OMP_NUM_THREADS=1 $PY -c "import torch; print('  OMP_NUM_THREADS=1 -> torch threads =', torch.get_num_threads())"

echo
echo "=== 臂 A：1 个竞争进程（16 线程 matmul）并行跑评测 ==="
burn 320 & CA=$!
sleep 15
time $PY eval/run_eval.py --embedder real --profile prod --cases cases/golden_real.jsonl --out eval/report_cpu_contend1.json 2>&1 | grep -E "口径|latency_ms|recall@5|报告已写入"
summarize eval/report_cpu_contend1.json
kill $CA 2>/dev/null
wait $CA 2>/dev/null
sleep 3

echo
echo "=== 臂 B：3 个竞争进程并行跑评测 ==="
burn 420 & C1=$!
burn 420 & C2=$!
burn 420 & C3=$!
sleep 15
time $PY eval/run_eval.py --embedder real --profile prod --cases cases/golden_real.jsonl --out eval/report_cpu_contend3.json 2>&1 | grep -E "口径|latency_ms|recall@5|报告已写入"
summarize eval/report_cpu_contend3.json
kill $C1 $C2 $C3 2>/dev/null
wait 2>/dev/null
sleep 5

echo
echo "=== 臂 C：单线程独自运行（测线程数轴）==="
time OMP_NUM_THREADS=1 $PY eval/run_eval.py --embedder real --profile prod --cases cases/golden_real.jsonl --out eval/report_cpu_t1.json 2>&1 | grep -E "口径|latency_ms|recall@5|报告已写入"
summarize eval/report_cpu_t1.json

echo
echo "=== 汇总 ==="
echo "  目标（09-19 报告）:"
summarize eval/report_real_stale_20260919.json
echo "  今日 CPU 基线（16 线程、无争用）:"
summarize eval/report_cpu_20260924.json
echo "  今日 GPU 基线:"
summarize eval/report_real.json
echo "  实验完成"
