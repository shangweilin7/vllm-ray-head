#!/usr/bin/env bash
set -euo pipefail

HEAD_ADDRESS="${HEAD_IP}:6379"
# Shared by the startup memory gate and `vllm serve`.
GPU_MEMORY_UTILIZATION=0.80
: "${NODE_IP:?NODE_IP must be set}"
: "${HEAD_IP:?HEAD_IP must be set}"
: "${ROCE_IFACE:?ROCE_IFACE must be set}"

printf '%s\n' '[1/5] Cleaning stale Ray session...'
ray stop --force >/dev/null 2>&1 || true

printf '[2/5] Starting Ray head at %s...\n' "${HEAD_ADDRESS}"
ray start \
  --head \
  --node-ip-address="${NODE_IP}" \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-cpus=20 \
  --num-gpus=1

printf '%s\n' '[3/5] Waiting for the second Ray worker...'
python3 - <<'PY'
import os
import time
import ray

address = f"{os.environ['HEAD_IP']}:6379"
deadline = time.time() + 600
while True:
    try:
        if not ray.is_initialized():
            ray.init(address=address, ignore_reinit_error=True)
        alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
        gpu_total = float(ray.cluster_resources().get("GPU", 0))
        print(f"Ray cluster: alive_nodes={len(alive_nodes)}, GPU={gpu_total}", flush=True)
        if len(alive_nodes) >= 2 and gpu_total >= 2:
            ray.shutdown()
            break
        ray.shutdown()
    except Exception as exc:
        print(f"Ray not ready: {exc}", flush=True)
        try:
            ray.shutdown()
        except Exception:
            pass
    if time.time() >= deadline:
        raise SystemExit("Timed out waiting for two Ray nodes and two GPUs.")
    time.sleep(5)
PY

printf '[4/5] Waiting for >= %s of unified memory to be available...\n' "${GPU_MEMORY_UTILIZATION}"
# vLLM rejects startup when MemAvailable (UMA) < total * gpu_memory_utilization.
# Crashing here restarts the container, which replaces the Ray GCS session and
# knocks the worker raylet out ("GCS returned an authentication error").
# Wait in place instead so the Ray cluster stays intact.
python3 - "${GPU_MEMORY_UTILIZATION}" <<'PY'
import sys
import time

utilization = float(sys.argv[1])
headroom_gib = 1.0
last_report = 0.0


def meminfo_gib(key: str) -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1024 / 1024
    raise KeyError(key)


total = meminfo_gib("MemTotal")
required = total * utilization + headroom_gib
while True:
    available = meminfo_gib("MemAvailable")
    if available >= required:
        print(f"Memory ready: available={available:.2f} GiB >= required={required:.2f} GiB", flush=True)
        break
    now = time.time()
    if now - last_report >= 60:
        print(
            f"Waiting for memory: available={available:.2f} GiB < required={required:.2f} GiB "
            "(check ComfyUI / other GPU workloads)",
            flush=True,
        )
        last_report = now
    time.sleep(10)
PY

printf '%s\n' '[5/5] Starting eugr B12X vLLM (DeepSeek V4 Flash 0731, TP=2, DSpark)...'
exec vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash-0731 \
  --host 0.0.0.0 \
  --port 8000 \
  --distributed-executor-backend ray \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 10240 \
  --enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --trust-remote-code \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --default-chat-template-kwargs.thinking=true \
  --default-chat-template-kwargs.reasoning_effort=high \
  --moe-backend b12x \
  --linear-backend b12x \
  --attention-backend B12X \
  --max-cudagraph-capture-size 48 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"]}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","attention_backend":"B12X"}'
