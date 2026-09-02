#!/usr/bin/env bash
set -euo pipefail

HEAD_ADDRESS="${HEAD_IP}:6379"
: "${NODE_IP:?NODE_IP must be set}"
: "${HEAD_IP:?HEAD_IP must be set}"
: "${ROCE_IFACE:?ROCE_IFACE must be set}"

printf '%s\n' '[1/4] Cleaning stale Ray session...'
ray stop --force >/dev/null 2>&1 || true

printf '[2/4] Starting Ray head at %s...\n' "${HEAD_ADDRESS}"
ray start \
  --head \
  --node-ip-address="${NODE_IP}" \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-cpus=20 \
  --num-gpus=1

printf '%s\n' '[3/4] Waiting for the second Ray worker...'
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

# vLLM v0.28.0's bundled FlashInfer lacks the required SM120/121 DSV4
# sparse-MLA decode specialization (flashinfer#4380). Keep DSpark disabled
# until an official vLLM image includes that specialization.
printf '%s\n' '[4/4] Starting official vLLM v0.28.0 without DSpark on TP=2...'
exec vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash-0731 \
  --host 0.0.0.0 \
  --port 8000 \
  --distributed-executor-backend ray \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len auto \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}'
