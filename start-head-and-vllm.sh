#!/usr/bin/env bash
set -euo pipefail

HEAD_ADDRESS="${HEAD_IP}:6379"

# SM121 requirement: DeepGEMM must remain enabled. Do not export
# VLLM_USE_DEEP_GEMM=0; that forces the unsupported CUTLASS fallback.
unset VLLM_USE_DEEP_GEMM || true
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_DEEP_GEMM_WARMUP=full

# Require the vLLM-native partition variable explicitly.
: "${VLLM_PP_LAYER_PARTITION:?VLLM_PP_LAYER_PARTITION must be set, e.g. 14,29}"
export VLLM_PP_LAYER_PARTITION

# The first PP=2 validation intentionally runs without DSpark. The DSpark
# draft model does not implement SupportsPP in vLLM 0.27.1.
# Keep #51835 and the sparse-SWA patch in the image for a later TP topology test.

echo "[1/4] Cleaning stale Ray session..."
ray stop --force >/dev/null 2>&1 || true

echo "[2/4] Starting Ray Head at ${HEAD_ADDRESS}..."
ray start \
  --head \
  --node-ip-address="${NODE_IP}" \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265 \
  --num-cpus=20 \
  --num-gpus=1

echo "[3/4] Waiting for GX10-2 Ray Worker..."
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

        print(
            f"Ray cluster: alive_nodes={len(alive_nodes)}, GPU={gpu_total}",
            flush=True,
        )

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

echo "[4/4] Starting vLLM API without DSpark (PP=2 validation)..."
exec vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash-0731 \
  --host 0.0.0.0 \
  --port 8000 \
  --distributed-executor-backend ray \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-memory 7516192768 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --trust-remote-code
