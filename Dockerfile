# vLLM 0.27.0 for DeepSeek-V4-Flash-0731 on NVIDIA GB10 / SM121.
# Includes: #50796 DeepGEMM pin, #51835 DSpark quantization fix,
# and the SM121 sparse-SWA topk dispatch workaround.
FROM vllm/vllm-openai:v0.27.0

ARG DEEPGEMM_REF=2fd67329ec2942f65ba35d561256ab6ed3b903cb

COPY patches/50796-deepgemm-sm12x.diff /opt/vllm-patches/50796-deepgemm-sm12x.diff
COPY patches/51835-dspark-quantization.diff /opt/vllm-patches/51835-dspark-quantization.diff
COPY patches/sparse-swa-topk-sm121.diff /opt/vllm-patches/sparse-swa-topk-sm121.diff

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends git patch libcusparse-dev-13-0 libcublas-dev-13-0 libcusolver-dev-13-0 cuda-nvrtc-dev-13-0; \
    rm -rf /var/lib/apt/lists/*; \
    VLLM_SITE=/usr/local/lib/python3.12/dist-packages/vllm; \
    patch --batch --forward -p2 -d "${VLLM_SITE}" < /opt/vllm-patches/51835-dspark-quantization.diff; \
    patch --batch --forward -p2 -d "${VLLM_SITE}" < /opt/vllm-patches/sparse-swa-topk-sm121.diff; \
    git clone --recursive --shallow-submodules https://github.com/vllm-project/DeepGEMM.git /tmp/deepgemm; \
    git -C /tmp/deepgemm fetch --depth=1 origin "${DEEPGEMM_REF}"; \
    git -C /tmp/deepgemm checkout --detach "${DEEPGEMM_REF}"; \
    git -C /tmp/deepgemm rev-parse HEAD | grep -Fx "${DEEPGEMM_REF}"; \
    (cd /tmp/deepgemm && python3 setup.py bdist_wheel); \
    mkdir -p /tmp/deepgemm-wheel; \
    cp /tmp/deepgemm/dist/*.whl /tmp/deepgemm-wheel/; \
    python3 -m pip install --no-cache-dir --force-reinstall --no-deps /tmp/deepgemm-wheel/*.whl; \
    python3 -m py_compile \
      "${VLLM_SITE}/config/speculative.py" \
      "${VLLM_SITE}/v1/attention/backends/mla/sparse_swa.py"; \
    rm -rf /tmp/deepgemm /tmp/deepgemm-wheel

RUN python3 - <<'PY'
import deep_gemm
import vllm
from vllm.config.speculative import SpeculativeConfig
from vllm.v1.attention.backends.mla.sparse_swa import _round_up_to_supported_topk

assert vllm.__version__ == "0.27.0", vllm.__version__
assert _round_up_to_supported_topk(256) == 512
assert "deepseek_v4_fp8" in SpeculativeConfig.__post_init__.__code__.co_consts
print("vLLM", vllm.__version__)
print("DeepGEMM", getattr(deep_gemm, "__file__", "unknown"))
print("SM121 sparse-SWA topk 256 ->", _round_up_to_supported_topk(256))
PY

RUN python3 -m pip install --no-cache-dir --upgrade "ray[cgraph]" \
 && python3 -m pip uninstall -y cupy-cuda12x
