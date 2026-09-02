# Official vLLM v0.28.0 multi-architecture image, pinned to its published manifest.
# Ray is an optional vLLM dependency but is required by this two-node executor.
FROM vllm/vllm-openai:v0.28.0@sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14

# vLLM 0.28.0 declares Ray 2.58.0. Keep the worker and head byte-compatible.
RUN python3 -m pip install --no-cache-dir "ray==2.58.0" \
 && ray --version \
 && python3 - <<'PY'
import ray
import vllm
assert ray.__version__ == "2.58.0", ray.__version__
assert vllm.__version__ == "0.28.0", vllm.__version__
print(f"vLLM {vllm.__version__}; Ray {ray.__version__}")
PY
