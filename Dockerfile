# Official vLLM v0.29.0 image + Ray (local build). The official vllm-openai
# image does NOT bundle the `ray` package, but this two-node executor depends
# on a manual Ray cluster, so layer Ray 2.58.0 on top (same proven version the
# previous baseline / eugr build used). Keep the worker and head byte-compatible.
FROM vllm/vllm-openai:v0.29.0

RUN python3 -m pip install --no-cache-dir "ray==2.58.0" \
 && ray --version \
 && python3 - <<'PY'
import ray
import vllm
assert ray.__version__ == "2.58.0", ray.__version__
assert vllm.__version__ == "0.29.0", vllm.__version__
print(f"vLLM {vllm.__version__}; Ray {ray.__version__}")
PY
