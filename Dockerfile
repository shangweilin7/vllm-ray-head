FROM vllm/vllm-openai:v0.26.0

RUN python3 -m pip install --no-cache-dir --upgrade "ray[cgraph]" \
 && python3 -m pip uninstall -y cupy-cuda12x
