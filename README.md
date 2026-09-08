# vllm-ray-head — DeepSeek V4 Flash 0731 雙節點 vLLM + Ray 部署（GX10-1 / Head + API）

> 本文件只講**操作**：啟動、重啟、注意事項、debug。
> 實際設定檔（`compose.yaml`、`start-head-and-vllm.sh`、`Dockerfile`）以 repo 內檔案為準，這裡不重複貼內容。

本 repo 是 GX10-1 的部署（Ray Head + vLLM API）。另一台 GX10-2 為 Ray Worker，對應 repo `~/vllm-ray-worker`。

## 目前實際配置（2026-09-09）

| 項目 | 值 |
|---|---|
| 節點 / 角色 | GX10-1 `gx10-5173` — Ray Head + vLLM API（容器 `deepseek-v4-ray-head`） |
| Ray Worker | GX10-2 — 容器 `deepseek-v4-ray-worker`（repo `~/vllm-ray-worker`） |
| Image | `eugr/spark-vllm-b12x:latest`（vLLM `0.1.dev20596` 2026-09-07 nightly，內含 Ray 2.58.0） |
| Topology | Ray、**TP=2、PP=1**（PP 已拿掉，因 DSpark 不支援 pipeline parallel） |
| 模型 | `deepseek-ai/DeepSeek-V4-Flash-0731` → served id `deepseek-v4-flash-0731` |
| API | `http://127.0.0.1:8000`（health `/health`） |
| DSpark | 開啟，`--speculative-config dspark`（5 speculative tokens、probabilistic） |
| Serving flags 重點 | `fp8_ds_mla`、`block-size 256`、`max-model-len 200000`、`max-num-seqs 3`、`max-num-batched-tokens 4096`、`gpu-memory-utilization 0.75`、B12X backends（`moe/linear/attention`） |
| Env（`.env`，gitignored） | `NODE_IP`、`HEAD_IP`、`ROCE_IFACE`、`HOST_HOME`、`HF_TOKEN` |

> `VLLM_PP_LAYER_PARTITION` 在 `.env` 裡仍是舊的 `14,29`，**B12X TP=2/PP=1 已不使用**，可忽略（compose 未引用）。

## 檔案分工（不需要再重新生出這些檔）

- `compose.yaml` — head service `ray-head`：eugr image、host network、`gpus: all`、healthcheck（`:8000`）、entrypoint 指向 `start-head-and-vllm.sh`、掛載 HF / vLLM 快取。
- `start-head-and-vllm.sh` — entrypoint：`ray stop` → `ray start --head` → 等待第二節點（2 nodes / 2 GPUs）→ `exec vllm serve`（含 B12X + DSpark flags）。
- `Dockerfile` — 目前使用預拉好 image，通常不需要 build。
- `patches/` — eugr B12X / SM121 相關 patch（僅供 future build 參考）。

## 啟動

### 標準順序：先 Worker、後 Head

兩邊都要就緒，head 的腳本會等第二節點出現，才開始載模型。

**GX10-2（Worker）**
```bash
cd ~/vllm-ray-worker
docker compose --env-file .env up -d ray-worker
```

**GX10-1（Head + API）**
```bash
cd ~/vllm-ray-head
docker compose --env-file .env up -d ray-head
docker compose logs -f ray-head
```

Head log 依序出現這些代表步驟已跑：

```text
[2/4] Starting Ray head ...
Ray cluster: alive_nodes=1, GPU=1.0      # worker 尚未加入
Ray cluster: alive_nodes=2, GPU=2.0      # worker 已加入
[4/4] Starting eugr B12X vLLM ...
Application startup complete.            # API 就緒
```

確認 API：

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

## 重啟

| 改了什麼 | 動作 | 需要 build？ |
|---|---|---|
| `start-head-and-vllm.sh` 的 vLLM 參數 | `docker compose up -d --force-recreate ray-head` | 否（bind-mount，腳本會重跑） |
| `.env` / `compose.yaml`（PP、IP、RoCE、volume…） | `docker compose up -d --force-recreate ray-head` | 否 |
| `Dockerfile` / 想重建 image | `docker compose build`（懷疑 cache 問題才 `--no-cache`），再 `--force-recreate` | 是 |
| 只重啟整個服務（乾淨） | `docker compose down --remove-orphans` → 依「啟動」順序重啟 | 視情況 |

> 改了 IP / RoCE / topology 時，最穩做法是**兩邊都 down**，再依 Worker → Head 順序重建，避免舊 Ray session / GPU actor 殘留。

## 注意事項

- **`.env` 含真實 `HF_TOKEN`，已被 `.gitignore` 排除；任何情況下不要把 token 或 `.env` commit 進 git。**
- **DSpark 必須 PP=1**。現行 TP=2 / PP=1；若改回 `--pipeline-parallel-size 2`，DSpark 會在載入前失敗（`dspark with pipeline parallel is not supported`）。
- **SM121 限制**：GB10 是 sm_121，必須用 B12X build（`eugr/spark-vllm-b12x`），mainline image 的 DeepSeek V4 Flash recipe 用 MTP 不是 DSpark。
- **client 必須用 served id `deepseek-v4-flash-0731`，不是 HF repo id** `deepseek-ai/DeepSeek-V4-Flash-0731`，否則回 404。
- **記憶體**：GB10 unified memory 與 ComfyUI 共用的 121 GiB pool 競爭；模型權重約 ~85 GiB/GPU（fixed），KV 不跨 TP GPU 切割。RAM 吃緊屬預期，真要看佔用看 `nvidia-smi` process table + `free -h`，別只看 `docker stats`/`MemFree`。
- **`reasoning_effort`**：top-level HTTP 參數只會開/關 thinking，無法設定 high/low；要設定要送 `chat_template_kwargs: {thinking: true, reasoning_effort: "..."}`。
- **`ROCE_IFACE` 大小寫敏感、每台不同**。此機為 `enp1s0f0np0`；worker 的值以自己的 `ip -4 route get 192.168.100.1` 回傳的 netdev 為準，別用這台的介面名直接套過去。
- healthcheck `restart: unless-stopped`；容器重啟會由腳本重跑 Ray head + vLLM。

## Debug

```bash
# 健康
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models

# 日誌
docker compose logs -f --tail=200 ray-head

# 實際版本 + 真實啟動參數（以 live 進程為準，別只信 entrypoint 檔）
docker exec deepseek-v4-ray-head sh -c 'python3 -c "import vllm; print(vllm.__version__)"'
docker exec deepseek-v4-ray-head sh -c 'tr "\0" " " </proc/1/cmdline; echo'
docker exec deepseek-v4-ray-head sh -c 'ray status'   # 期望 2 nodes / 2 GPUs

# 記憶體 / GPU（unified pool）
nvidia-smi
free -h
sudo dmesg -T | grep -Ei 'out of memory|oom-killer|killed process|nvrm|xid'

# vLLM「non-default args」與 KV 容量
docker compose logs ray-head | grep -E 'non-default args|GPU KV cache size|Application startup complete'
```

常見徵狀與方向：

| 徵狀 | 方向 |
|---|---|
| Head log 停在 `alive_nodes=1` | Worker 沒起來 / 連不上 RoCE；`ping 192.168.100.2`、`nc -vz 192.168.100.2 6379` |
| `No node info found matching attributes` / 連不上 Ray | 別把 Ray Head 與 vLLM 拆兩個容器；確認共用 host network 與 same image |
| 載入時 `dspark with pipeline parallel is not supported` | 把 `--pipeline-parallel-size` 拿掉（PP=1） |
| 載入時 FlashInfer / `family(100)` 類錯誤 | SM121 硬限制 → 確認使用 B12X image，別用 mainline |
| API 404 `does not exist` | client 用了 HF repo id，改用 `deepseek-v4-flash-0731` |
| RAM/OOM、worker dead | 統一記憶體被 ComfyUI 吃掉或 KV/context 設太大；調 `max-num-seqs`、`max-model-len` |
