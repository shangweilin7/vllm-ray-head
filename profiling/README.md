# DSpark ON/OFF A/B memory comparison — 2026-09-08

Goal: prove/refute whether DSpark itself pins ~20 GB, by diffing model-load
initial-free-memory under the SAME version + SAME fixed KV budget with
DSpark ON vs OFF.

## Step 1 — Docker image / version (ON)
- Image/tag: `eugr/spark-vllm-b12x:latest`
- In-container version:
  `python3 -c "import vllm; print(vllm.__version__); print(vllm.__file__)"`
  → `0.1.dev20596+g2a979314d.d20260907`
  → `/usr/local/lib/python3.12/dist-packages/vllm/__init__.py`

## Step 2 — vmstat 1 10 DURING generation (DSpark ON)
Both hosts running active decode; si/so columns (KB/s, after the first
average row). swpd = swap already resident = leftover, not active R/W.

HEAD (gx10-5173, 192.168.100.1): swpd ~7.68 GiB
  si:  0 0 68 4 0 0 4 0 40 4   (mostly 0, tiny spikes)
  so:  0 0  0 0 0 0 0 0  0 0   (all 0)
 → no sustained swap I/O during generation.

WORKER (gx10-f036, 192.168.100.2): swpd ~2.14 GiB
  si:  0 0 0 0 0 0 0 0 0 0     (all 0)
  so:  0 0 0 0 0 0 0 0 0 0     (all 0)
 → no swap I/O at all during generation.

Full output: profiling/dspark_ON/vmstat ... (see profiling/vmstat_dspark_on.sh + head/worker txt)

## Step 3 — A/B model-load memory (same version, same KV 7.2 GiB)
kv-cache-memory-bytes 7730941133 = 7.2 GiB for BOTH runs (untouched).
"Initial free memory" = GPU free AFTER model (+ spec) load, before KV reserve.

DSpark ON  (docker log dspark_ON_docker.log):
  Worker_TP0 (head): Initial free memory 106.18 GiB
  Worker_TP1 (ip=192.168.100.2): Initial free memory 112.18 GiB
  Both reserved 7.2 GiB KV (skipped profiling).

DSpark OFF (docker log dspark_OFF_docker.log — capture after manual restart):
  Worker_TP0: TBD
  Worker_TP1: TBD
→ diff = DSpark's model-load footprint.

## free -h (for reference)
DSpark ON:
  HEAD:   used 109Gi  avail 12Gi | swap 15Gi total, 7.3Gi used
  WORKER: used 104Gi  avail 17Gi | swap 15Gi total, 2.1Gi used
DSpark OFF (after restart to ready):
  HEAD:   TBD
  WORKER: TBD
