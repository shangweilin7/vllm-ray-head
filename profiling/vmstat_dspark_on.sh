#!/usr/bin/env bash
# Item 2: vmstat 1 10 on both hosts DURING active generation (DSpark ON).
set -u
WORKER=shang@192.168.100.2
OUT=/tmp/vmstat_dspark_on

rm -f ${OUT}_head.txt ${OUT}_worker.txt ${OUT}_req.out

# Start worker vmstat first (ssh), then head vmstat, ~simultaneous.
ssh -o StrictHostKeyChecking=no "$WORKER" "vmstat 1 10 > /tmp/vmstat_worker.txt 2>&1" &
WPID=$!
vmstat 1 10 > ${OUT}_head.txt 2>&1 &
HPID=$!
sleep 0.5

# Fire a long generation so decode is active across the vmstat window (~10-16 tok/s).
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "deepseek-v4-flash-0731",
    "messages": [{"role":"user","content":"Please write a long, detailed technical essay on distributed systems and memory management, going into depth and length. Do not stop early."}],
    "max_tokens": 600,
    "temperature": 0.7,
    "stream": false
  }' > ${OUT}_req.out 2>&1 &
CPID=$!

# Let both vmstat windows (10s each) fully capture active generation.
wait $HPID
# worker vmstat ends ~same time; ensure it finished
kill -0 $WPID 2>/dev/null && wait $WPID

# Stop the ongoing generation request now that the window is captured.
kill $CPID 2>/dev/null

echo "===== HEAD vmstat 1 10 (DSpark ON) ====="
cat ${OUT}_head.txt
echo
echo "===== WORKER (192.168.100.2) vmstat 1 10 (DSpark ON) ====="
ssh -o StrictHostKeyChecking=no "$WORKER" "cat /tmp/vmstat_worker.txt"
echo
echo "===== generation request status ====="
tail -c 300 ${OUT}_req.out
echo
