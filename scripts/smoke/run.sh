#!/usr/bin/env bash
# =============================================================================
# Meshy Service-stack test entry point.
#
# Consolidates the per-module smoke tests and the two integration topologies
# behind one reproducible command with a PASS/FAIL summary.
#
# Usage:
#   scripts/smoke/run.sh [--large] [TEST ...]
#
#   TEST is one or more of:
#     infer      module: SGLangInferenceService (start/ready/generate/marker)
#     colo_mem   module: colocate release/resume GPU memory hand-off
#     train      module: TitanTrainingService (TQ samples -> step -> gen gate -> dump)
#     agentloop  module: unified AgentLoop driver (TQ + fake trainer, lock-step)
#     disagg     integration: disaggregate (torchrun, N inference + 1 training)
#     colocate   integration: colocate (single-pass launcher, shared cards)
#   With no TEST given, runs all of them in order.
#
#   --large   scale preset: 1.7B model, disagg=8 cards (4 infer + 4-card FSDP),
#             colocate=4 cards (tp4 inference + 4-card FSDP), bigger batches.
#
# Everything is env-overridable; see the knobs block below. Cleanup between
# tests is targeted (launcher / torchrun / sglang / recipe), never a blanket
# `pkill python`.
# =============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

PY="${XRL_PY:-python3}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ── knobs (all overridable) ──────────────────────────────────────────────────
MODULE_MODEL_ID="${XRL_MODULE_MODEL_ID:-Qwen/Qwen3-0.6B}"
MODULE_FLAVOR="${XRL_MODULE_FLAVOR:-0.6B}"

SCALE="small"
for a in "$@"; do [ "$a" = "--large" ] && SCALE="large"; done

if [ "$SCALE" = "large" ]; then
    INT_MODEL_ID="${XRL_INT_MODEL_ID:-Qwen/Qwen3-1.7B}"
    INT_FLAVOR="${XRL_INT_FLAVOR:-1.7B}"
    DISAGG_GPUS="${XRL_DISAGG_GPUS:-0,1,2,3,4,5,6,7}"; DISAGG_NGPUS=8; DISAGG_TRAIN=4
    COLO_GPUS="${XRL_COLO_GPUS:-0,1,2,3}"; COLO_NGPUS=4
    INT_SEQ_LEN="${XRL_SEQ_LEN:-2048}"; INT_MAX_NEW="${XRL_MAX_NEW_TOKENS:-256}"
    INT_ROLLOUT_BATCH="${XRL_ROLLOUT_BATCH:-8}"; INT_GROUP="${XRL_GROUP_SIZE:-4}"
    INT_TARGET_VERSION="${XRL_TARGET_VERSION:-3}"; INT_TIMEOUT="${XRL_TIMEOUT:-900}"
else
    INT_MODEL_ID="${XRL_INT_MODEL_ID:-Qwen/Qwen3-0.6B}"
    INT_FLAVOR="${XRL_INT_FLAVOR:-0.6B}"
    DISAGG_GPUS="${XRL_DISAGG_GPUS:-0,1}"; DISAGG_NGPUS=2; DISAGG_TRAIN=1
    COLO_GPUS="${XRL_COLO_GPUS:-0}"; COLO_NGPUS=1
    INT_SEQ_LEN="${XRL_SEQ_LEN:-512}"; INT_MAX_NEW="${XRL_MAX_NEW_TOKENS:-64}"
    INT_ROLLOUT_BATCH="${XRL_ROLLOUT_BATCH:-2}"; INT_GROUP="${XRL_GROUP_SIZE:-2}"
    INT_TARGET_VERSION="${XRL_TARGET_VERSION:-3}"; INT_TIMEOUT="${XRL_TIMEOUT:-600}"
fi

MODULE_GPU="${XRL_MODULE_GPU:-0}"

declare -A RESULT

# ── helpers ──────────────────────────────────────────────────────────────────
resolve_model() {  # <hf-id> -> prints local snapshot path
    "$PY" - "$1" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_files_only=True))
PY
}

cleanup() {  # targeted teardown of worker-stack processes.
    # NB: patterns must NOT match this runner ("bash scripts/smoke/run.sh") or
    # cleanup would kill itself. Module smoke scripts run synchronously and
    # self-clean (finally + PR_SET_PDEATHSIG), so only engine leftovers matter.
    pkill -INT -f "scripts/launch.py"       2>/dev/null
    sleep 2
    pkill -9   -f "torch.distributed.run"   2>/dev/null
    pkill -9   -f "recipe.grpo_gsm8k"       2>/dev/null
    pkill -9   -f "sglang.launch_server"    2>/dev/null
    sleep 2
}

banner() { echo; echo "==================== $* ===================="; }

# run a python module smoke test; PASS iff it prints "RESULT: PASS"
run_module() {  # <name> <script> <gpu> [extra env assignments...]
    local name="$1" script="$2" gpu="$3"; shift 3
    local log="/tmp/xrl_${name}.log"
    banner "MODULE ${name} (gpu ${gpu})"
    cleanup; rm -f "$log"
    local mdl; mdl="$(resolve_model "$MODULE_MODEL_ID")"
    ( env "$@" HF_ENDPOINT="$HF_ENDPOINT" CUDA_VISIBLE_DEVICES="$gpu" \
        XRL_MODEL="$mdl" XRL_RUNTIME_DIR="/tmp/xrl_rt_${name}" \
        "$PY" "$script" ) >"$log" 2>&1
    if grep -q "RESULT: PASS" "$log"; then
        RESULT[$name]=PASS; echo "[$name] PASS"; grep -E "^\[" "$log" | tail -6
    else
        RESULT[$name]=FAIL; echo "[$name] FAIL (tail:)"; tail -15 "$log"
    fi
    cleanup
}

# launch an integration topology via the launcher, poll the log until the
# trainer version reaches a target (PASS) or an error/timeout occurs (FAIL).
run_integration() {  # <name> <topology> <gpus> <ngpus> [train_gpus]
    local name="$1" topo="$2" gpus="$3" ngpus="$4" train_gpus="${5:-}"
    local log="/tmp/xrl_${name}.log" rt="/tmp/xrl_rt_${name}"
    banner "INTEGRATION ${name} (${topo}, gpus ${gpus}, target v${INT_TARGET_VERSION})"
    cleanup; rm -rf "$rt" "$log"
    local mdl; mdl="$(resolve_model "$INT_MODEL_ID")"

    # Build the child env as an array: a conditional expansion in the bare
    # assignment-prefix position (${x:+VAR=..}) silently breaks assignment-word
    # recognition, so subsequent VAR=val tokens get run as a command. `env`
    # sidesteps that entirely.
    local -a envv=(
        CUDA_VISIBLE_DEVICES="$gpus" HF_ENDPOINT="$HF_ENDPOINT"
        XRL_TOPOLOGY="$topo" XRL_NGPUS="$ngpus"
        XRL_MODEL="$mdl" XRL_MODEL_NAME=qwen3 XRL_MODEL_FLAVOR="$INT_FLAVOR"
        XRL_SEQ_LEN="$INT_SEQ_LEN" XRL_MAX_NEW_TOKENS="$INT_MAX_NEW"
        XRL_ROLLOUT_BATCH="$INT_ROLLOUT_BATCH" XRL_GROUP_SIZE="$INT_GROUP"
        XRL_EPOCHS=1 XRL_RUNTIME_DIR="$rt"
    )
    [ -n "$train_gpus" ] && envv+=(XRL_TRAIN_GPUS="$train_gpus")

    env "${envv[@]}" "$PY" scripts/launch.py --recipe recipe.grpo_gsm8k >"$log" 2>&1 &
    local pid=$!

    local ok=0 reason="timeout" deadline=$((SECONDS + INT_TIMEOUT))
    while [ $SECONDS -lt $deadline ]; do
        if grep -qE "Traceback|cannot be accessed from Triton|EADDRINUSE|should be called only when server is idle|CUDA error|out of memory|OutOfMemory" "$log"; then
            reason="error-in-log"; break
        fi
        local v; v="$(grep -oE "version advanced [0-9]+ -> [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+$")"
        if [ -n "${v:-}" ] && [ "$v" -ge "$INT_TARGET_VERSION" ]; then ok=1; reason="reached v$v"; break; fi
        kill -0 "$pid" 2>/dev/null || { reason="launcher-exited"; break; }
        sleep 3
    done

    # Verify that each training step produced a checkpoint for inference.
    local wdirs
    wdirs="$(ls -d "$rt"/weights/*/v* 2>/dev/null | wc -l | tr -d ' ')"
    echo "[$name] ok=$ok reason=$reason weight_dumps=$wdirs"
    grep -E "cards discovered|marker published|AgentLoop: inference|version advanced|synced weights" "$log" | tail -8
    if [ "$ok" = 1 ] && [ "${wdirs:-0}" -ge 1 ]; then RESULT[$name]=PASS; else RESULT[$name]=FAIL; tail -20 "$log"; fi
    cleanup
}

# ── test registry ────────────────────────────────────────────────────────────
do_infer()     { run_module infer     scripts/smoke/infer.py     "$MODULE_GPU"; }
do_colo_mem()  { run_module colo_mem  scripts/smoke/infer.py     "$MODULE_GPU" COLO=1; }
do_train()     { run_module train     scripts/smoke/train.py     "$MODULE_GPU" XRL_MODEL_FLAVOR="$MODULE_FLAVOR" XRL_SEQ_LEN=512; }
do_agentloop() { run_module agentloop scripts/smoke/agentloop.py "$MODULE_GPU"; }
do_disagg()    { run_integration disagg   disaggregate "$DISAGG_GPUS" "$DISAGG_NGPUS" "$DISAGG_TRAIN"; }
do_colocate()  { run_integration colocate colocate     "$COLO_GPUS"   "$COLO_NGPUS"; }

# ── main ─────────────────────────────────────────────────────────────────────
TESTS=()
for a in "$@"; do [ "$a" = "--large" ] || TESTS+=("$a"); done
[ ${#TESTS[@]} -eq 0 ] && TESTS=(infer colo_mem train agentloop disagg colocate)

echo "scale=$SCALE module_model=$MODULE_MODEL_ID int_model=$INT_MODEL_ID tests=${TESTS[*]}"
for t in "${TESTS[@]}"; do
    case "$t" in
        infer) do_infer ;; colo_mem) do_colo_mem ;; train) do_train ;;
        agentloop) do_agentloop ;; disagg) do_disagg ;; colocate) do_colocate ;;
        *) echo "unknown test: $t"; RESULT[$t]=SKIP ;;
    esac
done

banner "SUMMARY (scale=$SCALE)"
fail=0
for t in "${TESTS[@]}"; do
    r="${RESULT[$t]:-?}"; printf "  %-10s %s\n" "$t" "$r"
    [ "$r" = PASS ] || fail=1
done
exit $fail
