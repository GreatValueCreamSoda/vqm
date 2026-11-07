#!/usr/bin/env bash
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
#
# Nsight Systems captures for every code path of both metrics. The capture
# matrix, the gateway discipline (never `nsys stats`) and the attribution rules
# are the ones AGENTS.md documents; this script is that section, executable.
#
#   bench/profile_nsys.sh [--metric ssimu2|butteraugli|all]
#                         [--mode all|resident|streaming|half-planes|workers]
#                         [--pair ORIG DIST] [--out DIR]
#                         [--counters SET|none] [--iterations N] [--keep]
#
# Every capture writes <out>/<metric>-<mode>.nsys-rep plus a <metric>-<mode>.txt
# summary holding the gateway's report facts, the dispatch count and the
# binary's own unprofiled stage table (the profiler's layer costs 1.3-1.8% per
# pair, so timing claims never come from the profiled run).
set -euo pipefail

readonly REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
readonly BENCH="$REPO_ROOT/build/vqm_bench"
readonly SKILL_DIR=/opt/nvidia-nsight-systems-bin/skills/nsight-systems
readonly GATEWAY="$SKILL_DIR/scripts/nsys_skill_cli.py"
readonly ADMIN_GATE=/sys/module/nvidia/parameters/NVreg_RestrictProfilingToAdminUsers

metric=all
mode=all
pair_orig=/tmp/uhd_orig.png
pair_dist=/tmp/uhd_dist.png
out_dir=
counters=none
iterations=
keep=0
list=0

usage() {
    cat <<'EOF'
Usage: bench/profile_nsys.sh [--metric ssimu2|butteraugli|all]
       [--mode all|resident|streaming|half-planes|workers]
       [--pair ORIG DIST] [--out DIR] [--counters SET|none]
       [--iterations N] [--keep] [--list]
Defaults: --metric all --mode all --pair /tmp/uhd_orig.png /tmp/uhd_dist.png
          --out /tmp/gpu-prof/<date> --counters none
Each capture keeps its .nsys-rep and a .txt summary; --keep also keeps the
profiled bench log.
--list prints the capture matrix without running a profiler.
EOF
}

fail() {
    printf 'profile_nsys: %s\n' "$*" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case $1 in
        --metric) metric=${2:?--metric needs a value}; shift 2 ;;
        --mode) mode=${2:?--mode needs a value}; shift 2 ;;
        --pair)
            pair_orig=${2:?--pair needs two paths}
            pair_dist=${3:?--pair needs two paths}
            shift 3 ;;
        --out) out_dir=${2:?--out needs a value}; shift 2 ;;
        --counters) counters=${2:?--counters needs a value}; shift 2 ;;
        --iterations) iterations=${2:?--iterations needs a value}; shift 2 ;;
        --keep) keep=1; shift ;;
        --list) list=1; shift ;;
        --help | -h) usage; exit 0 ;;
        *) fail "unknown argument '$1' (try --help)" ;;
    esac
done

case $metric in ssimu2 | butteraugli | all) ;; *) fail "bad --metric '$metric'" ;; esac
case $mode in all | resident | streaming | half-planes | workers) ;; *)
    fail "bad --mode '$mode'" ;;
esac
if [ "$metric" = butteraugli ] && [ "$mode" = half-planes ]; then
    fail "--mode half-planes is available only for ssimu2"
fi
if [ -n "$iterations" ] && [[ ! "$iterations" =~ ^[1-9][0-9]*$ ]]; then
    fail "--iterations must be a positive integer"
fi

captures=()
for which in ssimu2 butteraugli; do
    [ "$metric" = all ] || [ "$metric" = "$which" ] || continue
    for this_mode in resident streaming half-planes workers; do
        [ "$mode" = all ] || [ "$mode" = "$this_mode" ] || continue
        [ "$which" != butteraugli ] || [ "$this_mode" != half-planes ] || continue
        captures+=("$which:$this_mode")
    done
done
if [ "$list" -eq 1 ]; then
    for entry in "${captures[@]}"; do
        printf '%s %s\n' "${entry%%:*}" "${entry#*:}"
    done
    exit 0
fi

# --- fixtures -------------------------------------------------------------
[ -f "$BENCH" ] || fail "$BENCH is missing; build it first (cmake --build build -j8)"
for image in "$pair_orig" "$pair_dist"; do
    [ -f "$image" ] || fail "$image is missing. /tmp fixtures vanish on reboot:
regenerate a pair of differing images of one size, e.g.
  ffmpeg -f lavfi -i testsrc=size=3840x2160:rate=1 -frames:v 1 $pair_orig"
done

# --- bootstrap ------------------------------------------------------------
[ -f "$GATEWAY" ] || fail "the Nsight Systems skill is missing; read
$SKILL_DIR/SKILL.md and install it, or pass a different --out workflow"
[ -f "$SKILL_DIR/SKILL.md" ] || fail "read $SKILL_DIR/SKILL.md first; it is missing"
# The skill's own bootstrap names the platform, the nsys CLI and the bundled
# Python; it is the only supported way to find them.
platform_line=$(
    cd "$SKILL_DIR" && sh scripts/determine_local_unix_platform.sh
) || fail "determine_local_unix_platform.sh failed; read $SKILL_DIR/SKILL.md"
NSYS=$(printf '%s\n' "$platform_line" | sed -n 's/^Nsys CLI: "\(.*\)"$/\1/p')
PY=$(printf '%s\n' "$platform_line" |
    sed -n 's/^Nsys Python interpreter: "\(.*\)"$/\1/p')
[ -n "$NSYS" ] && [ -x "$NSYS" ] || fail "could not resolve the nsys CLI; read $SKILL_DIR/SKILL.md"
[ -n "$PY" ] && [ -x "$PY" ] || fail "could not resolve the bundled Python; read $SKILL_DIR/SKILL.md"

# The gateway's report analysis needs duckdb/pandas/pyarrow in the user site:
# the bundled Python is not writable.
if ! "$PY" -c 'import duckdb, pandas, pyarrow' >/dev/null 2>&1; then
    printf 'profile_nsys: installing the gateway report dependencies into the user site\n'
    "$PY" -m pip install --user --quiet duckdb pandas pyarrow ||
        fail "could not install duckdb/pandas/pyarrow; read $SKILL_DIR/SKILL.md"
    "$PY" -c 'import duckdb, pandas, pyarrow' >/dev/null 2>&1 ||
        fail "the gateway report dependencies still do not import; read $SKILL_DIR/SKILL.md"
fi
# `doctor` also hashes the bundled skill pack; a local SKILL.md edit (or a
# pack built elsewhere) fails that check without touching the capture path, so
# it is reported and not fatal.
doctor=$("$PY" "$GATEWAY" doctor 2>&1 || true)
if ! printf '%s' "$doctor" | grep -q '"ok": true'; then
    printf 'profile_nsys: doctor reports a problem (continuing; report path is checked below)\n'
    printf '%s\n' "$doctor" | grep -E '"name"|"status"' | paste - - | grep fail || true
fi

# --- counters -------------------------------------------------------------
metric_args=()
if [ "$counters" != none ]; then
    admin=$(cat "$ADMIN_GATE" 2>/dev/null || echo unknown)
    if [ "$admin" != 0 ]; then
        printf 'profile_nsys: counters skipped: %s reads %s, not 0.\n' \
            "$ADMIN_GATE" "$admin"
        printf '  Add this line to /etc/modprobe.d/nvidia-profiling.conf:\n'
        printf '    options nvidia NVreg_RestrictProfilingToAdminUsers=0\n'
        printf '  then mkinitcpio -P and reboot (or run with sudo).\n'
        counters=none
    else
        metric_args+=("--gpu-metrics-devices=all" "--gpu-metrics-set=$counters")
    fi
fi

# --- output ---------------------------------------------------------------
if [ -z "$out_dir" ]; then
    out_dir="/tmp/gpu-prof/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$out_dir"
printf 'profile_nsys: captures go to %s\n' "$out_dir"

# --- capture --------------------------------------------------------------
# capture <metric> <mode>: the metric is the script's short name (ssimu2), the
# bench and the CLI spell it ssimulacra2.
capture() {
    local which=$1
    local bench_metric=$1
    if [ "$which" = ssimu2 ]; then bench_metric=ssimulacra2; fi
    local this_mode=$2
    shift 2
    local name="$which-$this_mode"
    local rep="$out_dir/$name"
    local summary="$out_dir/$name.txt"
    local arm=(--pair "$pair_orig" "$pair_dist")
    local resid=400 warm=20
    case $this_mode in
        streaming) resid=50; warm=5 ;;
        half-planes | workers) ;;
    esac
    case $this_mode in
        resident) arm+=(--resident 2) ;;
        streaming) ;;
        half-planes) arm+=(--resident 2 --half-planes) ;;
        workers) arm+=(--workers 2) ;;
    esac
    if [ -n "$iterations" ]; then resid=$iterations; warm=1; fi

    printf 'profile_nsys: %s (%s iterations)\n' "$name" "$resid"
    # -f takes a value in this nsys version; -o must not collide with a stale
    # report, hence -f true and a name unique to this run.
    "$NSYS" profile -f true -t vulkan,osrt -s process-tree \
        --vulkan-gpu-workload=individual "${metric_args[@]+"${metric_args[@]}"}" \
        -o "$rep" "$BENCH" --metric "$bench_metric" "${arm[@]}" \
        --iterations "$resid" --warmup "$warm" >"$out_dir/$name.bench.log" 2>&1 ||
        fail "nsys capture of $name failed; see $out_dir/$name.bench.log"

    local gateway_status=0 unprofiled_status=0
    {
        printf '== %s ==\n' "$name"
        printf 'report: %s.nsys-rep\n\n' "$rep"
        printf -- '-- report inventory --\n'
        "$PY" "$GATEWAY" report-context --report "$rep.nsys-rep" \
            --nsys-path "$NSYS" 2>&1 || gateway_status=$?
        printf '\n-- graphics api summary --\n'
        "$PY" "$GATEWAY" report-fact --report "$rep.nsys-rep" \
            --nsys-path "$NSYS" --intent graphics_api_summary 2>&1 || gateway_status=$?
        printf '\n-- graphics api distribution --\n'
        # max-rows matters: the distribution is ranked by total time, and the
        # command-recording calls (vkCmdBindPipeline and friends) are cheap
        # enough to fall off the default page even though they are what the
        # attribution rule counts.
        "$PY" "$GATEWAY" report-fact --report "$rep.nsys-rep" \
            --nsys-path "$NSYS" --intent graphics_api_distribution \
            --max-rows 300 2>&1 || gateway_status=$?
        printf '\n-- dispatch attribution --\n'
        # The Vulkan layer does not enumerate vkCmdDispatch and reports one GPU
        # workload row per command buffer, so the dispatch count is the
        # vkCmdBindPipeline count (one bind per dispatch). The gateway's
        # distribution lists the API functions; sum the bind entries.
        "$PY" "$GATEWAY" report-fact --report "$rep.nsys-rep" \
            --nsys-path "$NSYS" --intent graphics_api_distribution \
            --max-rows 300 2>&1 |
            "$PY" -c '
import json, sys
try:
    payload = json.load(sys.stdin)
except Exception:
    print("vkCmdBindPipeline count: unavailable (gateway returned no JSON)")
    sys.exit(1)
if not payload.get("ok", False):
    print("vkCmdBindPipeline count: unavailable (gateway reported an error)")
    sys.exit(1)
entries = []
def walk(node):
    if isinstance(node, dict):
        label = node.get("api_name") or node.get("name") or node.get("function")
        count = node.get("count")
        if isinstance(label, str) and isinstance(count, (int, float)):
            entries.append((label, int(count)))
        for value in node.values():
            walk(value)
    elif isinstance(node, list):
        for value in node:
            walk(value)
walk(payload)
if not any(label == "vkCmdBindPipeline" for label, count in entries):
    print("vkCmdBindPipeline count: unavailable (no binding data)")
    sys.exit(1)
binds = sum(count for label, count in entries if label == "vkCmdBindPipeline")
dispatches = sum(
    count for label, count in entries
    if label in ("vkCmdDispatch", "vkCmdDispatchIndirect")
)
print(f"vkCmdBindPipeline total: {binds} (one bind per dispatch)")
if dispatches:
    print(f"vkCmdDispatch total: {dispatches}")
' 2>&1 || gateway_status=$?
        printf '\n-- unprofiled stage table (timing claims come from here) --\n'
        "$BENCH" --metric "$bench_metric" "${arm[@]}" --iterations "$resid" \
            --warmup "$warm" 2>&1 || unprofiled_status=$?
    } >"$summary"
    if [ "$unprofiled_status" -ne 0 ]; then
        fail "unprofiled run of $name exited $unprofiled_status; see $summary"
    fi
    if [ "$gateway_status" -ne 0 ]; then
        fail "gateway analysis of $name exited $gateway_status; see $summary"
    fi
    printf 'profile_nsys: wrote %s\n' "$summary"
    if [ "$keep" -eq 0 ]; then
        rm -f "$out_dir/$name.bench.log"
    fi
}

for entry in "${captures[@]}"; do
    capture "${entry%%:*}" "${entry#*:}"
done

printf 'profile_nsys: done; summaries in %s\n' "$out_dir"
