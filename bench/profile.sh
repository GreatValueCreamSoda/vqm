#!/bin/bash
# Derived from Cloudinary SSIMULACRA 2 (Copyright (c) Jon Sneyers, Cloudinary).
# BSD-3-Clause; see LICENSE.
#
# Profiles the metric: sampling profiler when available (perf, then callgrind),
# otherwise the built-in stage timing from vqm_bench, which is always
# available and needs no privileges. Run from the repository root.
#
# For GPU-side captures -- both metrics, every code path, Nsight Systems with
# per-stage attribution -- use bench/profile_nsys.sh instead; this script never
# touches the GPU timeline.
set -euo pipefail

build_dir=build
out_dir=artifacts/reports/profile
tool=auto
iterations=20
pairs=()

usage() {
  cat >&2 <<EOF
usage: $0 [--build-dir DIR] [--out-dir DIR] [--tool auto|perf|callgrind|stages]
          [--iterations N] [--pair ORIG DIST]...
  --out-dir DIR  profiler data directory (default: artifacts/reports/profile)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --build-dir) build_dir="${2:?--build-dir needs a value}"; shift 2 ;;
    --out-dir) out_dir="${2:?--out-dir needs a value}"; shift 2 ;;
    --tool) tool="${2:?--tool needs a value}"; shift 2 ;;
    --iterations) iterations="${2:?--iterations needs a value}"; shift 2 ;;
    --pair) pairs+=("${2:?--pair needs two paths}" "${3:?--pair needs two paths}"); shift 3 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done

bench="$build_dir/vqm_bench"
if [ ! -x "$bench" ]; then
  echo "missing $bench; build it with:" >&2
  echo "  cmake --build $build_dir --target vqm_bench" >&2
  exit 1
fi

if [ ${#pairs[@]} -eq 0 ]; then
  root=validation/data/CID22_validation_set
  if [ -d "$root/original" ] && [ -d "$root/compressed" ] &&
     original=$(find "$root/original" -name '*.png' | sort | sed -n '1p') &&
     [ -f "$original" ]; then
    stem=$(basename "$original" .png)
    compressed="$root/compressed/$stem"
    if [ -d "$compressed" ] &&
       distorted=$(find "$compressed" -type f \
         \( -name '*.png' -o -name '*.jpg' \) | sort | sed -n '1p') &&
       [ -n "$distorted" ]; then
      pairs=("$original" "$distorted")
    fi
  fi
fi
if [ ${#pairs[@]} -eq 0 ]; then
  if [ -f "$build_dir/fixtures/gradient.png" ] && [ -f "$build_dir/fixtures/gradient_B.png" ]; then
    pairs=("$build_dir/fixtures/gradient.png" "$build_dir/fixtures/gradient_B.png")
  else
    echo "no pair given and no CID22 data or fixtures found" >&2
    echo "run 'python3 validation/fetch_cid22.py' or 'python3 tests/make_goldens.py'" >&2
    exit 2
  fi
fi

pair_args=()
for ((i = 0; i < ${#pairs[@]}; i += 2)); do
  pair_args+=(--pair "${pairs[i]}" "${pairs[i + 1]}")
done

run_stages() {
  "$bench" --iterations "$iterations" "${pair_args[@]}"
}

perf_usable() {
  command -v perf >/dev/null 2>&1 || return 1
  # The sysctl alone does not determine whether this process can open the
  # default sampling event; capabilities and other policy can restrict it.
  local probe status=0
  probe=$(mktemp "${TMPDIR:-/tmp}/vqm-perf-probe.XXXXXX") || return 1
  perf record -g --call-graph dwarf -o "$probe" -- true >/dev/null 2>&1 || status=$?
  rm -f "$probe"
  return "$status"
}

run_perf() {
  mkdir -p "$out_dir"
  local data="$out_dir/perf.data"
  echo "== perf record (-g) -> $data"
  perf record -g --call-graph dwarf -o "$data" -- \
    "$bench" --iterations "$iterations" "${pair_args[@]}"
  echo "== perf report"
  perf report --stdio --percent-limit 1 -i "$data"
}

valgrind_usable() {
  command -v valgrind >/dev/null 2>&1 && command -v callgrind_annotate >/dev/null 2>&1
}

run_callgrind() {
  mkdir -p "$out_dir"
  local out="$out_dir/callgrind.out"
  echo "== callgrind -> $out"
  valgrind --tool=callgrind --callgrind-out-file="$out" -- \
    "$bench" --iterations "$iterations" "${pair_args[@]}"
  echo "== callgrind_annotate (threshold 95%)"
  callgrind_annotate --threshold=95 "$out"
}

case "$tool" in
  perf)
    if ! perf_usable; then
      echo "perf sampling is unavailable for this process; check perf permissions" >&2
      exit 1
    fi
    run_perf
    ;;
  callgrind)
    if ! valgrind_usable; then
      echo "valgrind is not installed; use --tool stages instead" >&2
      exit 1
    fi
    run_callgrind
    ;;
  stages)
    run_stages
    ;;
  auto)
    if perf_usable; then
      run_perf
    elif valgrind_usable; then
      run_callgrind
    else
      echo "no sampling profiler available (perf, valgrind); using the"
      echo "built-in stage timing. Check perf sampling permissions for a"
      echo "symbol-level profile."
      run_stages
    fi
    ;;
  *)
    usage
    exit 2
    ;;
esac
