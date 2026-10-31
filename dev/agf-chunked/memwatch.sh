#!/usr/bin/env bash
# Run a command and record the peak memory of its whole process tree (the R
# master plus forked chunk workers). Linux only: sums PSS from
# /proc/<pid>/smaps_rollup, which splits pages shared copy-on-write between
# processes fairly instead of counting them once per process (as RSS would).
#
#   dev/agf-chunked/memwatch.sh results/mem.csv -- Rscript ... args
#
# Appends "timestamp,peak_pss_gb,peak_procs,exit_code,command" to the CSV.
set -u
out="$1"; shift
[ "$1" = "--" ] && shift
"$@" &
root=$!
peak=0; peak_n=0
tree() {  # pids of root and all descendants
    local p=$1; echo "$p"
    for c in $(cat /proc/"$p"/task/*/children 2>/dev/null); do tree "$c"; done
}
while kill -0 "$root" 2>/dev/null; do
    tot=0; n=0
    for p in $(tree "$root"); do
        kb=$(awk '/^Pss:/ {print $2}' /proc/"$p"/smaps_rollup 2>/dev/null)
        [ -n "$kb" ] && tot=$((tot + kb)) && n=$((n + 1))
    done
    [ "$tot" -gt "$peak" ] && peak=$tot && peak_n=$n
    sleep 0.5
done
wait "$root"; code=$?
mkdir -p "$(dirname "$out")"
[ -f "$out" ] || echo "timestamp,peak_pss_gb,peak_procs,exit_code,command" > "$out"
gb=$(awk -v k="$peak" 'BEGIN {printf "%.2f", k / 1048576}')
echo "$(date -Iseconds),$gb,$peak_n,$code,\"$*\"" >> "$out"
echo "memwatch: peak $gb GB over $peak_n processes (exit $code)" >&2
exit "$code"
