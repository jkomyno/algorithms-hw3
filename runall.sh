#!/bin/bash

set -eu

# Usage: ./runall.sh

run_script="run.sh"
programs="KargerMinCut KargerSteinMinCut"
full_contractions="1 0"

mkdir -p "benchmark"

output_file=$(date '+%Y%m%d_%H%M%S_%3N')

n=0
for program in ${programs}; do
    echo "Running ${program}..."

    full_contraction=${full_contractions[$n]}

    ./${run_script} ${program} ${full_contraction} > "benchmark/${output_file}.csv"

    n=$(($n+1))
done
