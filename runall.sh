#!/bin/bash

set -eu

# Usage: ./runall.sh

IFS=' '
read -ra os <<< "$(uname -a)"
os=${os[0]}

if [[ $os ==  MINGW* ]];
then
    # git bash on Windows
    exe_folder="./x64/Release";
    ext=".exe";
else
    # UNIX-like OS
    exe_folder=".";
    ext=".out";
fi

run_script="run.sh"
algorithms="KargerMinCut KargerSteinMinCut KargerMinCutTimeout"
full_contractions=(1 0 1)

mkdir -p "benchmark"

output_file=$(date '+%Y%m%d_%H%M%S_%3N')

n=0
for algorithm in ${algorithms}; do
    echo "Running ${algorithm}..."

    program="${algorithm}${ext}"

    full_contraction=${full_contractions[$n]}

    ./${run_script} "${exe_folder}/${program}" ${full_contraction} > "benchmark/${algorithm}_${output_file}.csv"

    n=$(($n+1))
done
