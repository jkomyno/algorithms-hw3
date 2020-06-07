#!/bin/bash

set -eu

# Usage: ./run.sh [program-name] [measure-full-contraction (0/1)] > output.csv
# Note: this file needs the LF line endings.

# program=$1
# measure_full_contraction=$2
program="./x64/Release/KargerMinCut.exe"
measure_full_contraction="1"
dataset="dataset"

header="filename;expected_min_cut;nodes;k;min_cut;program_time;discovery_time"

if [ $measure_full_contraction -eq "1" ]; then
    header="${header};full_contraction"
fi

echo $header

for input_file in ${dataset}/input_random*.txt; do
  IFS='_' read -ra ADDR <<< "$input_file"

  # number of nodes in the graph
  nodes=${ADDR[3]%.txt}

  output_file=$(echo ${input_file/input/output})
  expected_min_cut=$(cat ${output_file})

  result=$(./${program} $input_file | python process.py $nodes $expected_min_cut $measure_full_contraction)
  echo $result
done
