#!/bin/bash

set -euo pipefail

function random_string() {
    cat /dev/urandom | tr -cd 'a-f0-9' | head -c 32
}

function transfer_csv_to() {
    mv $(find timing/$1 -name *.csv -type f) "$2/$(random_string).csv"
}

function setup_single() {
    local runner_dir="$(mktemp -d)"

    ln -s "$(realpath -L ../../dataset)" "$runner_dir/dataset"
    ln -s "$(realpath -L ../$1)" "$runner_dir/$1"

    echo "$runner_dir"
}

function run_pipelineComplete() {
    local working_dir="$(setup_single TimePipComplete)"
    cd "$working_dir"

    mkdir -p prog
    cd prog

    mkdir -p timing/pipeline_complete

    taskset -c $1 ./../TimePipComplete $2 $3

    transfer_csv_to pipeline_complete /tmp/pipeline_complete

    cd /
    rm -rf "$working_dir"
}

function run_pipelineMatchingSteps() {
    local working_dir="$(setup_single TimeMatchingSteps)"
    cd "$working_dir"

    mkdir -p prog
    cd prog

    mkdir -p timing/pipeline_steps/postalignment
    mkdir -p timing/pipeline_steps/distance

    taskset -c $1 ./../TimeMatchingSteps $2 $3

    transfer_csv_to pipeline_steps/postalignment /tmp/pipeline_steps/postalignment
    transfer_csv_to pipeline_steps/distance /tmp/pipeline_steps/distance

    cd /
    rm -rf "$working_dir"
}

function run_pipelineSteps() {
    local working_dir="$(setup_single TimeSteps)"
    cd "$working_dir"

    mkdir -p prog
    cd prog

    mkdir -p timing/pipeline_steps/edge_mask
    mkdir -p timing/pipeline_steps/prealignment
    mkdir -p timing/pipeline_steps/maximum_curvature

    taskset -c $1 ./../TimeSteps $2 $3

    transfer_csv_to pipeline_steps/edge_mask /tmp/pipeline_steps/edge_mask
    transfer_csv_to pipeline_steps/prealignment /tmp/pipeline_steps/prealignment
    transfer_csv_to pipeline_steps/maximum_curvature /tmp/pipeline_steps/maximum_curvature

    cd /
    rm -rf "$working_dir"
}

# Start complete pipeline timing

mkdir -p /tmp/pipeline_complete
rm -f /tmp/pipeline_complete/*

( run_pipelineComplete 0 0 266 ) &
( run_pipelineComplete 1 266 532 ) &
( run_pipelineComplete 2 532 798 ) &

wait

cat /tmp/pipeline_complete/*.csv | sort -h | sed 's/, /,/g; s/,$//g' > "pipeline_complete/$(date -u +'%Y-%m-%dT%H:%M:%SZ')completePip.csv"

rm -rf /tmp/pipeline_complete

# Start matching steps timing

mkdir -p /tmp/pipeline_steps/postalignment
mkdir -p /tmp/pipeline_steps/distance
rm -f /tmp/pipeline_steps/postalignment/*
rm -f /tmp/pipeline_steps/distance/*

( run_pipelineMatchingSteps 0 0 266 ) &
( run_pipelineMatchingSteps 1 266 532 ) &
( run_pipelineMatchingSteps 2 532 798 ) &

wait

cat /tmp/pipeline_steps/postalignment/*.csv | sort -h | sed 's/, /,/g; s/,$//g' > "pipeline_steps/postalignment/$(date -u +'%Y-%m-%dT%H:%M:%SZ')postalignment.csv"
cat /tmp/pipeline_steps/distance/*.csv | sort -h | sed 's/, /,/g; s/,$//g' > "pipeline_steps/distance/$(date -u +'%Y-%m-%dT%H:%M:%SZ')distance.csv"

rm -rf /tmp/pipeline_steps

# Start pipeline steps timing

mkdir -p /tmp/pipeline_steps/edge_mask
mkdir -p /tmp/pipeline_steps/prealignment
mkdir -p /tmp/pipeline_steps/maximum_curvature
rm -f /tmp/pipeline_steps/edge_mask/*
rm -f /tmp/pipeline_steps/prealignment/*
rm -f /tmp/pipeline_steps/maximum_curvature/*

( run_pipelineSteps 0 0 266 ) &
( run_pipelineSteps 1 266 532 ) &
( run_pipelineSteps 2 532 798 ) &

wait

cat /tmp/pipeline_steps/edge_mask/*.csv | sort -h | sed 's/, /,/g; s/,$//g' > "pipeline_steps/edge_mask/$(date -u +'%Y-%m-%dT%H:%M:%SZ')edge_mask.csv"
cat /tmp/pipeline_steps/prealignment/*.csv | sort -h | sed 's/, /,/g; s/,$//g' > "pipeline_steps/prealignment/$(date -u +'%Y-%m-%dT%H:%M:%SZ')prealignment.csv"
cat /tmp/pipeline_steps/maximum_curvature/*.csv | sort -h | sed 's/, /,/g; s/,$//g' > "pipeline_steps/maximum_curvature/$(date -u +'%Y-%m-%dT%H:%M:%SZ')maxCurv.csv"

rm -rf /tmp/pipeline_steps