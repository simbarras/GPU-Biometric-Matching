#!/bin/bash

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

mkdir -p /tmp/pipeline_complete
rm -f /tmp/pipeline_complete/*

( run_pipelineComplete 0 0 266 ) &
( run_pipelineComplete 1 266 532 ) &
( run_pipelineComplete 2 532 798 ) &

wait

cat /tmp/pipeline_complete/*.csv | sort -h | sed 's/, /,/g; s/,$//g' > "pipeline_complete/$(date -u +'%Y-%m-%dT%H:%M:%SZ')completePip.csv"

rm -rf /tmp/pipeline_complete
