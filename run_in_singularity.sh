#!/bin/bash

# if [ -z "$1" ]; then
#   echo "Usage: $0 <script_to_run>"
#   exit 1
# fi

# Define the current directory
CURRENT_DIR=$(pwd)

# Define the paths to mount
# source $(dirname $(readlink -f $0))/env
PROJECT_ROOT=$(dirname $(readlink -f $0))

# Define the path to the Singularity image
SINGULARITY_IMAGE=/scratch/e1310988/transformers_latest.sif

# script to run
# SCRIPT_TO_RUN=adapt_starlink.sh
# SCRIPT_TO_RUN=$1

if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
  GPU_FLAG="--nv"
else
  GPU_FLAG=""
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run the Singularity container with mounted directories
singularity exec ${GPU_FLAG} --bind ${PROJECT_ROOT}:${PROJECT_ROOT} ${SINGULARITY_IMAGE} bash -c "
  python main.py
" > ${PROJECT_ROOT}/log/${PBS_JOBID}_${TIMESTAMP}.log 2>&1