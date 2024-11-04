#!/bin/bash
# Job name:
#PBS -N cv-proj
# Output and error files:
#PBS -j oe
#PBS -o cv-proj.log
# Queue name:
#PBS -q auto
# Resource requests:
#PBS -l select=1:ngpus=1
# Walltime (maximum run time):
#PBS -l walltime=8:30:00
# Project code:
#PBS -P H100011

# Load Singularity module
module load singularity

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

nvidia-smi

# Run the Singularity run script
./run_in_singularity.sh