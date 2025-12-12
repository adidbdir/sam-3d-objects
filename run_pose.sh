#!/bin/bash
#SBATCH --gres=gpu:a100:1
# sbatch -p part_80gb run.sh
apptainer exec --nv ./env/sam.sif python export_voxel_pose.py input/rgb \
  --output-root outputs/mesh/rgb