#!/bin/bash
#SBATCH --gres=gpu:a100:1
# sbatch -p part_80gb run.sh
apptainer exec --nv ./env/sam3d.sif python demo.py input/strawberry_pack --visualize