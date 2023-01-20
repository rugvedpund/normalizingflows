#!/bin/bash
#SBATCH -p long
#SBATCH -t 5:00:00
#SBATCH -A astro 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4

#SBATCH --mail-user=rugved.pund@stonybrook.edu
#SBATCH --mail-type=ALL 


export PYTHONPATH=~/LuSEE/repos/SINF/:~/LuSEE/repos/luseepy/:/astro/u/rugvedpund/LuSEE/repos/normalizingflows

srun python runFlow.py 2.0 10 --galcut 20 --noise 0.01 --noiseSeed 0

