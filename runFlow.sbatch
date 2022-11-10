#!/bin/bash
#SBATCH -p long
#SBATCH -t 15:00:00
#SBATCH -A astro 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4

#SBATCH --mail-user=rugved.pund@stonybrook.edu
#SBATCH --mail-type=ALL 


export PYTHONPATH=~/LuSEE/repos/SINF/:~/LuSEE/repos/luseepy/:/astro/u/rugvedpund/LuSEE/repos/normalizingflows
srun python runFlow.py 8.0 20 --galcut 10 
srun python runFlow.py 4.0 20 --galcut 10 
srun python runFlow.py 2.0 20 --galcut 10 

