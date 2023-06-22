#!/bin/bash
#SBATCH -p long
#SBATCH -t 24:00:00
#SBATCH -A astro 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4

#SBATCH --mail-user=rugved.pund@stonybrook.edu
#SBATCH --mail-type=ALL 

module load python/3.9-anaconda-2022-5
export PYTHONPATH=~/LuSEE/repos/SINF/:~/LuSEE/repos/luseepy/:/astro/u/rugvedpund/LuSEE/repos/normalizingflows

python runAll.py 2.0 --galcut 20 --noise 0.0  --noiseSeed 0 --noisyT21 --plot 'all' --DA_factor 1.0 --subsampleSigma 2.0  --gainFluctuationLevel 0.0 --gFdebug 0  --append "_alpha0.999,0.999" --retrain
#srun python runFlow.py 4.0 5 --galcut 20 --noise 0.0001 --noiseSeed 0
#srun python runFlow.py 4.0 5 --galcut 20 --noise 0.0001 --noiseSeed 0 --chromatic
#srun python runFlow.py 8.0 5 --galcut 20 --noise 0.0001 --noiseSeed 0
#srun python runFlow.py 8.0 5 --galcut 20 --noise 0.0001 --noiseSeed 0 --chromatic
#
