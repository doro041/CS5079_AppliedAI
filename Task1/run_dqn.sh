#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu 

#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=u14jp20@abdn.ac.uk 

module load miniconda3
source activate task1_venv

srun python new_dqn.py --train --ep=1000000 --decay=0.9999975  --gamma=$1 --lr=$2