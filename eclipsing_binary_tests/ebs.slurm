#!/usr/bin/bash

#SBATCH --job-name=ebs
#SBATCH --mail-type=END,FAIL # (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zclaytor@ufl.edu
#SBATCH --time=1-00:00:00
#SBATCH --mem=6G
#SBATCH --output=%x-%A.out
#SBATCH --error=%x-%A.err


module load conda
conda activate kbonus
python -u measure_transits.py
