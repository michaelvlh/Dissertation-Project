#!/bin/bash
#$ -l h_rt=8:00:00  #time needed
#$ -pe openmp 4 #number of cores
#$ -l rmem=5G #number of memory
#$ -N melSpec
#$ -o ../Output/melSpecLogs.txt  #This is where your output are logged.
#$ -e ../Output/melSpecErrors.txt  #This is where your errors are logged.
#$ -M lhvong1@sheffield.ac.uk
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/python/conda
module load libs/libsndfile/1.0.28/gcc-4.9.4
source activate newpytorch

export OMP_NUM_THREADS=$NSLOTS

python ../Code/runMelSpecTest.py