#!/bin/bash
#SBATCH --job-name=hw2         # create a short name for your job
#SBATCH --partition=gpu          # gpu partition
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gpus-per-task=v100:1
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --account=irahbari_1147


# which gpu node was used
echo "Running on host" $(hostname)

# print the slurm environment variables sorted by name
printenv | grep -i slurm | sort

cd /scratch1/zhangray/TAC450-DataScience-Fall2026
source .venv/bin/activate
cd homework2

echo "Using 1 worker:"
kernprof -o ${SLURM_JOBID}-single.lprof -l mnist_classify.py --epochs=5 --num-workers=1

echo "Using 8 workers:"
kernprof -o ${SLURM_JOBID}-multi.lprof -l mnist_classify.py --epochs=5 --num-workers=8
