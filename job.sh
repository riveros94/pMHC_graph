#!/bin/bash
#SBATCH --job-name=CrossReactivity          # Nome do trabalho
#SBATCH --nodes=1                           # Solicita 1 nó
#SBATCH --cpus-per-task=8                   # Solicita 8 CPUs para a tarefa
#SBATCH -p batch-AMD
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate ic_bioinfo

export OMP_NUM_THREADS=1                    # Define o número de threads OpenMP

echo "CPUs disponíveis: $SLURM_CPUS_ON_NODE"
echo "Threads OpenMP: $OMP_NUM_THREADS"

bash run_identify_crossreactive_subgraphs.sh
