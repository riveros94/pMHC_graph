#!/bin/bash
#SBATCH --job-name=CrossReactivity          # Nome do trabalho
#SBATCH --nodes=1                           # Solicita 1 nó
#SBATCH --cpus-per-task=1                   # Solicita 8 CPUs para a tarefa
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1  # Solicita 1 GPU
#SBATCH --mem=50G      # Ajuste conforme necessário
#SBATCH --time=02:00:00
#SBATCH --output=meu_job.out
#SBATCH --error=meu_job.err

source ~/.bashrc
conda activate ic_imuno

export OMP_NUM_THREADS=1             # Define o número de threads OpenMP

echo "CPUs disponíveis: $SLURM_CPUS_ON_NODE"
echo "Threads OpenMP: $OMP_NUM_THREADS"

bash run_identify_crossreactive_subgraphs.sh
