#! /bin/bash

#SBATCH --job-name=hyp_tuning
#SBATCH --array=0-99
#SBATCH --output=logs/slurm_hyp_tuning_%A_%a.out
#SBATCH --error=logs/slurm_hyp_tuning_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=16G
#SBATCH --mail-user=joelne@umich.edu

source ~/virtual_envs/bayesflow/bin/activate
python hyp_tuning.py
echo "SLURM job ${SLURM_ARRAY_TASK_ID} done"
