#!/bin/bash
#SBATCH --job-name="mappo_hytgt_test"
#SBATCH --partition=gpu
#SBATCH --account=research
#SBATCH --qos=level0
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH -o "/scratch/kongxiangyu/Experiments/mate/hytgt_test.%j.%N.out"
#SBATCH -e "/scratch/kongxiangyu/Experiments/mate/hytgt_test.%j.%N.err"

module purge
module load anaconda3/2021.11
module load slurm

source /cm/shared/apps/anaconda3/bin/activate
conda init bash
conda activate mate-env

srun python -m examples.mappo_hytgt.camera.train \
	--project mate-camera \
	--num-workers 16 --num-envs-per-worker 8 --num-gpus 0.3 \
	--timesteps-total 10E6 --seed 0 1 2 
