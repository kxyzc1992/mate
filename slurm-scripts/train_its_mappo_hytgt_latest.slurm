#!/bin/bash
#SBATCH -J mappo_its_hytgt_test                   # 作业名为 test
#SBATCH -o /scratch/kongxiangyu/Experiments/mate/its_hytgt_test.%j.%N.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH -e /scratch/kongxiangyu/Experiments/mate/its_hytgt_test.%j.%N.err
#SBATCH -p DGX                    # 作业提交的分区为 HGX
#SBATCH --account=research
#SBATCH --qos=lv0b               # 作业使用的 QoS 为 lv0b
#SBATCH -N 1                      # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1       # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=64         # 单任务使用的 CPU 核心数为 4
#SBATCH -t 1-00:00:00                # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:1              # 单个节点使用 1 块 GPU 卡
#SBATCH --mem=10G                  #--mem=<size[units]>：设定每个节点的内存大小，后缀可以为[K|M|G|T]，默认为MB

module purge
module add cuda/9.0
module add slurm/BigAI/23.02.2

source /scratch/kongxiangyu/anaconda3/bin/activate
conda init bash
conda activate mate-env

srun python -m examples.its.mappo_hytgt.camera.train \
	--project mate-camera \
	--num-workers 16 --num-envs-per-worker 8 --num-gpus 0.3 \
	--timesteps-total 10E6 --seed 0 1 2
