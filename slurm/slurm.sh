#!/bin/bash
#SBATCH -c 2                               # Request one core
#SBATCH -t 0-5:00                         # Runtime in D-HH:MM format
#SBATCH -p gpu_quad                           # Partition to run in
#SBATCH --gres=gpu:1                           # Partition to run in
#SBATCH --mem=48G                         # Memory total in MiB (for all cores)
#SBATCH -o %j.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e %j.err                 # File to which STDERR will be written, including job ID (%j)
                                           # You can change the filenames given with -o and -e to any filenames you'd like

# You can change hostname to any command you would like to run
cd /home/jex451/XrayGPT
module load miniconda3/23.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/n/app/miniconda3/23.1.0/lib
source activate xraygpt
/home/jex451/.conda/envs/xraygpt/bin/python3 inference_batch.py --cfg-path eval_configs/xraygpt_eval.yaml  --gpu-id 0
