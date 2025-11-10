#!/bin/bash

#SBATCH -J gpt2-PP	                # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani   # Which partition will run your job
#SBATCH --time=0-00:10            # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:1              # (optional) Requesting type and number of GPUs
#SBATCH --mem=64G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=/mnt/lustre/work/eickhoff/esx670/out/gpt2-PP_%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/lustre/work/eickhoff/esx670/err/gpt2-PP_%j.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=frauke.andersen@student.uni-tuebingen.de #Email to which notifications will be sent
# Diagnostic and Analysis Phase - please leave these in.

scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus

# Setup Phase
# add possibly other setup code here, e.g.
# - copy singularity images or datasets to local on-compute-node storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environment variables
# - determine commandline arguments for `srun` calls
eval "$(conda shell.bash hook)"
conda activate /mnt/lustre/work/eickhoff/esx670/.conda/py-311
export LD_LIBRARY_PATH=/mnt/lustre/work/eickhoff/esx670/.conda/py-311-pytorch/lib/
export PYTHONPATH=$PYTHONPATH:$HOME/AcceleratedPathPatching/


for importance_threshold in 2.5 2 1.5 1
do
    for min_value_threshold in 0.02 0.01 0.002 0.001
    do
    echo "Running with importance_threshold=$importance_threshold"
    
    srun python3 /home/eickhoff/esx670/AcceleratedPathPatching/Patching/AutomatedPathPatching.py \
        --model_name=gpt2 \
        --task=IOI \
        --patching_method=path \
        --N=100 \
        --metric=logits_diff \
        --pruning_circuit=none \
        --importance_threshold=$importance_threshold \
        --min_value_threshold=$min_value_threshold \
        --device=cuda \
        --out_path=results \
        --cache_dir=llm_weights/ \
        --seed=12345432 \
        --save_every_x_steps=50 \
        --calc_FLOPS \
        --save_text
    done
done