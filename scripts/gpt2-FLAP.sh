#!/bin/bash

#SBATCH -J gpt2-FLAP	                # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani   # Which partition will run your job
#SBATCH --time=0-00:10            # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:1              # (optional) Requesting type and number of GPUs
#SBATCH --mem=64G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=/mnt/lustre/work/eickhoff/esx670/out/gpt2-FLAP_%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/lustre/work/eickhoff/esx670/err/gpt2-FLAP_%j.err        # File to which STDERR will be written - make sure this is not on $HOME
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


for lowest_sparsity in 60
do
    for cliff_point in biggest first fixed
    do
    echo "Running with lowest_sparsity=$lowest_sparsity and cliff_point=$cl"

    srun python3  /home/eickhoff/esx670/AcceleratedPathPatching/Pruning/FLAP/lib/FLAP.py \
        --task=ioi \
        --nsamples=200 \
        --batch_size=25 \
        --model_name=gpt2 \
        --device=cuda \
        --cliff_point=$cliff_point \
        --lowest_sparsity=$lowest_sparsity \
        --step_size=1 \
        --seed=239501 \
        --save_img \
        --save_txt \
        --out_path=results/ \
        --cache_dir=llm_weights 
    done
done