#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=3:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/<user>/slurm-%j.out  # Write the log in $SCRATCH

# 1. Create your environement locally
module load miniconda3
source activate py37

python inference.py --checkpoint_path /home/mcao610/scratch/BART_models/checkpoints_xsum_iter --checkpoint_type checkpoint_last.pt --data_path /home/mcao610/scratch/summarization/XSum/iterative_files/xsum-bin --test_path /home/mcao610/scratch/summarization/XSum/iterative_files/test.source --output_file preds/xsum_iter_bm1_cplast.hypo --batch_size 96 --beam_size 1 --max_len 60 --min_len 10 --lenpen 1.0
