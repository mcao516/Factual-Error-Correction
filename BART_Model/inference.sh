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

python inference.py --checkpoint_path ~/scratch/BART_models/bart.large.cnn --checkpoint_type model.pt --data_path ~/scratch/BART_models/bart.large.cnn --test_path /home/mcao610/scratch/summarization/cnn_dm/fairseq_files/train_rest.source --output_file preds/cnndm_train_rest_bm4_all.hypo --batch_size 24 --beam_size 4 --max_len 140 --min_len 55 --lenpen 2.0 --write_all
