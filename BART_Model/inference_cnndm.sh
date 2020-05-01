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

#python inference.py --checkpoint_path /home/mcao610/scratch/BART_models/bart.large.cnn \
#     --checkpoint_type model.pt \
#     --data_path /home/mcao610/scratch/BART_models/bart.large.cnn \
#     --test_path /home/mcao610/scratch/summarization/cnn_dm/fairseq_files/test.source \
#     --output_file preds/this_is_a_test.hypo \
#     --batch_size 96 \
#     --beam_size 1 \
#     --max_len 140 \
#     --min_len 55 \
#     --lenpen 2.0;

#python inference.py --checkpoint_path /home/mcao610/scratch/BART_models/checkpoints_uni \
#     --checkpoint_type checkpoint_best_1.pt \
#     --data_path /home/mcao610/scratch/summarization/cnn_dm/corrupted_nodup_files/cnn_dm-bin/ \
#     --test_path /home/mcao610/scratch/summarization/cnn_dm/corrupted_nodup_files/test.source \
#     --output_file preds/TEST_corrupted_nodup.hypo \
#     --batch_size 96 \
#     --beam_size 1 \
#     --max_len 140 \
#     --min_len 55 \
#     --lenpen 2.0;

python inference.py --checkpoint_path /home/mcao610/scratch/BART_models/checkpoints_cnndm_split1 \
    --checkpoint_type checkpoint_best.pt \
    --data_path /home/mcao610/scratch/summarization/cnn_dm/fairseq_files_1/cnn_dm-bin/ \
    --test_path /home/mcao610/scratch/summarization/cnn_dm/fairseq_files/test.source \
    --output_file preds/INFERENCE_TEST.hypo \
    --batch_size 96 \
    --beam_size 1 \
    --max_len 140 \
    --min_len 55 \
    --lenpen 2.0;
