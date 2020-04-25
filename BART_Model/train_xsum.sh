#!/bin/bash
#SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=3:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/<user>/slurm-%j.out  # Write the log in $SCRATCH

# 1. Create your environement locally
module load miniconda3
source activate py37

DATA_PATH=/home/mcao610/scratch/summarization/XSum/fairseq_files/xsum-bin.zip
BART_PATH=/home/mcao610/scratch/BART_models/bart.large.tar.gz
SAVE_DIR=/home/mcao610/scratch/BART_models/checkpoints_xsum/

# 2. Copy your dataset on the compute node
cp $DATA_PATH $SLURM_TMPDIR
cp $BART_PATH $SLURM_TMPDIR

# 3. Eventually unzip your dataset
unzip $SLURM_TMPDIR/xsum-bin.zip -d $SLURM_TMPDIR
tar -xvzf $SLURM_TMPDIR/bart.large.tar.gz -C $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
TOTAL_NUM_UPDATES=15000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=1024
UPDATE_FREQ=4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ~/fairseq/train.py $SLURM_TMPDIR/xsum-bin \
    --restore-file $SLURM_TMPDIR/bart.large \
    --save-dir $SAVE_DIR \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;

# 5. Copy whatever you want to save on $SCRATCH
# cp $SLURM_TMPDIR/<to_save> $SCRATCH