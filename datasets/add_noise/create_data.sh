#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=32G                        # Ask for 32 GB of RAM
#SBATCH --time=3:00:00                   # The job will run for 3 hours
#SBATCH -o /scratch/<user>/slurm-%j.out  # Write the log in $SCRATCH
FILE_TYPE=sample

# 1. Create your environement locally
module load miniconda3
source activate py37

# 2. Copy your dataset on the compute node
# IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
cp $SCRATCH/summarization/cnn_dm/fairseq_files/$FILE_TYPE.source $SLURM_TMPDIR
cp $SCRATCH/summarization/cnn_dm/fairseq_files/$FILE_TYPE.target $SLURM_TMPDIR

# 3. Eventually unzip your dataset
# unzip $SLURM_TMPDIR/<dataset.zip> -d $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
python create_data.py --source_file $SLURM_TMPDIR/$FILE_TYPE.source --target_file $SLURM_TMPDIR/$FILE_TYPE.target --augmentations entity_swap pronoun_swap date_swap number_swap negation --save_intermediate

# 5. Copy whatever you want to save on $SCRATCH
cp $SLURM_TMPDIR/*.jsonl $SCRATCH/summarization/cnn_dm/corrupted_files
