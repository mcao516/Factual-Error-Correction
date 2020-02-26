INPUT_PATH=$SCRATCH/summarization/cnn_dm/fairseq_files
OUTPUT_PATH=$SCRATCH/summarization/cnn_dm/fairseq_files

for SPLIT in sample
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --inputs "$INPUT_PATH/$SPLIT.$LANG" \
    --outputs "$OUTPUT_PATH/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done
