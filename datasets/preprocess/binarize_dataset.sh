INPUT_PATH=$SCRATCH/summarization/cnn_dm/fairseq_files
OUTPUT_PATH=$SCRATCH/summarization/cnn_dm/fairseq_files

fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "$INPUT_PATH/sample.bpe" \
  --validpref "$INPUT_PATH/sample.bpe" \
  --destdir "$OUTPUT_PATH/cnn_dm-bin-sample/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
