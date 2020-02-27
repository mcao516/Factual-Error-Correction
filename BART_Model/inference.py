import torch
from fairseq.models.bart import BARTModel

MODEL_PATH='~/scratch/BART_models/checkpoints_iter'
DATA_PATH='~/scratch/summarization/cnn_dm/iterative_files/cnn_dm-bin/'


bart = BARTModel.from_pretrained(
    MODEL_PATH,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path=DATA_PATH
)
print('- model loaded.')

bart.cuda()
bart.eval()
bart.half()
bart.model = nn.DataParallel(bart.model)
count = 1
bsz = 16

print('- batch size: {}'.format(bsz))

TEST_PATH='~/scratch/summarization/cnn_dm/iterative_files/test.source'
OUTPUT_FILE='iter_preds.hypo'

with open(source_file) as source, open(OUTPUT_FILE, 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count % bsz == 0:
            with torch.no_grad():
                hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
            slines = []

        slines.append(sline.strip())
        count += 1

    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()