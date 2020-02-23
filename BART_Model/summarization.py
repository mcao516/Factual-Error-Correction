import torch
from fairseq.models.bart import BARTModel

bart = BARTModel.from_pretrained(
    'bart.large.cnn/',
    checkpoint_file='model.pt'
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 16

source_file = '/home/cadenc9020/Two-Steps-Summarization/datasets/cnn_dm/fairseq_files/train.source'
# target_file = '/home/cadenc9020/Two-Steps-Summarization/datasets/cnn_dm/fairseq_files/test.target'

with open(source_file) as source, open('train_preds.hypo', 'w') as fout:
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