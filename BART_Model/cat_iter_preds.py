pred_path = 'preds/xsum_test_bm1.hypo'
source_path = '/home/mcao610/scratch/summarization/XSum/fairseq_files/test.source'
output_path = '/home/mcao610/scratch/summarization/XSum/iterative_files/test.source'

with open(pred_path, 'r', encoding='utf-8') as pf, \
    open(source_path, 'r', encoding='utf-8') as sf, \
    open(output_path, 'w', encoding='utf-8') as wf:
    for p, s in zip(pf, sf):
        p, s = p.strip(), s.strip()
        if len(p) > 0:
            wf.write(p + ' </s> ' + s + '\n')
