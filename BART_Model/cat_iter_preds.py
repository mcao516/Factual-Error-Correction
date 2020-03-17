pred_path = 'preds/iter_preds_bm1_cpbest.hypo'
source_path = '/home/mcao610/scratch/summarization/cnn_dm/fairseq_files/test.source'
output_path = '/home/mcao610/scratch/summarization/cnn_dm/iterative_files/test2.source'

with open(pred_path, 'r', encoding='utf-8') as pf, \
    open(source_path, 'r', encoding='utf-8') as sf, \
    open(output_path, 'w', encoding='utf-8') as wf:
    for p, s in zip(pf, sf):
        p, s = p.strip(), s.strip()
        if len(p) > 0:
            wf.write(p + ' </s> ' + s + '\n')
