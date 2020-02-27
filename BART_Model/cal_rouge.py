import rouge
import nltk

# nltk.download('punkt')

pred_file_path = 'iter_preds.hypo'
tgt_file_path = '/home/mcao610/scratch/summarization/cnn_dm/iterative_files/test.target'

with open(pred_file_path) as pf, open(tgt_file_path) as tf:
    preds, tgts = [], []
    for p, t in zip(pf, tf):
        preds += [p.strip()]
        tgts += [t.strip()]

print(len(preds))
print(len(tgts))

print("Make sure they are matched:")
print(preds[len(preds) // 2])
print(tgts[len(tgts) // 2])

evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                        max_n=4,
                        limit_length=True,
                        length_limit=150,
                        length_limit_type='words',
                        apply_avg=True,
                        alpha=0.5, # Default F1_score
                        weight_factor=1.2,
                        stemming=True)

scores = evaluator.get_scores(preds, tgts)

def prepare_results(m, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

print()
print("Evaluation:")
for metric, results in sorted(scores.items(), key=lambda x: x[0]):
    print(prepare_results(metric, results['p'], results['r'], results['f']))
