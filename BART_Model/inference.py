import torch
import torch.nn as nn
import argparse
from fairseq.models.bart import BARTModel


def main(args):
    print(args.checkpoint_path)
    print(args.data_path)
    bart = BARTModel.from_pretrained(
        args.checkpoint_path,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=args.data_path
    )
    print('- model loaded.')

    bart.cuda()
    bart.eval()
    bart.half()
    count = 1
    bsz = 32

    print('- batch size: {}'.format(bsz))

    with open(args.test_path) as source, open(args.output_file, 'w') as fout:
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
            if count % 100 == 0:
                print(count)

        if slines != []:
            hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()

    # DATA_PATH='~/scratch/summarization/cnn_dm/iterative_files/cnn_dm-bin/'
    # MODEL_PATH='~/scratch/BART_models/checkpoints_iter'
    # TEST_PATH='~/scratch/summarization/cnn_dm/iterative_files/test.source'


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--checkpoint_path", type=str, help="checkpoint directory.")
    PARSER.add_argument("--data_path", type=str, help="train, val or test.")
    PARSER.add_argument("--test_path", type=str, help="train, val or test.")
    PARSER.add_argument("--output_file", type=str, default='test.hypo', help="train, val or test.")
    ARGS = PARSER.parse_args()
    main(ARGS)
