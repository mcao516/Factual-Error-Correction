import torch
import torch.nn as nn
import argparse

from tqdm import tqdm
from fairseq.models.bart import BARTModel


def main(args):
    print("- checkpoint path: {}".format(args.checkpoint_path))
    print("- checkpoint type: {}".format(args.checkpoint_type))
    print("- data name or path: {}".format(args.data_path))

    bart = BARTModel.from_pretrained(
        args.checkpoint_path,
        checkpoint_file=args.checkpoint_type,
        data_name_or_path=args.data_path
    )
    print('- model loaded.')

    bart.cuda()
    bart.eval()
    bart.half()

    print('- read test data from: {}'.format(args.test_path))
    data = []
    with open(args.test_path) as source:
        for line in source:
            data.append(line.strip())
    print('- total data: {}'.format(len(data)))
    print('- sample: \n{}\n'.format(data[0]))

    count = 1
    print('- start inference (batch size = {}):'.format(args.batch_size))
    with open(args.output_file, 'w') as fout:
        slines = [data[0]]
        for sline in tqdm(data[1:]):
            if count % args.batch_size == 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines,
                                                   beam=args.beam_size, lenpen=args.lenpen, 
                                                   max_len_b=args.max_len, min_len=args.min_len,
                                                   no_repeat_ngram_size=3)
                for hypothesis in hypotheses_batch:
                    if type(hypothesis) == type([]) and args.write_all:
                        for h in hypothesis:
                            fout.write(h + '\n')
                            fout.flush()
                    elif type(hypothesis) == type([]):
                        fout.write(hypothesis[0] + '\n')
                        fout.flush()
                    else:
                        fout.write(hypothesis + '\n')
                        fout.flush()
                slines = []

            slines.append(sline)
            count += 1

        if slines != []:
            hypotheses_batch = bart.sample(slines,
                                           beam=args.beam_size, lenpen=args.lenpen,
                                           max_len_b=args.max_len, min_len=args.min_len,
                                           no_repeat_ngram_size=3)
            for hypothesis in hypotheses_batch:
                if type(hypothesis) == type([]) and args.write_all:
                    for h in hypothesis:
                        fout.write(h + '\n')
                        fout.flush()
                elif type(hypothesis) == type([]):
                    fout.write(hypothesis[0] + '\n')
                    fout.flush()
                else:
                    fout.write(hypothesis + '\n')
                    fout.flush()

# python inference.py --checkpoint_path ~/scratch/BART_models/checkpoints_iter --checkpoint_type checkpoint1.pt --data_path ~/scratch/summarization/cnn_dm/iterative_files/cnn_dm-bin/ --test_path ~/scratch/summarization/cnn_dm/iterative_files/test.source --output_file preds/iter_preds_cp1.hypo --batch_size 64

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--checkpoint_path", type=str, help="checkpoint directory.")
    PARSER.add_argument("--checkpoint_type", type=str, default='checkpoint_best.pt', help="checkpoint type to use")
    PARSER.add_argument("--data_path", type=str, default=None, help="cnn_dm-bin in training dataset.")
    PARSER.add_argument("--test_path", type=str, help="test source data")
    PARSER.add_argument("--output_file", type=str, default='test.hypo', help="output file.")
    PARSER.add_argument("--batch_size", type=int, default=32, help="batch size.")
    PARSER.add_argument("--max_len", type=int, default=120, help="max summary length.")
    PARSER.add_argument("--min_len", type=int, default=30, help="min summary length.")
    PARSER.add_argument("--beam_size", type=int, default=1, help="beam search size.")
    PARSER.add_argument("--lenpen", type=float, default=2.0, help="length penality.")
    PARSER.add_argument("--write_all", action='store_true', help="write all beam hypothesis.")
    ARGS = PARSER.parse_args()
    main(ARGS)
