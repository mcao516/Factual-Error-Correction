#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import argparse

from os.path import join
from datetime import datetime
from utils import get_logger, set_seed
from model import Model, MODEL_CLASSES, ALL_MODELS
from data_utils import load_and_cache_examples, processors
from torch.utils.data import DataLoader, SequentialSampler
from extract_sentences import make_data


# These are the file paths where the validation/test set will be mounted (read only)
# into your Docker container.
METADATA_FILEPATH = ''
ARTICLES_FILEPATH = ''

# This is the filepath where the predictions should be written to.
OUTPUT_DIR = './'


def write_results(output_dir, guids, preds):
    output_eval_file = os.path.join(output_dir, "predictions.txt")
    with open(output_eval_file, "w") as writer:
        for i, p in zip(guids, preds):
            writer.write("%d,%d\n" % (int(i), p))


def main():
    # directory for training outputs
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())

    # required parameters
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name: " + ", ".join(ALL_MODELS))
    parser.add_argument("--checkpoint", default='', type=str, required=True,
                        help="where to load pre-trained model.")
    parser.add_argument("--max_seq_length", default=512, type=int, required=True,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--use_pretrained", action='store_true', default=True,
                        help="If use pre-trained model weights.")

    # other parameters
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--num_epochs", default=30, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--task_name", default='lpc', type=str,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=3, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--no_cuda", default=False, type=bool,
                        help="Do not use cuda.")
    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="Do lower case.")
    parser.add_argument("--seed", default=610, type=int,
                        help="Random seed.")
    parser.add_argument("--num_labels", default=3, type=int,
                        help="Classification label number.")
    parser.add_argument("--scheduler", default='warmup', type=str,
                        help="Which type of scheduler to use.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--overwrite_cache', action='store_true', default=False,
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--write_summary', default=True, type=bool,
                        help="If write summary into tensorboard.")
    parser.add_argument('--fp16', action='store_true', default=False,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    # data directory
    parser.add_argument("--data_dir", default='../data/TF-IDF', type=str,
                        help="data directory where pickle dataset is stored.")
    parser.add_argument("--output_dir", default=output_dir, type=str,
                        help="output directory for model, log file and summary.")
    parser.add_argument("--log_path", default=join(output_dir, "log.txt"), type=str,
                        help="Path to log.txt.")
    parser.add_argument("--summary_path", default=join(output_dir, "summary"), type=str,
                        help="Path to summary file.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    args.logger = get_logger(args.log_path)

    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.logger.info("- device: {}, n_gpu: {}".format(args.device, args.n_gpu))
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # set seed
    set_seed(args.seed)

    # build model
    args.logger.info("Build model...")
    model = Model(args)

    # make data
    make_data(ARTICLES_FILEPATH, METADATA_FILEPATH, args.data_dir, file_name='dev')

    # build dataset
    args.logger.info("Loading dataset...")
    eval_dataset, guids = load_and_cache_examples(args, args.task_name, model.tokenizer, evaluate=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # training
    args.logger.info("Start testing:")
    preds = model.test(eval_dataloader)
    assert len(preds) == len(guids), "Prediction list and GUID list length do NOT equal!!!"

    # write results
    args.logger.info("Write prediction results:")
    write_results(OUTPUT_DIR, guids, preds)
    args.logger.info("Save results at: {}".format(os.path.join(OUTPUT_DIR, 'predictions.txt')))


if __name__ == '__main__':
    main()
