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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel


def main():
    # directory for training outputs
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())

    # required parameters
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name: " + ", ".join(ALL_MODELS))

    # data directory
    parser.add_argument("--data_dir", default='', type=str,
                        help="data directory where pickle dataset is stored.")
    parser.add_argument("--output_dir", default=output_dir, type=str,
                        help="output directory for model, log file and summary.")
    parser.add_argument("--log_path", default=join(output_dir, "log.txt"), type=str,
                        help="Path to log.txt.")
    parser.add_argument("--summary_path", default=join(output_dir, "summary"), type=str,
                        help="Path to summary file.")
    parser.add_argument("--model_dir", default=join(output_dir, "model/"), type=str,
                        help="where to load pre-trained model.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--max_summary_length", default=200, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--overwrite_cache', action='store_true', default=False,
                        help="Overwrite the cached training and evaluation sets")


    args = parser.parse_args()
    args.logger = get_logger(args.log_path)

    # Setup CUDA, GPU & distributed training
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.logger.info("- device: {}, n_gpu: {}".format(args.device, args.n_gpu))

    # set seed
    set_seed(args.seed)

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    # build dataset
    args.logger.info("Loading dataset...")
    train_dataset, _ = load_and_cache_examples(args, tokenizer, evaluate=False)
    eval_dataset, _ = load_and_cache_examples(args, tokenizer, evaluate=True)


if __name__ == '__main__':
    main()
