"""
Script for generating synthetic data for FactCC training.

Script expects source documents in `jsonl` format with each source document
embedded in a separate json object.

Json objects are required to contain `id` and `text` keys.
"""

import argparse
import json
import os
import pickle

from tqdm import tqdm

import augmentation_ops as ops


def read_fairseq_files(source_path, target_path):
    """Read fairseq documents (source) and summaries (target).
    """
    data = []
    with open(source_path, 'r', encoding='utf-8') as source, \
        open(target_path, 'r', encoding='utf-8') as target:
        for i, (s, t) in enumerate(zip(source, target)):
            s, t = s.strip(), t.strip()
            data.append({
                'id': i, 'text': s, 'summary': t
            })
    return data


def load_source_docs(file_path, to_dict=False):
    with open(file_path, 'r', encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if to_dict:
        data = {example["id"]: example for example in data}
    return data


def load_pickle(file_path):
    return pickle.load(open(file_path, 'rb'))


def save_data(args, data, name_suffix):
    output_file = os.path.splitext(args.source_file)[0] + "-" + name_suffix +  ".jsonl"

    with open(output_file, "w", encoding="utf-8") as fd:
        for example in data:
            example = dict(example)
            example["text"] = example["text"].text
            example["summary"] = example["summary"].text
            example["claim"] = example["claim"].text
            fd.write(json.dumps(example, ensure_ascii=False) + "\n")


def apply_transformation(data, operation):
    new_data = []
    for example in tqdm(data):
        try:
            new_example = operation.transform(example)
            if new_example:
                new_data.append(new_example)
        except Exception as e:
            print("Caught exception:", e)
    return new_data


def main(args):
    # load data
    # source_docs = load_source_docs(args.data_file, to_dict=False)
    # print("Loaded %d source documents." % len(source_docs))

    # # create or load positive examples
    # print("Creating data examples")
    # sclaims_op = ops.SampleSentences()
    # data = apply_transformation(source_docs, sclaims_op)
    # {'id', 'text', 'claim', 'label': 'CORRECT', 'extraction_span': None,
    #  'backtranslation': False, 'augmentation': None, 'augmentation_span': None, 'noise': False}
    # print("Created %s example pairs." % len(data))

    # print("Augmentations: {}".format(args.augmentations))
    # # load data
    # source_docs = load_source_docs(args.data_file, to_dict=False)
    # print("Loaded %d source documents & summaries." % len(source_docs))

    # load data
    source_docs = read_fairseq_files(args.source_file, args.target_file)
    print("Loaded %d source documents & summaries." % len(source_docs))

    # create or load positive examples
    print("Creating data examples")
    format_op = ops.FormatTransformation()
    data = apply_transformation(source_docs, format_op)
    print("Created %s example pairs." % len(data))

    # if args.save_intermediate:
    #     save_data(args, data, "clean")

    # backtranslate
    data_btrans = []
    if not args.augmentations or "backtranslation" in args.augmentations:
        print("Creating backtranslation examples")
        btrans_op = ops.Backtranslation()
        data_btrans = apply_transformation(data, btrans_op)
        print("Backtranslated %s example pairs." % len(data_btrans))

        if args.save_intermediate:
            save_data(args, data_btrans, "btrans")

    # save original & translation data
    # data_positive = data + data_btrans
    # if args.save_ensemble:
    #     print("- Positive %s example pairs." % len(data_positive))
    #     save_data(args, data_positive, "positive")

    data_positive = data_btrans

    # create negative examples
    data_pronoun = []
    if not args.augmentations or "pronoun_swap" in args.augmentations:
        print("Creating pronoun examples")
        pronoun_op = ops.PronounSwap()
        data_pronoun = apply_transformation(data_positive, pronoun_op)
        print("PronounSwap %s example pairs." % len(data_pronoun))

        if args.save_intermediate:
            save_data(args, data_pronoun, "pronoun")

    data_dateswp = []
    if not args.augmentations or "date_swap" in args.augmentations:
        print("Creating date swap examples")
        dateswap_op = ops.DateSwap()
        data_dateswp = apply_transformation(data_positive, dateswap_op)
        print("DateSwap %s example pairs." % len(data_dateswp))

        if args.save_intermediate:
            save_data(args, data_dateswp, "dateswp")

    data_numswp = []
    if not args.augmentations or "number_swap" in args.augmentations:
        print("Creating number swap examples")
        numswap_op = ops.NumberSwap()
        data_numswp = apply_transformation(data_positive, numswap_op)
        print("NumberSwap %s example pairs." % len(data_numswp))

        if args.save_intermediate:
            save_data(args, data_numswp, "numswp")

    data_entswp = []
    if not args.augmentations or "entity_swap" in args.augmentations:
        print("Creating entity swap examples")
        entswap_op = ops.EntitySwap()
        data_entswp = apply_transformation(data_positive, entswap_op)
        print("EntitySwap %s example pairs." % len(data_entswp))

        if args.save_intermediate:
            save_data(args, data_entswp, "entswp")

    data_negation = []
    if not args.augmentations or "negation" in args.augmentations:
        print("Creating negation examples")
        negation_op = ops.NegateSentences()
        data_negation = apply_transformation(data_positive, negation_op)
        print("Negation %s example pairs." % len(data_negation))

        if args.save_intermediate:
            save_data(args, data_negation, "negation")

    # add noise to all
    if args.save_ensemble:
        data_negative = data_pronoun + data_dateswp + data_numswp + data_entswp + data_negation
        print("- Negative %s example pairs." % len(data_negative))
        save_data(args, data_negative, "negative")

    # add light noise
    data_pos_low_noise, data_neg_low_noise = [], []
    if not args.augmentations or "noise" in args.augmentations:
        print("Adding light noise to data")
        low_noise_op = ops.AddNoise()

        data_pos_low_noise = apply_transformation(data_positive, low_noise_op)
        print("- PositiveNoisy %s example pairs." % len(data_pos_low_noise))
        save_data(args, data_pos_low_noise, "positive-noise")

        data_neg_low_noise = apply_transformation(data_negative, low_noise_op)
        print("- NegativeNoisy %s example pairs." % len(data_neg_low_noise))
        save_data(args, data_neg_low_noise, "negative-noise")


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    # PARSER.add_argument("data_file", type=str, help="Path to file containing source documents.")
    PARSER.add_argument("--source_file", type=str, help="Path to file contains source documents.")
    PARSER.add_argument("--target_file", type=str, help="Path to file contains target summaries.")
    PARSER.add_argument("--augmentations", type=str, nargs="+", default=(), help="List of data augmentation applied to data.")
    PARSER.add_argument("--all_augmentations", action="store_true", help="Flag whether all augmentation should be applied.")
    PARSER.add_argument("--save_intermediate", action="store_true", help="Flag whether intermediate data from each transformation should be saved in separate files.")
    PARSER.add_argument("--save_ensemble", action="store_true", help="Flag whether all negative data should be saved in a single file.")
    ARGS = PARSER.parse_args()
    main(ARGS)