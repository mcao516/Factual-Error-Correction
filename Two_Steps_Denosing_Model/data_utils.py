#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Build dataset for model training and testing.

   Author: Meng Cao
"""
import os
import json
import copy
import torch
import pickle

from torch.utils.data import TensorDataset

BOS_TOKEN, EOS_TOKEN = '[BOS]', '[EOS]'


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class Processor(object):
    """Processor for the Leader-prize competition data set"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.pkl"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "val.pkl"), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "test.pkl"), "test")

    def _create_examples(self, path, set_type):
        """Creates examples for the training and dev sets."""
        with open(path, 'rb') as handle:
            metadata = pickle.load(handle)

        examples = []
        for i, m in enumerate(metadata):
            guid = set_type + '-' + str(i)
            # text_a: story, text_b: noisy_summary, label: summary
            text_a, text_b, label = m[0], m[2], m[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def load_and_cache_examples(args, tokenizer, evaluate=False):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    processor = Processor()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        args.model_type,
        str(args.max_seq_length),
        str(args.max_summary_length)))
    cached_guids_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}_guids'.format(
        'dev' if evaluate else 'train',
        args.model_type,
        str(args.max_seq_length),
        str(args.max_summary_length)))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        args.logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        guids = torch.load(cached_guids_file)
    else:
        args.logger.info("Creating features from dataset file at %s", args.data_dir)

        examples = processor.get_dev_examples(args.data_dir) if evaluate \
            else processor.get_train_examples(args.data_dir)

        pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        story_features, guids = convert_inputs_to_features(examples, tokenizer,
                                                           summary=False,
                                                           max_length=args.max_story_length,
                                                           pad_on_left=False,
                                                           pad_token=pad_token_id,
                                                           pad_token_segment_id=0)

        summary_features, _ = convert_inputs_to_features(examples, tokenizer,
                                                         summary=True,
                                                         max_length=args.max_summary_length,
                                                         pad_on_left=False,
                                                         pad_token=pad_token_id,
                                                         pad_token_segment_id=0)

        src_ids, tgt_ids = convert_outputs_to_features(examples, tokenizer,
                                                       max_length=args.max_summary_length,
                                                       pad_on_left=False,
                                                       pad_token=pad_token_id,
                                                       pad_token_segment_id=0)
        features = [story_features, summary_features, src_ids, tgt_ids]

        if args.local_rank in [-1, 0]:
            args.logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
            torch.save(guids, cached_guids_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    story_features, summary_features, src_ids, tgt_ids = features
    # Convert to Tensors and build dataset
    story_ids = torch.tensor([f.input_ids for f in story_features], dtype=torch.long)
    story_attention_mask = torch.tensor([f.attention_mask for f in story_features], dtype=torch.long)
    story_token_type_ids = torch.tensor([f.token_type_ids for f in story_features], dtype=torch.long)

    summary_ids = torch.tensor([f.input_ids for f in summary_features], dtype=torch.long)
    summary_attention_mask = torch.tensor([f.attention_mask for f in summary_features], dtype=torch.long)
    summary_token_type_ids = torch.tensor([f.token_type_ids for f in summary_features], dtype=torch.long)

    src_ids = torch.tensor([f.input_ids for f in src_ids], dtype=torch.long)
    tgt_ids = torch.tensor([f.input_ids for f in tgt_ids], dtype=torch.long)
    src_attention_mask = torch.tensor([f.attention_mask for f in src_ids], dtype=torch.long)
    src_token_type_ids = torch.tensor([f.token_type_ids for f in src_ids], dtype=torch.long)

    dataset = TensorDataset(story_ids, story_attention_mask, story_token_type_ids,
                            summary_ids, summary_attention_mask, summary_token_type_ids
                            src_ids, tgt_ids, src_attention_mask, src_token_type_ids)
    return dataset, guids


def tokenzie(tokenizer, sentence, max_length, add_special_tokens=True):
    """Tokenize string to list of ids.
    """    
    inputs = tokenizer.encode_plus(
        sentence,
        add_special_tokens=add_special_tokens,
        max_length=max_length,
    )
    return inputs["input_ids"], inputs["token_type_ids"]


def convert_outputs_to_features(examples, tokenizer,
                                max_length=200,
                                pad_on_left=False,
                                pad_token=0,
                                pad_token_segment_id=0,
                                mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet
            where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    src_features, tgt_features = [], []
    for (ex_index, example) in enumerate(examples):
        inputs = tokenzie(tokenizer, example.label, max_length - 1, False)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # add [BOS] & [EOS] tokens
        source_ids = [tokenizer.bos_token_id] + input_ids
        target_ids = input_ids + [tokenizer.eos_token_id]
        token_type_ids = token_type_ids + [0]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(source_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(source_ids)
        if pad_on_left:
            source_ids = ([pad_token] * padding_length) + source_ids
            target_ids = ([pad_token] * padding_length) + target_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            source_ids = source_ids + ([pad_token] * padding_length)
            target_ids = target_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(source_ids) == max_length, \
            "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(target_ids) == max_length, \
            "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, \
            "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, \
            "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        # if ex_index < 5:
        #     print("*** Example ***")
        #     print("guid: %s" % (example.guid))
        #     print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))

        src_features.append(InputFeatures(input_ids=source_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids))
        tgt_features.append(InputFeatures(input_ids=target_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids))
    return src_features, tgt_features


def convert_inputs_to_features(examples, tokenizer,
                               summary=True,
                               max_length=512,
                               pad_on_left=False,
                               pad_token=0,
                               pad_token_segment_id=0,
                               mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples.
        summary: if ture convert summary, otherwise convert story 
        max_length: Maximum example length
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet
            where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    features, guids = [], []
    for (ex_index, example) in enumerate(examples):
        # if ex_index % 10000 == 0:
        #     print("Writing example %d" % (ex_index))
        example_data = example.text_b if summary else example.text_a

        inputs = tokenzie(tokenizer, example_data, max_length, False)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, \
            "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, \
            "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, \
            "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        # if ex_index < 5:
        #     print("*** Example ***")
        #     print("guid: %s" % (example.guid))
        #     print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))

        guids.append(example.guid)
        features.append(InputFeatures(input_ids=input_ids,
                                      attention_mask=attention_mask,
                                      token_type_ids=token_type_ids))
    return features, guids
