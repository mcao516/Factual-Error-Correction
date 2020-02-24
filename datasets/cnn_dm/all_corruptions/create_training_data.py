import json
import random
import argparse


SEP_TOKEN = '</s>'


def create_uniform(data_dict, source_file, target_file, corruption_types):
    """Create uniform data. Half clean summaries and half sampling from other types.

    Arguments:
        data_dict -- {id: {'text': '', 'clean': '', 'numswp': ''}}
        source_file -- str, source file path
        target_file -- str, target file path
        corruption_types -- [description]
    """
    sampled_type_counts = {t: 0 for t in corruption_types}

    with open(source_file, 'w', encoding='utf-8') as sf, open(target_file, 'w', encoding='utf-8') as tf:
        for id in data_dict:
            doc_text = data_dict[id]['text']
            summary = data_dict[id]['clean']

            # 50% chance sample corrupted summaries
            if id % 2 == 0:
                sample_type = 'clean'
            else:
                types = list(data_dict[id].keys())
                types.remove('text')
                if len(types) >= 2:
                    types.remove('clean')
                sample_type = random.sample(types, 1)[0]

            corrputed = data_dict[id][sample_type]
            sampled_type_counts[sample_type] += 1

            sf.write(corrputed + ' {} '.format(SEP_TOKEN) + doc_text + '\n')
            sf.flush()
            tf.write(summary + '\n')
            tf.flush()

    print('Create uniform data:')
    print(sampled_type_counts)


def create_mixed(data_dict, source_file, target_file, corruption_types):
    """Create mixed data. For each summary, use one clean sample and one corruption sample.

    Arguments:
        data_dict -- {id: {'text': '', 'clean': '', 'numswp': ''}}
        source_file -- str, source file path
        target_file -- str, target file path
        corruption_types -- [description]
    """
    sampled_type_counts = {t: 0 for t in corruption_types}

    with open(source_file, 'w', encoding='utf-8') as sf, open(target_file, 'w', encoding='utf-8') as tf:
        for id in data_dict:
            doc_text = data_dict[id]['text']
            summary = data_dict[id]['clean']

            types = list(data_dict[id].keys())
            types.remove('text')
            if len(types) <= 1:
                continue
            types.remove('clean')
            sample_type = random.sample(types, 1)[0]

            corrputed = data_dict[id][sample_type]
            sampled_type_counts['clean'] += 1
            sampled_type_counts[sample_type] += 1

            sf.write(corrputed + ' {} '.format(SEP_TOKEN) + doc_text + '\n')
            sf.write(summary + ' {} '.format(SEP_TOKEN) + doc_text + '\n')
            sf.flush()
            tf.write(summary + '\n')
            tf.write(summary + '\n')
            tf.flush()

    print('Create mixed data:')
    print(sampled_type_counts)


def read_data(file_type, corruption_types):
    # read corruption data
    data_dict, type_counts = {}, {t: 0 for t in corruption_types}
    for ct in corruption_types:
        print('- read {} data'.format(ct))
        ct_file_path = file_type + '-' + ct + '.jsonl'
        with open(ct_file_path, 'r', encoding='utf-8') as handle:
            for line in handle:
                d = json.loads(line.strip())
                if d["id"] not in data_dict:
                    data_dict[d["id"]] = {}
                    data_dict[d["id"]]['text'] = d["text"]
                data_dict[d["id"]][ct] = d["claim"]
                type_counts[ct] += 1

    return data_dict, type_counts


def main(args):
    file_type = args.data_type
    corruption_types = list(args.corruption_types)
    print('All corruption types:')
    print(corruption_types)

    data_dict, type_counts = read_data(file_type, corruption_types)
    print('Data size:')
    print(len(data_dict))
    print('Total corruption counts:')
    print(type_counts)

    create_uniform(data_dict, file_type + '-uni.source', file_type + '-uni.target', corruption_types)
    create_mixed(data_dict, file_type + '-mixed.source', file_type + '-mixed.target', corruption_types)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--data_type", type=str, help="train, val or test.")
    PARSER.add_argument("--corruption_types", type=str, nargs="+", default=(), help="")
    # clean dateswp entswp negation numswp pronoun
    ARGS = PARSER.parse_args()
    main(ARGS)