import json
import random
import argparse

SEP_TOKEN = '</s>'


def ctype_sampler_1(data):
    """50% probability sample a clean summary; 50% uniformly sample a corrupted summary.
    """
    if random.random() < 0.5:
        sample_type = 'clean'
    else:
        types = list(data.keys())
        types.remove('text')
        if len(types) >= 2:
            types.remove('clean')
        sample_type = random.sample(types, 1)[0]
    return [sample_type]


def ctype_sampler_2(data):
    """For each data sample, sample a clean summary and a corrupted one.
    """
    types = list(data.keys())
    types.remove('text')

    samples = ['clean']
    if len(types) > 1:
        types.remove('clean')
        samples.append(random.sample(types, 1)[0])
    return samples


def create_training_data(data_dict, source_file, target_file, corruption_types, duplicate=False):
    """Create training data. Half clean summaries and half sampling from other types.

    Arguments:
        data_dict -- {id: {'text': '', 'clean': '', 'numswp': ''}}
        source_file -- str, source file path
        target_file -- str, target file path
        corruption_types -- list of types
    """
    sampled_type_counts = {t: 0 for t in corruption_types}

    with open(source_file, 'w', encoding='utf-8') as sf, open(target_file, 'w', encoding='utf-8') as tf:
        for id in data_dict:
            doc_text = data_dict[id]['text']
            summary = data_dict[id]['clean']

            if duplicate:
                sampled_types = ctype_sampler_2(data_dict[id])
            else:
                sampled_types = ctype_sampler_1(data_dict[id])

            for st in sampled_types:
                corrputed = data_dict[id][st]
                sampled_type_counts[st] += 1

                sf.write(corrputed + ' {} '.format(SEP_TOKEN) + doc_text + '\n')
                sf.flush()
                tf.write(summary + '\n')
                tf.flush()

    print('Created dataset:')
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
    print('Dataset size: {}'.format(len(data_dict)))
    print('All corruption counts:')
    print(type_counts)

    if args.duplicate:
        create_training_data(data_dict, args.source_fout, args.target_fout, corruption_types, True)
    else:
        create_training_data(data_dict, args.source_fout, args.target_fout, corruption_types, False)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("--data_type", type=str, help="train, val or test.")
    PARSER.add_argument("--corruption_types", type=str, nargs="+", default=(), help="")
    PARSER.add_argument("--source_fout", type=str, help="")
    PARSER.add_argument("--target_fout", type=str, help="")
    PARSER.add_argument('--duplicate', action='store_true')

    # clean dateswp entswp negation numswp pronoun
    ARGS = PARSER.parse_args()
    main(ARGS)