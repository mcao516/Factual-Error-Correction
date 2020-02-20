import os
import json
import argparse

from tqdm import tqdm


def create_training_data(data, file_path, sep_token='</s>'):
    with open(file_path + '.source', 'w') as sfout, open(file_path + '.target', 'w') as tfout:
        for d in tqdm(data):
            source = d['claim'] + ' {} '.format(sep_token) + d['text']
            target = d['summary']

            sfout.write(source + '\n')
            sfout.flush()
            tfout.write(target + '\n')
            tfout.flush()

def main(args):
    with open(args.data_file, 'r', encoding='utf-8') as handle:
        data = [json.loads(line) for line in handle]
    print("Load {} data samples.".format(len(data)))

    output_file = os.path.splitext(args.data_file)[0] + "-corrupted"
    create_training_data(data, output_file)
    print("Write data to: {}.".format(output_file))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("data_file", type=str, help="Path to file containing source documents.")
    ARGS = PARSER.parse_args()
    main(ARGS)