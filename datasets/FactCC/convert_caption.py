input_file, output_file = 'val/val.source', 'val/val_upper.source'

with open(input_file, 'r', encoding='utf-8') as sf, open(output_file, 'w', encoding='utf-8') as tf:
    for line in sf:
        line = line.strip()
        if len(line) > 0:
            tf.write(line[0].upper() + line[1:] + '\n')