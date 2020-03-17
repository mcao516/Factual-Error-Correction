input_file, output_file = 'fc_test_preds_bm1_cpbest2.hypo', 'fc_test_preds_bm1_cpbest2_.hypo'

with open(input_file, 'r', encoding='utf-8') as sf, open(output_file, 'w', encoding='utf-8') as tf:
    for line in sf:
        line = line.strip()
        if len(line) > 0 and (line[0] == '"' or line[0] == "'"):
            if line.find(line[0], 1) == -1:
                line = line[1:]
        tf.write(line + '\n')
