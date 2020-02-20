FILE_TYPE=test

nohup python create_data.py --source_file ../cnn_dm/fairseq_files/$FILE_TYPE.source --target_file ../cnn_dm/fairseq_files/$FILE_TYPE.target --augmentations backtranslation entity_swap pronoun_swap date_swap number_swap negation --save_intermediate &
