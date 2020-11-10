FILE_TYPE=val

python create_data.py --source_file ../cnn-dailymail/$FILE_TYPE.source --target_file ../cnn-dailymail/$FILE_TYPE.target --augmentations entity_swap pronoun_swap date_swap number_swap --save_intermediate
