SET DATA_PATH=C:/Users/iustin.sirbu/Documents/IUSTIN/DoinaData
python -m scripts.add_eda_to_jsonl ^
--input_file "%DATA_PATH%/data_splits/humanitarian_orig/train.jsonl" ^
--output_file "%DATA_PATH%/data_splits/humanitarian_orig/train_eda.jsonl" ^
--input_file "%DATA_PATH%/data_splits/informative_orig/train.jsonl" ^
--output_file "%DATA_PATH%/data_splits/informative_orig/train_eda.jsonl" ^
--input_file "%DATA_PATH%/data_splits/unlabeled/unlabeled_clean.jsonl" ^
--output_file "%DATA_PATH%/data_splits/unlabeled/unlabeled_eda.jsonl" ^
--num_augmentations 10 ^
--override
