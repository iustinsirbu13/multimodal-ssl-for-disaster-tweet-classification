SET DATA_PATH=C:/Users/iustin.sirbu/Documents/IUSTIN/DoinaData
python -m scripts.clean_tweets_jsonl ^
--input_file "%DATA_PATH%/data_splits_legacy/humanitarian_orig/dev.jsonl" ^
--output_file "%DATA_PATH%/data_splits/humanitarian_orig/dev.jsonl" ^
--input_file "%DATA_PATH%/data_splits_legacy/humanitarian_orig/test.jsonl" ^
--output_file "%DATA_PATH%/data_splits/humanitarian_orig/test.jsonl" ^
--input_file "%DATA_PATH%/data_splits_legacy/humanitarian_orig/train.jsonl" ^
--output_file "%DATA_PATH%/data_splits/humanitarian_orig/train.jsonl" ^
--input_file "%DATA_PATH%/data_splits_legacy/informative_orig/dev.jsonl" ^
--output_file "%DATA_PATH%/data_splits/informative_orig/dev.jsonl" ^
--input_file "%DATA_PATH%/data_splits_legacy/informative_orig/test.jsonl" ^
--output_file "%DATA_PATH%/data_splits/informative_orig/test.jsonl" ^
--input_file "%DATA_PATH%/data_splits_legacy/informative_orig/train.jsonl" ^
--output_file "%DATA_PATH%/data_splits/informative_orig/train.jsonl"
