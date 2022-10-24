import jsonlines
import os

# data_path = "D:\IUSTIN\DoinaData\data_splits\humanitarian_orig"
# in_file1 = "train_bt_en-de-en.jsonl"
# in_file2 = "train_bt_en-de-fr-en.jsonl"
# out_file = "train_bt.jsonl"

data_path = r"D:/IUSTIN/DoinaData/data_splits/unlabeled"
in_file1 = "unlabeled_bt_en-de-en.jsonl"
in_file2 = "unlabeled_bt_en-de-fr-en.jsonl"
out_file = "unlabeled_bt.jsonl"

def main():

    with jsonlines.open(os.path.join(data_path, in_file1)) as f1, \
        jsonlines.open(os.path.join(data_path, in_file2)) as f2, \
        jsonlines.open(os.path.join(data_path, out_file), 'w') as f_out:
            for obj1, obj2 in zip(f1, f2):
                assert obj1['id'] == obj2['id']
                obj = {**obj1, **obj2}
                f_out.write(obj)


if __name__ == "__main__":
    main()
