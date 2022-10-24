import jsonlines
import json
import numpy as np
import os
from PIL import Image
import random
from collections import Counter
import torch
from torch.utils.data import Dataset

from utils.utils import truncate_seq_pair, numpy_seed


class JsonlDataset(Dataset):
    def __init__(self, data_path, img_path, tokenizer, transforms, vocab, args,
                 num_expanded=None, labeled_examples_per_class=0, drop_overlapped=[],
                 text_aug0='none', text_aug=None):
        self.data = [obj for obj in jsonlines.open(data_path)]
        print(f'Read {len(self.data)} images from file {data_path.split("/")[-1]}.')
        
        if drop_overlapped:
            with open('/home/ubuntu/CrisisMMD/MmbtSplits/overlapped.json') as f:
                overlapped = json.load(f)
            to_drop = []
            for k, v in overlapped.items():
                for t in drop_overlapped:
                    to_drop += v[t]
            to_drop = set(to_drop)
            self.data = [x for x in self.data if x['id'] not in to_drop]
#             print(f'Kept {len(self.data)} non-overlapping images.')
        
        labels = {x['label'] for x in self.data}
#         print(f'Labels found are: {Counter(labels)}.')
        
        if labeled_examples_per_class > 0:
            data = []
            for label in labels:
                label_data = [x for x in self.data if x['label'] == label]
                label_data = random.sample(label_data, labeled_examples_per_class)
                data += label_data
            random.shuffle(data)
            self.data = data
            print(f'Loaded {len(self.data)} images equally ditributed from {len(labels)} classes.')
            
        if num_expanded is not None and num_expanded > 0:
            indexes = self._expansion_indexes(len(self.data), num_expanded)
            self.data = [self.data[i] for i in indexes]
            print(f'Loaded {len(self.data)} images after expansion.')
        print(f'Final data split as: {Counter([x["label"] for x in self.data])}')
        print()
        
        self.img_dir = img_path
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["image"] = None

        self.max_seq_len = args.max_seq_len
        if args.model == "mmbt":
            self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms
        self.errors = 0
        self.text_aug0 = text_aug0
        self.text_aug = text_aug

            
    def __len__(self):
        return len(self.data)
            
    def _expansion_indexes(self, initial_size, size):
        expand_count = size // initial_size
        indexes = np.arange(initial_size)
        expanded = np.hstack([indexes for _ in range(expand_count)]) if expand_count else np.array([], dtype=int)
        
        if expanded.shape[0] < size:
            diff = size - expanded.shape[0]
            expanded = np.hstack([expanded, np.random.choice(indexes, diff)])
        
        assert expanded.shape[0] == size
        return expanded
    
    def get_classic_item(self, index):
        try:
            image = Image.open(
                os.path.join(self.img_dir, self.data[index]["image"])
            ).convert("RGB")
        except:
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
        image = self.transforms(image)
        
        label = self.args.labels.index(self.data[index]["label"])

        return image, label
        
    
    def get_multimodal_item_orig(self, index):
        if self.args.task == "vsnli":
            sent1 = self.tokenizer(self.data[index]["sentence1"])
            sent2 = self.tokenizer(self.data[index]["sentence2"])
            truncate_seq_pair(sent1, sent2, self.args.max_seq_len - 3)
            sentence = self.text_start_token + sent1 + ["[SEP]"] + sent2 + ["[SEP]"]
            segment = torch.cat(
                [torch.zeros(2 + len(sent1)), torch.ones(len(sent2) + 1)]
            )
        else:
            sentence = (
                self.text_start_token
                + self.tokenizer(self.data[index]["text"])[
                    : (self.args.max_seq_len - 1)
                ]
            )
            segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"]]
            ] = 1
        else:
            label = torch.LongTensor(
                [self.args.labels.index(self.data[index]["label"])]
            )

        image = None
        try:
            if self.args.model in ["img", "concatbow", "concatbert", "mmbt"]:
                if self.data[index]["image"]:
                    image = Image.open(
                        os.path.join(self.img_dir, self.data[index]["image"])
                    ).convert("RGB")
                else:
                    image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
                image = self.transforms(image)
        except:
            self.errors += 1
            if self.errors > len(self.data) / 100:
                raise ValueError('wrong paths')
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            image = self.transforms(image)
        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1

        return sentence, segment, image, label

    def get_mmbt_sentence_and_segment(self, text):
        sentence = (
            self.text_start_token
            + self.tokenizer(text)[
                : (self.args.max_seq_len - 1)
            ]
        )
        segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )
        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1

        return sentence, segment

    def get_mmbt_label(self, label):
        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            label[
                [self.args.labels.index(tgt) for tgt in label]
            ] = 1
        else:
            label_idx = self.args.labels.index(label) if label != 'unknown' else -1
            label = torch.LongTensor([label_idx])
        return label

    def get_mmbt_image(self, img_path):
        image = None
        try:
            if self.args.model in ["img", "concatbow", "concatbert", "mmbt"]:
                if img_path:
                    image = Image.open(
                        os.path.join(self.img_dir, img_path)
                    ).convert("RGB")
                else:
                    image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
                image = self.transforms(image)
        except:
            self.errors += 1
            if self.errors > len(self.data) / 100:
                raise ValueError('wrong paths')
            image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            image = self.transforms(image)
        return image

    def _get_text(self, index, aug):
        if aug == 'none':
            text = self.data[index]["text"]
        else:
            text = random.choice(self.data[index][aug])

        sentence, segment = self.get_mmbt_sentence_and_segment(text)
        return sentence, segment


    def get_multimodal_item(self, index):
        sentence, segment = self._get_text(index, self.text_aug0)

        label = self.get_mmbt_label(self.data[index]["label"])
        image = self.get_mmbt_image(self.data[index]["image"])
        
        if self.text_aug is None:
            return sentence, segment, image, label
        else:
            sentence_aug, segment_aug = self._get_text(index, self.text_aug)
            return sentence, segment, image, label, sentence_aug, segment_aug
        
    
    def __getitem__(self, index):
        if self.args.my_format == 'classic':
            return self.get_classic_item(index)
        elif self.args.my_format =='multimodal':
            return self.get_multimodal_item(index)
        else:
            raise ValueError(f'Unrecognized args.my_format {self.args.my_format}')
