import functools
import jsonlines
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer

from data.myDataset import JsonlDataset
from data.vocab import Vocab

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from data.randaugment import RandAugmentMC


PRECOMPUTED_MEANS = {
    'CrisisMMD/informative_orig': (0.485, 0.456, 0.406),    # ImageNet
    'CrisisMMD/humanitarian_orig': (0.485, 0.456, 0.406),   # ImageNet
}

PRECOMPUTED_STDS = {
    'CrisisMMD/informative_orig': (0.229, 0.224, 0.225),    # ImageNet
    'CrisisMMD/humanitarian_orig': (0.229, 0.224, 0.225),   # ImageNet
}


IMG_SIZE = 224
# TEXT_SIZE = 224
TEXT_SIZE = 192

def set_classic_arch_params(args):
    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        if args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 10
        if args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
            
    elif args.dataset in ['doina32', 'doina32m', 'CrisisMMD']:
        args.num_classes = 2
        if args.task == 'damage' or args.task == 'humanitarian':
            args.num_classes = 3
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        if args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
         
            
def set_my_format(args):
    args.my_format = 'classic'
    if args.arch in ['mmbt'] and args.model in ['mmbt']:
        args.my_format = 'multimodal'


def check_my_format(args):
    if args.arch in ['mmbt'] and args.model in ['mmbt']:
        assert args.my_format == 'multimodal'
    elif args.model in ['efficientnet_b3']:
        assert args.my_format == 'image_only'
    else:
        assert args.my_format == 'classic'


def get_precomputed_mean_and_std(args):
    if args.dataset in PRECOMPUTED_MEANS:
        key = args.dataset
    else:
        key = f'{args.dataset}/{args.task}'
        if key not in PRECOMPUTED_MEANS:
            raise ValueError('Mean and Std for dataset are not precomputed.')
    return PRECOMPUTED_MEANS[key], PRECOMPUTED_STDS[key]

def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [obj["label"] for obj in jsonlines.open(path)]

    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

#     print(list(label_freqs.keys()))
    print(f'Labels frequencies: {label_freqs}')

    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()
    if args.model in ["bert", "mmbt", "concatbert"]:
        bert_tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True
        )
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)

    else:
        word_list = get_glove_words(args.glove_path)
        vocab.add(word_list)

    return vocab


def prepare_text_segment_mask(batch, index_text, index_seg):
    lens = [len(row[index_text]) for row in batch]
    # bsz, max_seq_len = len(batch), max(lens) if TEXT_SIZE is None else TEXT_SIZE
    TEXT_SIZE2 = TEXT_SIZE #if index_text == 0 else 144
    bsz, max_seq_len = len(batch), max(lens) if TEXT_SIZE2 is None else TEXT_SIZE2

    mask_tensor = torch.zeros(bsz, TEXT_SIZE).long()
    text_tensor = torch.zeros(bsz, TEXT_SIZE).long()
    segment_tensor = torch.zeros(bsz, TEXT_SIZE).long()

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        if length > max_seq_len:
            length = max_seq_len
        tokens, segment = input_row[index_text], input_row[index_seg]
        text_tensor[i_batch, :length] = tokens[:length]
        segment_tensor[i_batch, :length] = segment[:length]
        mask_tensor[i_batch, :length] = 1
    
    return text_tensor, segment_tensor, mask_tensor


def collate_fn_mmbt(batch, args):
    text_tensor, segment_tensor, mask_tensor = prepare_text_segment_mask(batch, 0, 1)

    # if we have text_aug
    if len(batch[0]) == 6:
        text_aug_tensor, segment_aug_tensor, mask_aug_tensor = prepare_text_segment_mask(batch, 4, 5)
        text_aug_data = (text_aug_tensor, segment_aug_tensor, mask_aug_tensor)
    else:
        text_aug_data = None

    img_tensor = None
    if args.model in ["img", "concatbow", "concatbert", "mmbt"]:
        if isinstance(batch[0][2], tuple):
            assert len(batch[0][2]) == 2
            img_tensor1 = torch.stack([row[2][0] for row in batch])
            img_tensor2 = torch.stack([row[2][1] for row in batch])
            img_tensor = (img_tensor1, img_tensor2)
        else:
            img_tensor = torch.stack([row[2] for row in batch])
        assert img_tensor is not None

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[3] for row in batch])
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[3] for row in batch]).long()

    return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, text_aug_data

    
# def collate_fn(batch, args):
#     if args.arch == 'mmbt':
#         return collate_fn_mmbt(batch, args)
#     if args.arch in ['wideresnet', 'resnext']:
#         return collate_fn_resnet(batch, args)
#     raise NotImplemented('Unknown architecture.')



class TransformFix(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=IMG_SIZE,
                                  padding=int(IMG_SIZE*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=IMG_SIZE,
                                  padding=int(IMG_SIZE*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

    
def get_datasets(args, num_expand_x, num_expand_u):
    
    precomputed_mean, precomputed_std = get_precomputed_mean_and_std(args)

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    
    tokenizer = (
        BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        if args.model in ["bert", "mmbt", "concatbert"]
        else str.split
    )
    
    transform_lab = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=IMG_SIZE,
                              padding=int(IMG_SIZE*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=precomputed_mean, std=precomputed_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(size=IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=precomputed_mean, std=precomputed_std)
    ])
    transform_unl = TransformFix(mean=precomputed_mean, std=precomputed_std)

    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, f"{args.train_file}.jsonl")
    )
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)
    print('args.labels:', args.labels)
    print('args.label_freqs', args.label_freqs)
    print('n_classes:', args.n_classes)

    labeled_examples_per_class = 0 if args.unbalanced else num_expand_x//args.n_classes
    labeled_set = JsonlDataset(
        os.path.join(args.data_path, args.task, f"{args.train_file}.jsonl"),
        args.img_path,
        tokenizer,
        transform_lab,
        vocab,
        args,
        num_expanded=num_expand_x,
        labeled_examples_per_class=labeled_examples_per_class,
        text_aug0=args.text_soft_aug
    )

    args.train_data_len = len(labeled_set)

    unlabeled_set = JsonlDataset(
        os.path.join(args.data_path, 'unlabeled', f"{args.unlabeled_dataset}.jsonl"),
        args.img_path,
        tokenizer,
        transform_unl,
        vocab,
        args,
        # num_expanded=num_expand_u,
        # drop_overlapped=['test', 'dev'],
        text_aug0=args.text_soft_aug,
        text_aug=args.text_hard_aug
    )
    
    dev_set = JsonlDataset(
        os.path.join(args.data_path, args.task, "dev.jsonl"),
        args.img_path,
        tokenizer,
        transform_val,
        vocab,
        args,
        text_aug0='none'
    )
    
    test_set = JsonlDataset(
        os.path.join(args.data_path, args.task, "test.jsonl"),
        args.img_path,
        tokenizer,
        transform_val,
        vocab,
        args,
        text_aug0='none'
    )
    
    return labeled_set, unlabeled_set, dev_set, test_set


def get_multimodal_data_loaders(labeled_set, unlabeled_set, dev_set, test_set, args):
    collate = functools.partial(collate_fn_mmbt, args=args)
    
    labeled_loader = DataLoader(
        labeled_set,
        batch_size=args.batch_size,
#         sampler=train_sampler(train),
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True
    )
    
    unlabeled_loader = DataLoader(
        unlabeled_set,
#         sampler=train_sampler(unlabeled_set),
        shuffle=True,
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True
    )
    
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
#         sampler=SequentialSampler(dev),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
#         sampler=SequentialSampler(test_set),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    return labeled_loader, unlabeled_loader, dev_loader, test_loader

def get_classic_data_loaders(labeled_set, unlabeled_set, dev_set, test_set, args):
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_loader = DataLoader(
        labeled_set,
        sampler=train_sampler(labeled_set),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_loader = DataLoader(
        unlabeled_set,
        sampler=train_sampler(unlabeled_set),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)
    
    dev_loader = DataLoader(
        dev_set,
        sampler=SequentialSampler(dev_set),
        batch_size=args.batch_size,
        num_workers=args.num_workers) 

    test_loader = DataLoader(
        test_set,
        sampler=SequentialSampler(test_set),
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    
    
    return labeled_loader, unlabeled_loader, dev_loader, test_loader

def get_data_loaders(args):
    labeled_set, unlabeled_set, dev_set, test_set = get_datasets(args, args.k_img, args.k_img*args.mu)
    
    if args.local_rank == 0:
        torch.distributed.barrier()
        
    if args.my_format == 'classic':
            return get_classic_data_loaders(labeled_set, unlabeled_set, dev_set, test_set, args)
    elif args.my_format =='multimodal':
        return get_multimodal_data_loaders(labeled_set, unlabeled_set, dev_set, test_set, args)
    else:
        raise ValueError(f'Unrecognized args.my_format {args.my_format}')
