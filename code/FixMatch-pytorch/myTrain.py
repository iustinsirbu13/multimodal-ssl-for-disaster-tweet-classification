import argparse
import logging
import math
import os
import random
import shutil
import time
from copy import deepcopy
from collections import OrderedDict
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter, accuracy, binary_metrics, all_metrics

# From MMBT
import argparse
from tqdm import tqdm

import torch
from pytorch_pretrained_bert import BertAdam

from data.myHelpers import get_data_loaders, set_my_format, set_classic_arch_params

from models import get_model
from utils.utils import *
###

logger = logging.getLogger(__name__)


best_acc = 0
global_step = 0


def get_args(parser):
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'doina32', 'doina32m', 'CrisisMMD'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument('--arch', default='mmbt', type=str,
                        choices=['wideresnet', 'resnext', 'mmbt'],
                        help='dataset name')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0.1, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=False,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--k-img', default=65536, type=int,
                        help='number of labeled examples')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    
def get_my_args(parser):
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau"])
    parser.add_argument("--my_format", type=str, default="multimodal", choices=["classic", "multimodal"],
                        help="classic should be used for old fixmatch architectures (wideresnet, resnext), "
                        "multimodal for the ones added along with mmbt. Set using set_my_format method.")
    parser.add_argument("--unlabeled_dataset", type=str, default="train",
                        choices=["info_train", "unlabeled_122k", "unlabeled_bt", "unlabeled", "unlabeled_eda"])
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='Only to evaluate the checkpoint given by args.resume')
    parser.add_argument('--unbalanced', action='store_true', default=False,
                        help="Don't try to use balanced classes")
    parser.add_argument('--linear_lu', action='store_true', default=False,
                        help='Linearily increase lambda_u')
    parser.add_argument('--random_lu', action='store_true', default=False,
                        help='Randomly choose lambda_u for each batch.')
    parser.add_argument('--distil', type=str, default='none', choices=['none', 'linear', 'unlabeled'])
    parser.add_argument('--acc_lu', type=int, default=-1,
                       help='Use lambda_u according to last accuracy')
    parser.add_argument('--lambda-u-min', default=0, type=float,
                        help='starting coefficient of unlabeled loss in case of linear schedule for it')
    parser.add_argument('--lambda-u-max', default=50, type=float,
                        help='max coefficient of unlabeled loss in case of random schedule for it')
    parser.add_argument('--train_file', type=str, default='train')
#     parser.add_argument("--labeled_examples_per_class", type=int, default=0)
    parser.add_argument('--text_soft_aug', type=str, default='none', choices=['none', 'back_translate', 'eda', 'eda_01', 'eda_sr', 'en-de-en'])
    parser.add_argument('--text_hard_aug', type=str, default='none', choices=['none', 'back_translate', 'eda', 'eda_01', 'eda_02', 'en-de-fr-en'])
    parser.add_argument('--text_prob_aug', type=float, default=1.0, help='probability of using augmented text')

def get_mmbt_args(parser):
    parser.add_argument("--batch_sz", type=int, default=4)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--data_path", type=str,
                        default='/media/iustins/My Passport/Research/DoinaDisasters/data_splits')
                        # default="/home/ubuntu/CrisisMMD/MmbtSplits")
    parser.add_argument("--img_path", type=str,
                        default='/media/iustins/My Passport/Research/DoinaDisasters')
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=3)
    parser.add_argument("--freeze_txt", type=int, default=0) # it was 5
    parser.add_argument("--glove_path", type=str, default="/home/ubuntu/doina_data/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=10)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
#     parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    # parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="mmbt", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt"])
    # parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--num_image_embeds", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="/home/ubuntu/DoinaDisasters/FixMatch-pytorch/mmbt_tmp")
#     parser.add_argument("--seed", type=int, default=123)
    # parser.add_argument("--task", type=str, default="informative", choices=[
    #     "mmimdb", "vsnli", "food101", 'disaster_data', 'samples_4k_mmbt', 'informative', 'damage', 'humanitarian',
    #     'humanitarian1k', 'humanitarianO', 'informativeO', 'humanitarianM', 'humanitarianR'])
    parser.add_argument("--task", type=str, required=True, choices=['informative_orig', 'humanitarian_orig'])
    parser.add_argument("--task_type", type=str, default="classification", choices=["multilabel", "classification"])
#     parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def get_mmbt_optimizer(model, args):
    if args.model in ["bert", "concatbert", "mmbt"]:
        total_steps = (
            args.k_img
            / args.batch_size
            / args.gradient_accumulation_steps
            * args.epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=total_steps+1,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer

def get_mmbt_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=args.lr_patience, verbose=True, factor=args.lr_factor, min_lr=0.000001
    )

def get_optimizer(model, args):
    if args.optimizer == 'adam':
        return get_mmbt_optimizer(model, args)
    if args.optimizer == 'sgd':
        my_momentum = 0.9 #0.9
        print(f'Momentum set to {my_momentum}, nesterov set to {args.nesterov}')
        return optim.SGD(model.parameters(), lr=args.lr, momentum=my_momentum, nesterov=args.nesterov)
    raise ValueError('Invalid optimizer argument')

def get_scheduler(optimizer, args):
    if args.scheduler == 'plateau':
        return get_mmbt_scheduler(optimizer, args)
    if args.scheduler == 'cosine':
        return get_cosine_schedule_with_warmup(optimizer, args.warmup * args.iteration, args.total_steps)
    raise ValueError('Invalid scheduler argument')
    
    
def main():
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    get_args(parser)
    get_mmbt_args(parser)
    get_my_args(parser)
    args = parser.parse_args()
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=args.dropout,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
            
        elif args.arch == 'mmbt':
            model = get_model(args)

        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))

        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    args.device = device
    print("Num GPUs", args.n_gpu)
    print("Device", args.device)
    
    set_classic_arch_params(args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed != -1:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        writer = SummaryWriter(args.out)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    set_my_format(args)
    labeled_trainloader, unlabeled_trainloader, valid_loader, test_loader = get_data_loaders(args)

    model = create_model(args)
    model.to(args.device)
    
    optimizer = get_optimizer(model, args)
    scaler = torch.cuda.amp.GradScaler()

    args.iteration = args.k_img // args.batch_size // args.world_size
    args.total_steps = args.epochs * args.iteration // args.gradient_accumulation_steps
    
    scheduler = get_scheduler(optimizer, args)

    if args.use_ema:
        ema_model = ModelEMA(args, model, args.ema_decay, device)
    else:
        ema_model = None

    start_epoch = 0

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        print('GPU: ', args.gpu_id)
        # assert os.path.isfile(
        #     args.resume), "Error: no checkpoint directory found!"
        if os.path.isfile(args.resume):
            args.out = os.path.dirname(args.resume)
            # checkpoint = torch.load(args.resume, map_location='cpu')
            checkpoint = torch.load(args.resume)
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print(f'Model loaded from epoch {start_epoch}')
            model.to(args.device)
            print('Model sent to device')
            if args.use_ema:
                ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
                print('EMA model loaded')
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            print('Attention! No checkpoint directory found. Training from epoch 0.')

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    test_accs = []
#     model.zero_grad()
    print(f'Using {args.my_format} my_format.')
    
    if args.eval_only:
        print('eval_only is set')
        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model
#         etrain_loss, etrain_acc, bin_etrain = generic_test(args, labeled_trainloader, test_model, None,
#                                                             'EvalTrain', max_samples=1000)
#         print('Train metrics: ', bin_etrain)
#         valid_loss, valid_acc, bin_valid = evaluate_multimodal(args, valid_loader, test_model, None, 'Valid')
#         print('Valid metrics: ', bin_valid)

            
        valid_loss, valid_acc, bin_valid = evaluate_multimodal(args, valid_loader, test_model, None, 'Valid')
        print('Valid metrics: ', valid_loss, valid_acc)
        for k, v in bin_valid.items():
            print('', k, v, sep='\t')
        print()
        test_loss, test_acc, bin_test = evaluate_multimodal(args, test_loader, test_model, None, 'Test')
        print('Test metrics: ', test_loss, test_acc)
        for k, v in bin_test.items():
            print('', k, v, sep='\t')


        return
        
    
    for epoch in range(start_epoch, args.epochs):
        model.zero_grad()

        # Train step
        train_loss, train_loss_x, train_loss_u, mask_prob = generic_train(
                args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_model, scheduler, epoch, scaler)

        if args.no_progress:
            logger.info("Epoch {}. train_loss: {:.4f}. train_loss_x: {:.4f}. train_loss_u: {:.4f}."
                        .format(epoch+1, train_loss, train_loss_x, train_loss_u))

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        # Test step
        etrain_loss, etrain_acc, bin_etrain = generic_test(args, labeled_trainloader, test_model, epoch,
                                                            'EvalTrain', max_samples=1000)
        test_loss, test_acc, bin_test = generic_test(args, valid_loader, test_model, epoch, 'Valid')
#         valid_loss, valid_acc, bin_valid = generic_test(args, test_loader, test_model, epoch, 'Test') 

        if args.scheduler == 'plateau':
            tuning_metric = etrain_loss
            scheduler.step(tuning_metric)
            
        if args.acc_lu != -1:
            args.acc_lu = test_acc
            print('New args.acc_lu', args.acc_lu)
            
        if args.local_rank in [-1, 0]:
            writer.add_scalar('train/1.train_loss', train_loss, epoch)
            writer.add_scalar('train/2.train_loss_x', train_loss_x, epoch)
            writer.add_scalar('train/3.train_loss_u', train_loss_u, epoch)
            writer.add_scalar('train/4.mask', mask_prob, epoch)
            writer.add_scalar('test/1.test_acc', test_acc, epoch)
            writer.add_scalar('test/2.test_loss', test_loss, epoch)
#             writer.add_scalar('valid/1.valid_acc', valid_acc, epoch)
#             writer.add_scalar('valid/2.valid_loss', valid_loss, epoch)
            writer.add_scalar('eval_train/1.etrain_acc', etrain_acc, epoch)
            writer.add_scalar('eval_train/2.etrain_loss', etrain_loss, epoch)
#             for k, v in bin_valid.items():
#                 writer.add_scalar(f'valid/{k}', v, epoch)
            for k, v in bin_test.items():
                writer.add_scalar(f'test/{k}', v, epoch)
            for k, v in bin_etrain.items():
                writer.add_scalar(f'eval_train/{k}', v, epoch)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if args.local_rank in [-1, 0]:
            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

        test_accs.append(test_acc)
        logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
        logger.info('Mean top-1 acc: {:.2f}\n'.format(
            np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        writer.close()

        
def generic_train(args, labeled_loader, unlabeled_loader, model, optimizer, ema_model, scheduler, epoch, scaler):
    if args.my_format == 'multimodal':
        train_loss, train_loss_x, train_loss_u, mask_prob = my_train_v2(
                args, labeled_loader, unlabeled_loader, model, optimizer, ema_model, scheduler, epoch, scaler)
    else:
        train_loss, train_loss_x, train_loss_u, mask_prob = train(
            args, labeled_loader, unlabeled_loader, model, optimizer, ema_model, scheduler, epoch)
    return train_loss, train_loss_x, train_loss_u, mask_prob
        

def softXEnt (input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]

def my_train_v2(args, labeled_loader, unlabeled_loader, model, optimizer, ema_model, scheduler, epoch, scaler):
    if args.amp:
        from apex import amp
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()
    global global_step
    
    if not args.no_progress:
        p_bar = tqdm(range(args.iteration),
                     disable=args.local_rank not in [-1, 0])
        
    train_loader = zip(labeled_loader, unlabeled_loader)
    model.train()
    if args.linear_lu or args.distil =='linear':
        print('Lu weight: ', (epoch / args.epochs) * args.lambda_u)
    print('args.distil', args.distil)
    for batch_idx, (data_x, data_u) in enumerate(train_loader):
        text_x, segment_x, mask_x, img_x, tgt_x, text_aug_data_x = data_x
        text_u, segment_u, mask_u, (img_u_w, img_u_s), _, text_aug_data_u = data_u

        if text_aug_data_x is not None:
            raise NotImplementedError('treat this case')

        if (text_aug_data_u is not None) and (random.random() < args.text_prob_aug):
            text_aug_u, segment_aug_u, mask_aug_u = text_aug_data_u
            # img_u_s = img_u_w
            if batch_idx < 4:
                print('use aug text')
        else:
            text_aug_u, segment_aug_u, mask_aug_u = text_u, segment_u, mask_u

        data_time.update(time.time() - end)
        batch_size = text_x.shape[0]
        
        texts = torch.cat((text_x, text_u, text_aug_u)).to(args.device)
        segments = torch.cat((segment_x, segment_u, segment_aug_u)).to(args.device)
        masks = torch.cat((mask_x, mask_u, mask_aug_u)).to(args.device)
        images = torch.cat((img_x, img_u_w, img_u_s)).to(args.device)
        targets_x = tgt_x.to(args.device)

        # with torch.cuda.amp.autocast():
        logits = model(texts, masks, segments, images)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        del logits

    
        if args.distil == 'unlabeled':
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            Lu = softXEnt(logits_u_s, pseudo_label)
        
        elif args.distil == 'linear':
#             raise ValueError('unimplemented')
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            Lu_soft = softXEnt(logits_u_s, pseudo_label)
            Lu_hard = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
            alpha = 1 - epoch / args.epochs
            Lu = alpha * Lu_soft + (1-alpha) * Lu_hard
        
        else:
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
        
        if args.random_lu:
            args.lambda_u = random.randint(args.lambda_u_min, args.lambda_u_max)
            
        if args.linear_lu:
#             Lu = (epoch / args.epochs) * Lu
            Lu = (args.lambda_u_min + epoch * args.lambda_u / args.epochs) * Lu
        else:
            Lu = args.lambda_u * Lu
        
        loss = (Lx + Lu) / args.gradient_accumulation_steps
    #         loss = Lx + Lu

        loss.backward()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        losses.update(loss.item())
        losses_x.update(Lx.item())
        losses_u.update(Lu.item())
        
        global_step += 1
        if global_step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()            
            if args.scheduler == 'cosine':
                scheduler.step()

        if args.use_ema:
            ema_model.update(model)
        
        batch_time.update(time.time() - end)
        end = time.time()
        mask_prob = mask.mean().item()
        if not args.no_progress:
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.4f}. ".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
                lr=optimizer.param_groups[0]['lr'],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_x=losses_x.avg,
                loss_u=losses_u.avg,
                mask=mask_prob))
            p_bar.update()
    if not args.no_progress:
        p_bar.close()
    return losses.avg, losses_x.avg, losses_u.avg, mask_prob

        
def train(args, labeled_trainloader, unlabeled_trainloader,
          model, optimizer, ema_model, scheduler, epoch):
    if args.amp:
        from apex import amp
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()
    global global_step

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration),
                     disable=args.local_rank not in [-1, 0])

    train_loader = zip(labeled_trainloader, unlabeled_trainloader)
    model.train()
    if args.linear_lu:
        print('Lu weight: ', (epoch / args.epochs) * args.lambda_u)
    print('args.distil', args.distil)
    for batch_idx, (data_x, data_u) in enumerate(train_loader):
        inputs_x, targets_x = data_x
        (inputs_u_w, inputs_u_s), _ = data_u
        data_time.update(time.time() - end)
        batch_size = inputs_x.shape[0]
        inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s)).to(args.device)
        targets_x = targets_x.to(args.device)
        logits = model(inputs)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        del logits
        
#         print('Logits shape', logits_x.shape, logits_u_w.shape, logits_u_s.shape)
#         print('Targets shape', targets_x.shape)        
            
        #$$$$$$$$$$$$$$$$$
        
        if args.distil == 'unlabeled':
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            Lu = softXEnt(logits_u_s, pseudo_label)
            
        elif args.distil == 'linear':
#             raise ValueError('unimplemented')
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            Lu_soft = softXEnt(logits_u_s, pseudo_label)
            Lu_hard = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
            alpha = 1 - epoch / args.epochs
            Lu = alpha * Lu_soft + (1-alpha) * Lu_hard
            
        else:
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()
            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()         
            
        if args.random_lu:
            args.lambda_u = random.randint(args.lambda_u_min, args.lambda_u_max)
            
        if args.linear_lu:
            Lu = (args.lambda_u_min + epoch * args.lambda_u / args.epochs) * Lu
        else:
            Lu = args.lambda_u * Lu
        
        loss = (Lx + Lu) / args.gradient_accumulation_steps
        
        #$$$$$$$$$$$$$$$$$

#         Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

#         pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
#         max_probs, targets_u = torch.max(pseudo_label, dim=-1)
#         mask = max_probs.ge(args.threshold).float()

#         Lu = args.lambda_u * (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

#         if args.linear_lu:
#             Lu = (epoch / args.epochs) * Lu
            
#         loss = (Lx + Lu) / args.gradient_accumulation_steps
        loss.backward()

        losses.update(loss.item())
        losses_x.update(Lx.item())
        losses_u.update(Lu.item())
        
        global_step += 1
        if global_step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()            
            if args.scheduler == 'cosine':
                scheduler.step()

        if args.use_ema:
            ema_model.update(model)

        batch_time.update(time.time() - end)
        end = time.time()
        mask_prob = mask.mean().item()
        if not args.no_progress:
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.4f}. ".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
#                 lr=scheduler.get_last_lr()[0],
                lr=optimizer.param_groups[0]['lr'],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_x=losses_x.avg,
                loss_u=losses_u.avg,
                mask=mask_prob))
            p_bar.update()
    if not args.no_progress:
        p_bar.close()
    return losses.avg, losses_x.avg, losses_u.avg, mask_prob


def generic_test(args, test_loader, test_model, epoch, tst_name='Test', max_samples=-1):
    if args.my_format == 'multimodal':
        loss, acc, bin_metrics = test_doina32m(args, test_loader, test_model, epoch, tst_name, max_samples)
    else:
        loss, acc, bin_metrics = test(args, test_loader, test_model, epoch, tst_name, max_samples)
    return loss, acc, bin_metrics


def evaluate_multimodal(args, test_loader, model, epoch, tst_name='Test', max_samples=-1):
    outputs = []
    targets = []
    
    max_batches = max_samples / args.batch_size if max_samples > 0 else -1
    model.eval()
    with torch.no_grad():
        for batch_idx, data_x in enumerate(test_loader):
            if batch_idx == max_batches:
                break
            if args.my_format == 'multimodal':
                text_x, segment_x, mask_x, img_x, tgt_x, _ = data_x
                text_x, segment_x, mask_x, img_x, tgt_x = text_x.to(args.device), segment_x.to(args.device), mask_x.to(args.device), img_x.to(args.device), tgt_x.to(args.device)
                logits_x = model(text_x, mask_x, segment_x, img_x)
            else:
                img_x, tgt_x = data_x
                img_x, tgt_x = img_x.to(args.device), tgt_x.to(args.device)
                logits_x = model(img_x)
            
            outputs.append(logits_x)
            targets.append(tgt_x)
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
    
    
    loss = F.cross_entropy(outputs, targets)
    acc = accuracy(outputs, targets, topk=(1,))[0]
#     b_metrics = binary_metrics(outputs, targets)
    b_metrics = all_metrics(outputs, targets, n_decimals=4)
    
    stats = {
        'outputs': outputs.tolist(),
        'preds': outputs.argmax(axis=1).tolist(),
        'targets': targets.tolist(),
        'scores': b_metrics,
    }

    # stats_file = f'probabilities2/{args.task}/{stats_file}.json'
    stats_file = os.path.dirname(args.resume)
    stats_file = os.path.join(stats_file, 'stats_file.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f)
        
    return loss.item(), acc.item(), b_metrics


def test_doina32m(args, test_loader, model, epoch, tst_name='Test', max_samples=-1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    bin_metrics = {m : AverageMeter() for m in [
#         'label_0/precision', 'label_0/recall', 'label_0/f1',
#         'label_1/precision', 'label_1/recall', 'label_1/f1',
        'micro/precision', 'micro/recall', 'micro/f1',
        'macro/precision', 'macro/recall', 'macro/f1',
#         'accuracy'
    ]}
    end = time.time()
    
    if isinstance(test_loader, dict):
        test_loader = test_loader['test']

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
    max_batches = max_samples / args.batch_size if max_samples > 0 else -1
    with torch.no_grad():
        for batch_idx, data_x in enumerate(test_loader):
            if batch_idx == max_batches:
                break
            text_x, segment_x, mask_x, img_x, tgt_x, _ = data_x
            data_time.update(time.time() - end)
            model.eval()

            text_x, segment_x, mask_x, img_x, tgt_x = text_x.to(args.device), segment_x.to(args.device), mask_x.to(args.device), img_x.to(args.device), tgt_x.to(args.device)
            logits_x = model(text_x, mask_x, segment_x, img_x)

            outputs, targets = logits_x, tgt_x
            loss = F.cross_entropy(outputs, targets)

            prec1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), text_x.shape[0])
            top1.update(prec1.item(), text_x.shape[0])
            
            b_metrics = binary_metrics(outputs, targets)
            for metric, value in b_metrics.items():
                bin_metrics[metric].update(value, text_x.shape[0])
                
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(tst_name+" Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    for k in bin_metrics:
        bin_metrics[k] = bin_metrics[k].avg
    return losses.avg, top1.avg, bin_metrics

def test(args, test_loader, model, epoch, tst_name='Test', max_samples=-1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
#     top5 = AverageMeter()
#     bin_metrics = {m : AverageMeter() for m in ['label_0/precision', 'label_0/recall', 'label_0/f1', 'label_1/precision', 'label_1/recall', 'label_1/f1', 'f1_micro', 'accuracy']}
    bin_metrics = {m : AverageMeter() for m in [
#         'label_0/precision', 'label_0/recall', 'label_0/f1',
#         'label_1/precision', 'label_1/recall', 'label_1/f1',
        'micro/precision', 'micro/recall', 'micro/f1',
        'macro/precision', 'macro/recall', 'macro/f1',
#         'accuracy'
    ]}
    end = time.time()
    
    if isinstance(test_loader, dict):
        test_loader = test_loader['test']
        
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])
        
    max_batches = max_samples / args.batch_size if max_samples > 0 else -1
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if batch_idx == max_batches:
                break
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            
            loss = F.cross_entropy(outputs, targets)

#             prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            prec1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
#             top5.update(prec5.item(), inputs.shape[0])

            b_metrics = binary_metrics(outputs, targets)
            for metric, value in b_metrics.items():
                bin_metrics[metric].update(value, inputs.shape[0])
                
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(tst_name+" Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
#     logger.info("top-5 acc: {:.2f}".format(top5.avg))
    for k in bin_metrics:
        bin_metrics[k] = bin_metrics[k].avg
    return losses.avg, top1.avg, bin_metrics


class ModelEMA(object):
    def __init__(self, args, model, decay, device='', resume=''):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        self.wd = args.lr * args.wdecay
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        assert isinstance(checkpoint, dict)
        if 'ema_state_dict' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['ema_state_dict'].items():
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
                # weight decay
                if 'bn' not in k:
                    msd[k] = msd[k] * (1. - self.wd)


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
