import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict
import models.wideresnet as wideresnet_models
import models.resnext as resnext_models
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.ema import ModelEMA
from dataset.oct_loader import get_oct_loaders
from utils import AverageMeter, accuracy
import ssl
from transformers import get_cosine_schedule_with_warmup

ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger(__name__)
best_acc = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state, is_best, checkpoint, filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device == "cuda" and torch.cuda.manual_seed_all(args.seed)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main():
    parser = argparse.ArgumentParser(description="OCT PyTorch FixMatch Training")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers")
    parser.add_argument("--dataset", default="/mnt/e/昊翔_all/OCT_DATASET_DO_NOT_SHARE_WITH_ANYONE", type=str, help="dataset path")
    parser.add_argument("--arch", default="resnet18", type=str, choices=["wideresnet", "resnext", "resnet18"], help="architecture name")
    parser.add_argument("--epochs", default=1000, type=int, help="number of total epochs to run")
    parser.add_argument("--lr", "--learning-rate", default=0.03, type=float, help="initial learning rate")
    parser.add_argument("--warmup", default=0, type=float, help="warmup epochs (unlabeled data based)")
    parser.add_argument("--wdecay", default=1e-5, type=float, help="weight decay")
    parser.add_argument("--use-ema", action="store_true", default=True, help="use EMA model")
    parser.add_argument("--ema-decay", default=0.999, type=float, help="EMA decay rate")
    parser.add_argument("--batch-size", default=64, type=int, help="labeled batch size")
    parser.add_argument("--mu", default=7, type=int, help="coefficient of unlabeled batch size")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training")
    parser.add_argument("--lambda-u", default=1, type=float, help="coefficient of unlabeled loss")
    parser.add_argument("--T", default=1, type=float, help="pseudo label temperature")
    parser.add_argument("--threshold", default=0.95, type=float, help="pseudo label threshold")
    parser.add_argument("--out", default="./out", help="directory to output the result")
    parser.add_argument("--no-progress", action="store_true", help="don't use progress bar")
    parser.add_argument("--no-pin-memory", action="store_true", help="Don't pin CPU memory in DataLoader")
    args = parser.parse_args()

    if args.arch == "wideresnet":
        model = wideresnet_models.build_wideresnet(depth=28, widen_factor=8, dropout=0, num_classes=4)
    elif args.arch == "resnext":
        model = resnext_models.build_resnext(cardinality=8, depth=29, width=64, num_classes=4)
    elif args.arch == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 4)
        # load pretrained model
        model.load_state_dict(torch.load("../resnet/resnet18_e20.ckpt"))
    logger.info("Total trainable params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    model.to(device)

    labeled_trainloader, val_loader, unlabeled_trainloader, _ = get_oct_loaders(
        args.dataset, batch_size=args.batch_size, num_workers=args.num_workers, mu=args.mu, pin_memory=not args.no_pin_memory
    )

    if args.seed is not None:
        set_seed(args)

    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup * len(unlabeled_trainloader)),
        num_training_steps=len(unlabeled_trainloader) * args.epochs,
    )

    ema_model = ModelEMA(model, args.ema_decay) if args.use_ema else None

    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, val_loader, model, optimizer, ema_model, scheduler)


def train(args, labeled_trainloader, unlabeled_trainloader, val_loader, model, optimizer, ema_model, scheduler):
    global best_acc
    test_accs = []

    unlabeled_iter = iter(unlabeled_trainloader)
    unlabeled_epoch, unlabeled_current_epoch_percent = 0, 0.0
    model.train()
    for epoch in range(args.epochs):
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        if not args.no_progress:
            p_bar = tqdm(enumerate(labeled_trainloader), disable=0)
        for batch_idx, (inputs_x, targets_x) in p_bar:
            try:
                (inputs_u_weak, inputs_u_strong), _ = next(unlabeled_iter)
                unlabeled_current_epoch_percent += 1 / len(unlabeled_trainloader)
            except:
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_weak, inputs_u_strong), _ = next(unlabeled_iter)
                unlabeled_epoch += 1
                unlabeled_current_epoch_percent = 0.0

            batch_size = inputs_x.shape[0]
            inputs = interleave(torch.cat((inputs_x, inputs_u_weak, inputs_u_strong)), 2 * args.mu + 1).to(device)
            targets_x = targets_x.type(torch.LongTensor).to(device)
            logits = model(inputs)
            logits = de_interleave(logits, 2 * args.mu + 1)
            logits_x = logits[:batch_size]
            logits_u_weak, logits_u_strong = logits[batch_size:].chunk(2)
            del logits

            loss_x = F.cross_entropy(logits_x, targets_x, reduction="mean")

            pseudo_label = torch.softmax(logits_u_weak.detach() / args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            loss_u = (F.cross_entropy(logits_u_strong, targets_u, reduction="none") * mask).mean()
            loss = loss_x + args.lambda_u * loss_u
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            losses.update(loss.item())
            losses_x.update(loss_x.item())
            losses_u.update(loss_u.item())
            mask_probs.update(mask.mean().item())
            if not args.no_progress:
                p_bar.set_description(
                    f"Train Epoch: {epoch}/{args.epochs:4}. Unlabeled epoch: {(unlabeled_epoch + unlabeled_current_epoch_percent):.1f} LR: {scheduler.get_last_lr()[0]:.4f}. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask_probs.avg:.2f}. "
                )
                p_bar.update()

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        test_loss, test_acc = test(val_loader, test_model)

        args.writer.add_scalar("train/1.train_loss", losses.avg, epoch)
        args.writer.add_scalar("train/2.train_loss_x", losses_x.avg, epoch)
        args.writer.add_scalar("train/3.train_loss_u", losses_u.avg, epoch)
        args.writer.add_scalar("train/4.mask", mask_probs.avg, epoch)
        args.writer.add_scalar("test/1.test_acc", test_acc, epoch)
        args.writer.add_scalar("test/2.test_loss", test_loss, epoch)

        best_acc = max(test_acc, best_acc)

        model_to_save = model.module if hasattr(model, "module") else model
        if args.use_ema:
            ema_to_save = ema_model.ema.module if hasattr(ema_model.ema, "module") else ema_model.ema
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model_to_save.state_dict(),
                "ema_state_dict": ema_to_save.state_dict() if args.use_ema else None,
                "acc": test_acc,
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            test_acc > best_acc,
            args.out,
        )

        test_accs.append(test_acc)
        logger.info("Best top-1 acc: {:.2f}".format(best_acc))
        logger.info("Mean top-1 acc: {:.2f}\n".format(np.mean(test_accs[-20:])))


def test(val_loader, model):
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            model.eval()

            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            precent = accuracy(outputs, targets, topk=(1,))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(precent[0].item(), inputs.shape[0])
            print(f"Test Loss: {losses.avg:.4f}. Accuracy: {top1.avg:.2f}.")

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    return losses.avg, top1.avg


if __name__ == "__main__":
    main()
