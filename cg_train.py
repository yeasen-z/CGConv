"""Train CIFAR-10 ResNet and CGConv models."""

import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models
from cgToolkit import CGConv2d, do_sponge_loss
from utils import AverageMeter, RecorderMeter, convert_secs2time, time_string


MODEL_NAMES = sorted(
    name
    for name, value in models.__dict__.items()
    if (name.startswith("resnet") or name.startswith("cgc_resnet"))
    and callable(value)
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a CIFAR-10 network with optional CGConv regularisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", default="./data", help="dataset directory")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--arch", default="cgc_resnet56", choices=MODEL_NAMES)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--optimizer", default="SGD", choices=["SGD"])
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--decay", type=float, default=5e-4)
    parser.add_argument("--schedule", type=int, nargs="+", default=[80, 120])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.1])
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--save_path", default="./save")
    parser.add_argument("--resume", default="", help="checkpoint to resume/evaluate")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--model_only", action="store_true", help="save weights without optimizer state"
    )

    parser.add_argument("--ngpu", type=int, default=1, help="0 selects CPU")
    parser.add_argument("--gpu_id", type=int, default=0, help="first CUDA device")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--manualSeed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="prefer reproducibility over cuDNN benchmark performance",
    )

    parser.add_argument("--sponge_sigma", type=float, default=1e-4)
    parser.add_argument("--sponge_lb", type=float, default=0.05)
    parser.add_argument(
        "--sponge_criterion", default="l0", choices=["l0", "l2"]
    )
    parser.add_argument(
        "--cgc",
        action="store_true",
        help="add the CGConv activation sparsity regulariser",
    )
    return parser


def configure_runtime(args):
    if len(args.schedule) != len(args.gammas):
        raise ValueError("--schedule and --gammas must have equal lengths")
    if args.ngpu < 0:
        raise ValueError("--ngpu must be non-negative")

    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    use_cuda = args.ngpu > 0 and torch.cuda.is_available()
    if use_cuda:
        if args.gpu_id + args.ngpu > torch.cuda.device_count():
            raise ValueError(
                f"requested CUDA devices {args.gpu_id}.."
                f"{args.gpu_id + args.ngpu - 1}, but only "
                f"{torch.cuda.device_count()} are visible"
            )
        torch.cuda.manual_seed_all(args.manualSeed)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    cudnn.benchmark = use_cuda and not args.deterministic
    cudnn.deterministic = use_cuda and args.deterministic
    args.use_cuda = use_cuda
    return device


def build_loaders(args):
    mean = [value / 255 for value in (125.3, 123.0, 113.9)]
    std = [value / 255 for value in (63.0, 62.1, 66.7)]
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    train_data = datasets.CIFAR10(
        args.data_path, train=True, transform=train_transform, download=True
    )
    test_data = datasets.CIFAR10(
        args.data_path, train=False, transform=test_transform, download=True
    )
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": args.use_cuda,
        "persistent_workers": args.workers > 0,
    }
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True,
        **loader_options,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        **loader_options,
    )
    return train_loader, test_loader


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def build_model(args, device):
    model = models.__dict__[args.arch](num_classes=10)
    model = model.to(device)
    if args.use_cuda and args.ngpu > 1:
        device_ids = list(range(args.gpu_id, args.gpu_id + args.ngpu))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    return model


def restore_recorder(saved_recorder, total_epochs):
    recorder = RecorderMeter(total_epochs)
    if saved_recorder is None:
        return recorder
    count = min(saved_recorder.current_epoch, total_epochs)
    recorder.epoch_losses[:count] = saved_recorder.epoch_losses[:count]
    recorder.epoch_accuracy[:count] = saved_recorder.epoch_accuracy[:count]
    recorder.current_epoch = count
    return recorder


def load_checkpoint(path, model, optimizer, recorder, device, total_epochs):
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    unwrap_model(model).load_state_dict(state_dict)

    start_epoch = int(checkpoint.get("epoch", 0))
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    recorder = restore_recorder(checkpoint.get("recorder"), total_epochs)
    best_acc = float(checkpoint.get("best_acc", recorder.max_accuracy(False)))
    return start_epoch, recorder, best_acc


def main(cli_args=None):
    args = build_parser().parse_args(cli_args)
    device = configure_runtime(args)
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    log_path = save_path / f"log_seed_{args.manualSeed}.txt"
    with log_path.open("a" if args.resume else "w", encoding="utf-8") as log:
        print_log(f"save path : {save_path}", log)
        print_log(vars(args), log)
        print_log(f"device : {device}", log)
        print_log(f"python version : {sys.version.replace(os.linesep, ' ')}", log)
        print_log(f"torch version : {torch.__version__}", log)

        train_loader, test_loader = build_loaders(args)
        model = build_model(args, device)
        print_log(f"=> network:\n{model}", log)

        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.decay,
            nesterov=True,
        )
        recorder = RecorderMeter(args.epochs)
        start_epoch = 0
        best_acc = 0.0

        if args.resume:
            start_epoch, recorder, best_acc = load_checkpoint(
                args.resume, model, optimizer, recorder, device, args.epochs
            )
            print_log(f"=> resumed '{args.resume}' at epoch {start_epoch}", log)

        if args.evaluate:
            validate(test_loader, model, criterion, device, log)
            return

        cgc_layers = [
            module
            for module in unwrap_model(model).modules()
            if isinstance(module, CGConv2d)
        ]
        if args.cgc and not cgc_layers:
            raise ValueError("--cgc requires an architecture containing CGConv2d")
        print_log(f"CGConv layers : {len(cgc_layers)}", log)

        epoch_time = AverageMeter()
        for epoch in range(start_epoch, args.epochs):
            current_lr = adjust_learning_rate(
                optimizer, epoch, args.learning_rate, args.gammas, args.schedule
            )
            hours, minutes, seconds = convert_secs2time(
                epoch_time.avg * (args.epochs - epoch)
            )
            print_log(
                f"\n==>>{time_string()} [Epoch={epoch:03d}/{args.epochs:03d}] "
                f"[Need: {hours:02d}:{minutes:02d}:{seconds:02d}] "
                f"[LR={current_lr:.6f}] [Best={best_acc:.2f}]",
                log,
            )

            started = time.time()
            train_acc, train_loss = train(
                train_loader,
                model,
                criterion,
                optimizer,
                epoch,
                device,
                cgc_layers,
                args,
                log,
            )
            val_acc, _, val_loss = validate(
                test_loader, model, criterion, device, log
            )
            is_best = val_acc >= best_acc
            best_acc = max(best_acc, val_acc)
            recorder.update(epoch, train_loss, train_acc, val_loss, val_acc)

            if args.model_only:
                checkpoint_state = {
                    "arch": args.arch,
                    "state_dict": unwrap_model(model).state_dict(),
                }
            else:
                checkpoint_state = {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": unwrap_model(model).state_dict(),
                    "best_acc": best_acc,
                    "recorder": recorder,
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                }
            save_checkpoint(
                checkpoint_state,
                is_best,
                save_path,
                "checkpoint.pth.tar",
                log,
            )
            recorder.plot_curve(str(save_path / "curve.png"))
            accuracy_logger(save_path, epoch, train_acc, val_acc)
            epoch_time.update(time.time() - started)


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    device,
    cgc_layers,
    args,
    log,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()

    end = time.time()
    for batch_index, (inputs, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs = inputs.to(device, non_blocking=args.use_cuda)
        target = target.to(device, non_blocking=args.use_cuda)
        optimizer.zero_grad(set_to_none=True)

        if args.cgc:
            sponge_loss, output, _ = do_sponge_loss(
                model, inputs, cgc_layers, args
            )
            loss = criterion(output, target) + sponge_loss
        else:
            output = model(inputs)
            loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_index % args.print_freq == 0:
            print_log(
                f"  Epoch: [{epoch:03d}][{batch_index:03d}/{len(train_loader):03d}] "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Loss {losses.val:.4f} ({losses.avg:.4f}) "
                f"Prec@1 {top1.val:.3f} ({top1.avg:.3f}) "
                f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})",
                log,
            )

    print_log(
        f"  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} "
        f"Error@1 {100 - top1.avg:.3f}",
        log,
    )
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, device, log, summary_output=False):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    output_summary = []

    with torch.no_grad():
        for inputs, target in val_loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            target = target.to(device, non_blocking=device.type == "cuda")
            output = model(inputs)
            loss = criterion(output, target)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))
            if summary_output:
                output_summary.append(output.argmax(dim=1).cpu().numpy())

    print_log(
        f"  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} "
        f"Error@1 {100 - top1.avg:.3f}",
        log,
    )
    if summary_output:
        return top1.avg, top5.avg, losses.avg, np.concatenate(output_summary)
    return top1.avg, top5.avg, losses.avg


def print_log(message, log) -> None:
    print(message)
    log.write(f"{message}\n")
    log.flush()


def save_checkpoint(state, is_best, save_path, filename, log) -> None:
    filename = Path(save_path) / filename
    torch.save(state, filename)
    if is_best:
        best_name = Path(save_path) / "model_best.pth.tar"
        shutil.copyfile(filename, best_name)
        print_log(f"=> updated best model: {best_name}", log)


def adjust_learning_rate(optimizer, epoch, initial_lr, gammas, schedule):
    learning_rate = initial_lr
    for gamma, step in zip(gammas, schedule):
        if epoch < step:
            break
        learning_rate *= gamma
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    return learning_rate


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        max_k = max(topk)
        batch_size = target.size(0)
        predictions = output.topk(max_k, dim=1).indices.t()
        correct = predictions.eq(target.view(1, -1).expand_as(predictions))
        return [
            float(correct[:k].reshape(-1).float().sum() * (100.0 / batch_size))
            for k in topk
        ]


def accuracy_logger(base_dir, epoch, train_accuracy, test_accuracy) -> None:
    path = Path(base_dir) / "accuracy.txt"
    if not path.exists():
        path.write_text("epochs train test\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as log:
        log.write(f"{epoch} {train_accuracy} {test_accuracy}\n")


if __name__ == "__main__":
    main()
