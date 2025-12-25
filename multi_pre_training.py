import argparse
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda import amp
from torch.cuda.amp import autocast
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
import timm.optim.optim_factory as optim_factory

import models_mae
from util.load_data import MatchingImageDataset_mae


def get_args_parser():
    parser = argparse.ArgumentParser('MAE decoder pre-training (DDP)', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=80, type=int, help='Total training epochs')
    parser.add_argument('--accum_iter', default=2, type=int, help='Accumulate gradient iterations')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16_decoder4_512', type=str,
                        help='Model architecture name defined in models_mae.py')
    parser.add_argument('--input_size', default=512, type=int, help='Image input size')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Save config
    parser.add_argument('--save_ckpt_every', default=5, type=int, help='Save checkpoint every N epochs')
    parser.add_argument('--save_images_every', default=5, type=int,
                        help='Save reconstruction examples every N epochs (validation)')
    parser.add_argument('--output_dir', default='./output_pretrain',
                        help='Path where checkpoints and reconstructions are saved')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for schedulers that hit 0')
    parser.add_argument('--blr', default=1e-3, type=float,
                        help='Base LR for a reference effective batch size (see --lr_base_batch)' \
                        'absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--lr_base_batch', default=256, type=int,
                        help='Reference effective batch size for --blr scaling')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')

    # Dataset parameters
    parser.add_argument('--data_path', default='./COCO/train2017', type=str, help='Training dataset path')
    parser.add_argument('--val_data_path', default='./COCO/val2017', type=str, help='Validation dataset path')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Runtime
    parser.add_argument('--device', default='cuda', help='Device base (overridden by local_rank for DDP)')
    parser.add_argument('--resume', default='mae_pretrain_vit_large.pth',
                        help='Path to checkpoint for initialization / resume')
    parser.add_argument('--start_epoch', default=0, type=int, help='Start epoch')

    # Distributed
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', help='Distributed backend')

    return parser


def denormalize(imgs, mean, std):
    return torch.clamp(imgs * std + mean, 0, 1)


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


def is_main_process(args) -> bool:
    return args.rank == 0


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False
        args.gpu = 0
        return

    args.distributed = args.world_size > 1
    args.gpu = args.local_rank

    torch.cuda.set_device(args.gpu)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def save_reconstructions(model, imgs, preds, output_dir, epoch, mean, std):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    m = _unwrap_model(model)

    with torch.no_grad():
        recon = m.unpatchify(preds)
        recon = denormalize(recon, mean, std)
        imgs = denormalize(imgs, mean, std)

    save_dir = output_dir / 'recon'
    save_dir.mkdir(parents=True, exist_ok=True)

    grid = torch.cat([imgs, recon], dim=0)
    vutils.save_image(grid, save_dir / f'epoch_{epoch:03d}.png', nrow=imgs.shape[0])


def adjust_learning_rate(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * (epoch + 1) / args.warmup_epochs
    else:
        t = (epoch - args.warmup_epochs) / max(1, (args.epochs - args.warmup_epochs))
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1. + math.cos(math.pi * t))

    for param_group in optimizer.param_groups:
        scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = lr * scale
    return lr


def build_optimizer(model, args):
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)

    filtered_groups = []
    for g in param_groups:
        params = [p for p in g["params"] if p.requires_grad]
        if len(params) > 0:
            ng = dict(g)
            ng["params"] = params
            filtered_groups.append(ng)

    optimizer = torch.optim.AdamW(filtered_groups, lr=args.lr, betas=(0.9, 0.95))
    return optimizer


def load_checkpoint(model, optimizer, scaler, args):
    if not args.resume:
        return

    checkpoint = torch.load(args.resume, map_location='cpu')

    target_model = _unwrap_model(model)
    state_dict = checkpoint.get('model', checkpoint)

    for k in ['pos_embed', 'decoder_pos_embed', 'cls_token']:
        if k in state_dict:
            print(f"Removing key from checkpoint: {k}, shape={state_dict[k].shape}")
            del state_dict[k]

    msg = target_model.load_state_dict(state_dict, strict=False)
    if is_main_process(args):
        print("Resume load:", args.resume)
        print("  Missing keys:", len(msg.missing_keys))
        print("  Unexpected keys:", len(msg.unexpected_keys))

    if isinstance(checkpoint, dict) and 'optimizer' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        if is_main_process(args):
            print("Optimizer state restored.")
    if isinstance(checkpoint, dict) and 'scaler' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])
        if is_main_process(args):
            print("AMP scaler state restored.")
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        args.start_epoch = int(checkpoint['epoch']) + 1
        if is_main_process(args):
            print("Start epoch set to", args.start_epoch)


def reduce_loss(total_loss, count, device):
    total = torch.tensor([total_loss, count], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total)
    total_loss, total_count = total.tolist()
    return total_loss / max(total_count, 1.0)


def evaluate(model, data_loader, device, mean, std, output_dir=None, epoch=None, save_images=False, args=None):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for step, imgs in enumerate(data_loader):
            imgs = imgs.to(device, non_blocking=True)
            loss, pred = model(imgs)
            total_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)

            if save_images and step == 0 and output_dir is not None and epoch is not None and is_main_process(args):
                save_reconstructions(model, imgs, pred, output_dir, epoch, mean, std)

    return total_loss, count


def train_one_epoch(model, data_loader, optimizer, scaler, device, accum_iter, disable_progress=False):
    model.train()
    total_loss = 0.0
    count = 0

    optimizer.zero_grad(set_to_none=True)

    for step, imgs in enumerate(tqdm(data_loader, disable=disable_progress)):
        imgs = imgs.to(device, non_blocking=True)
        with autocast():
            loss, _ = model(imgs)

        loss_value = loss.item()
        total_loss += loss_value * imgs.size(0)
        count += imgs.size(0)

        loss = loss / accum_iter
        scaler.scale(loss).backward()

        if (step + 1) % accum_iter == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    return total_loss, count


def save_checkpoint(model, optimizer, scaler, epoch, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = _unwrap_model(model).state_dict()
    torch.save(
        {
            'model': state,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': scaler.state_dict(),
        },
        output_dir / f'checkpoint-{epoch}.pth'
    )


def train(args):
    init_distributed_mode(args)

    eff_batch = args.batch_size * args.world_size * args.accum_iter
    args.lr = args.blr * eff_batch / args.lr_base_batch
    if is_main_process(args):
        print(f"Effective batch={eff_batch}, scaled lr={args.lr:.6g} (blr={args.blr}, base_batch={args.lr_base_batch})")

    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else args.device)

    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss).to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    optimizer = build_optimizer(model, args)
    scaler = amp.GradScaler()

    load_checkpoint(model, optimizer, scaler, args)

    data_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MatchingImageDataset_mae(root_dir=args.data_path, transform=data_transform)
    val_dataset = MatchingImageDataset_mae(root_dir=args.val_data_path, transform=data_transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True) \
        if args.distributed else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False) \
        if args.distributed else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None,
                              num_workers=args.num_workers, pin_memory=args.pin_mem, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_mem, sampler=val_sampler)

    image_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    image_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    if args.output_dir and is_main_process(args):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if args.distributed and val_sampler is not None and hasattr(val_sampler, "set_epoch"):
            val_sampler.set_epoch(epoch)

        lr = adjust_learning_rate(optimizer, epoch, args)
        if is_main_process(args):
            print(f'Epoch {epoch} / {args.epochs - 1} | LR {lr:.6f}')

        train_total, train_count = train_one_epoch(
            model, train_loader, optimizer, scaler, device, args.accum_iter,
            disable_progress=not is_main_process(args)
        )
        train_loss = reduce_loss(train_total, train_count, device)

        val_total, val_count = evaluate(
            model, val_loader, device, image_mean, image_std,
            output_dir=args.output_dir, epoch=epoch,
            save_images=((epoch + 1) % args.save_images_every == 0),
            args=args
        )
        val_loss = reduce_loss(val_total, val_count, device)

        if is_main_process(args):
            print(f'Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}')

        if args.output_dir and is_main_process(args) and (
            ((epoch + 1) % args.save_ckpt_every == 0) or ((epoch + 1) == args.epochs)
        ):
            save_checkpoint(model, optimizer, scaler, epoch, args.output_dir)

        if args.distributed:
            dist.barrier()

    cleanup_distributed()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train(args)
