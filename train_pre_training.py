import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda import amp
from torch.cuda.amp import autocast
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
import timm.optim.optim_factory as optim_factory
import math

import models_mae
from util.load_data import MatchingImageDataset_mae


def get_args_parser():
    parser = argparse.ArgumentParser('MAE decoder pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=80, type=int, help='Total training epochs')
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations')

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
    parser.add_argument('--device', default='cuda:0', help='Device to use for training')
    parser.add_argument('--resume', default='mae_pretrain_vit_large.pth',
                        help='Path to checkpoint for initialization / resume')
    parser.add_argument('--start_epoch', default=0, type=int, help='Start epoch')

    return parser


def denormalize(imgs, mean, std):
    return torch.clamp(imgs * std + mean, 0, 1)

def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model if wrapped by DataParallel."""
    return model.module if isinstance(model, nn.DataParallel) else model


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
    """Warmup + half-cycle cosine decay."""
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
    """Create AdamW only for trainable params, keeping timm's weight-decay grouping."""
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)

    # keep only requires_grad params (important if you freeze encoder etc.)
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
    """
    Support two resume styles:
      1) Official MAE weights: checkpoint is usually a state_dict or dict without optimizer/scaler.
      2) Our training ckpt: dict with keys: model/optimizer/scaler/epoch
    """
    if not args.resume:
        return

    checkpoint = torch.load(args.resume, map_location='cpu')

    # detect model state dict
    state_dict = checkpoint.get('model', checkpoint)

    # remove incompatible keys if present (same as your original behavior)
    for k in ['pos_embed', 'decoder_pos_embed', 'cls_token']:
        if k in state_dict:
            print(f"Removing key from checkpoint: {k}, shape={state_dict[k].shape}")
            del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    print("Resume load:", args.resume)
    print("  Missing keys:", len(msg.missing_keys))
    print("  Unexpected keys:", len(msg.unexpected_keys))

    # restore optimizer/scaler/epoch if present
    if isinstance(checkpoint, dict) and 'optimizer' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Optimizer state restored.")
    if isinstance(checkpoint, dict) and 'scaler' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])
        print("AMP scaler state restored.")
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        args.start_epoch = int(checkpoint['epoch']) + 1
        print("Start epoch set to", args.start_epoch)


def evaluate(model, data_loader, device, mean, std, output_dir=None, epoch=None, save_images=False):
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for step, imgs in enumerate(data_loader):
            imgs = imgs.to(device)
            loss, pred = model(imgs)
            total_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)

            if save_images and step == 0 and output_dir is not None and epoch is not None:
                save_reconstructions(model, imgs, pred, output_dir, epoch, mean, std)

    avg_loss = total_loss / max(count, 1)
    return avg_loss


def train_one_epoch(model, data_loader, optimizer, scaler, device, accum_iter):
    model.train()
    total_loss = 0.0
    count = 0

    optimizer.zero_grad(set_to_none=True)

    for step, imgs in enumerate(tqdm(data_loader)):
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

    return total_loss / max(count, 1)

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
    device = torch.device(args.device)
    # --- model ---
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss).to(device)

    # DataParallel only when multi-GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # --- optimizer & scaler ---
    optimizer = build_optimizer(model, args)
    scaler = amp.GradScaler()

    # --- resume ---
    load_checkpoint(model, optimizer, scaler, args)

    # --- data ---
    data_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = MatchingImageDataset_mae(root_dir=args.data_path, transform=data_transform)
    val_dataset = MatchingImageDataset_mae(root_dir=args.val_data_path, transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=args.pin_mem)

    image_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    image_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


    # --- training loop ---
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args)
        print(f'Epoch {epoch} / {args.epochs - 1} | LR {lr:.6f}')

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args.accum_iter)

        val_loss = evaluate(
            model, val_loader, device, image_mean, image_std,
            output_dir=args.output_dir, epoch=epoch,
            save_images=((epoch + 1) % args.save_images_every == 0)
        )

        print(f'Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}')

        # save checkpoint
        if args.output_dir and (
            ((epoch + 1) % args.save_ckpt_every == 0) or ((epoch + 1) == args.epochs)
        ):
            save_checkpoint(model, optimizer, scaler, epoch, args.output_dir)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)