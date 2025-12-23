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

import models_mae
from util.load_data import MatchingImageDataset_mae


def get_args_parser():
    parser = argparse.ArgumentParser('MAE decoder pre-training', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Total training epochs')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations for larger effective batch size')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16_decoder4_640', type=str,
                        help='Model architecture name defined in models_mae.py')
    parser.add_argument('--input_size', default=640, type=int,
                        help='Image input size')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate')

    # Dataset parameters
    parser.add_argument('--data_path', default='./coco/train', type=str,
                        help='Training dataset path with an "images" folder')
    parser.add_argument('--val_data_path', default='./coco/val', type=str,
                        help='Validation dataset path with an "images" folder')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--output_dir', default='./output_pretrain',
                        help='Path where checkpoints and reconstructions are saved')
    parser.add_argument('--device', default='cuda:0',
                        help='Device to use for training')
    parser.add_argument('--resume', default='',
                        help='Path to MAE checkpoint for initialization')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Start epoch')
    parser.add_argument('--save_images_every', default=5, type=int,
                        help='Save reconstruction examples every N epochs (validation)')
    return parser


def denormalize(imgs, mean, std):
    return torch.clamp(imgs * std + mean, 0, 1)


def save_reconstructions(model, imgs, preds, output_dir, epoch, mean, std):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        recon = model.unpatchify(preds)
        recon = denormalize(recon, mean, std)
        imgs = denormalize(imgs, mean, std)

    save_dir = output_dir / 'recon'
    save_dir.mkdir(parents=True, exist_ok=True)

    grid = torch.cat([imgs, recon], dim=0)
    vutils.save_image(grid, save_dir / f'epoch_{epoch:03d}.png', nrow=imgs.shape[0])


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

    for step, imgs in enumerate(tqdm(data_loader)):
        imgs = imgs.to(device)
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


def train(args):
    device = torch.device(args.device)
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model = nn.DataParallel(model)

    param_groups = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scaler = amp.GradScaler()

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

    for epoch in range(args.start_epoch, args.epochs):
        print(f'Epoch {epoch} / {args.epochs - 1}')
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args.accum_iter)
        val_loss = evaluate(
            model, val_loader, device, image_mean, image_std,
            output_dir=args.output_dir, epoch=epoch,
            save_images=((epoch + 1) % args.save_images_every == 0)
        )

        print(f'Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}')

        if args.output_dir:
            ckpt_path = Path(args.output_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save({'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'scaler': scaler.state_dict()},
                       ckpt_path / f'checkpoint-{epoch}.pth')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
