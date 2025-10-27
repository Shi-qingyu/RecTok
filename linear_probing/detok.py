import argparse
import sys
import logging
import os
import time
import tqdm

from einops import rearrange
import torch
from torch import nn
import torchvision.transforms as transforms

from utils.builders import create_reconstruction_model, create_train_dataloader
from utils.loader import ListDataset, center_crop_arr
from utils.logger import MetricLogger, SmoothedValue
import utils.distributed as dist
import utils.misc as misc

logger = logging.getLogger("Linear Probing")


class LinearProbing(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int, last_layer_feature: bool = True):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.last_layer_feature = last_layer_feature

        if last_layer_feature:
            in_channels = model.encoder.width
            linear = nn.Linear(in_channels, num_classes)
            self.model.encoder.latent_head = linear
            self.freeze_model_without_head()
        else:
            in_channels = model.token_channels
            self.model.requires_grad_(False)
            linear = nn.Linear(in_channels, num_classes)
            self.cls_head = linear
            
        self.model.eval()
    
    def freeze_model_without_head(self):
        for name, param in self.model.named_parameters():
            if "encoder.latent_head" in name:
                logger.info("Training latent head only")
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.last_layer_feature:
            z = self.model.encoder(x)["z"]
            z = rearrange(z, "b (h w) c -> b c h w", h=self.model.seq_h)
        else:
            z = self.model.encode(x, sampling=False)["z_latents"]
            z = self.cls_head(z)
            z = rearrange(z, "b (h w) c -> b c h w", h=self.model.seq_h)

        logits = z.mean(dim=(-2, -1))
        return logits


def setup(args: argparse.Namespace):
    """setup distributed training, logging, and experiment configuration"""
    dist.enable_distributed()
    
    if args.exp_name is None:
        args.exp_name = f"linear_prob_{time.strftime('%Y%m%d_%H%M')}"

    base_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    args.log_dir = base_dir
    
    global_rank, world_size = dist.get_global_rank(), dist.get_world_size()
    args.world_size = world_size
    args.global_bsz = args.batch_size * world_size
    
    misc.fix_random_seeds(args.seed + global_rank)
    
    if global_rank == 0:
        os.makedirs(base_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(base_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
    
    logger.info(f"Distributed setup: rank {global_rank}/{world_size}")
    return global_rank == 0


def create_val_dataloader_with_labels(args):
    """Create validation dataloader that returns labels for classification"""
    transform_val = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    dataset_val = ListDataset(
        args.data_path.replace("train", "val"),
        data_list="data/val.txt",
        transform=transform_val,
        loader_name="img_loader",
        return_label=True,  # Enable labels for classification
        should_flip=False,
    )
    
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val,
        num_replicas=dist.get_world_size(),
        rank=dist.get_global_rank(),
        shuffle=False,
    )

    logger.info(f"Val dataset size: {len(dataset_val)}")

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.eval_bsz,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return data_loader_val


def evaluate(model, data_loader, device):
    """Evaluate the model on validation set"""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, disable=dist.get_global_rank() != 0):
            img = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            # Validate and clamp labels to prevent CUDA assertion errors
            labels = labels.long()  # Ensure labels are long integers
            labels = torch.clamp(labels, 0, 999)  # Clamp to valid range [0, 999] for ImageNet
            
            logits = model(img)
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == labels).sum().item()
            
            total_correct += correct
            total_samples += labels.size(0)
    
    # Aggregate across all processes
    total_correct_tensor = torch.tensor(total_correct, device=device)
    total_samples_tensor = torch.tensor(total_samples, device=device)
    
    if dist.get_world_size() > 1:
        torch.distributed.all_reduce(total_correct_tensor)
        torch.distributed.all_reduce(total_samples_tensor)
    
    accuracy = total_correct_tensor.item() / total_samples_tensor.item()
    return accuracy


def main(args: argparse.Namespace) -> int:
    is_main_process = setup(args)
    
    # First create model and move to device
    device = torch.device(f"cuda:{dist.get_global_rank()}")
    model = create_reconstruction_model(args)[0]
    logger.info(f"Loading checkpoint from {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    msg = model.load_state_dict(ckpt["model"], strict=False)
    logger.info(f"Checkpoint loaded: {msg}")
    linear_prob = LinearProbing(model, args.num_classes, args.last_layer_feature)
    
    # Then create dataloaders
    data_loader_train = create_train_dataloader(args)
    data_loader_val = create_val_dataloader_with_labels(args)
    
    # Move model to appropriate device
    device = torch.device(f"cuda:{dist.get_global_rank()}")
    linear_prob = linear_prob.to(device)
    
    # Wrap model with DistributedDataParallel
    if args.world_size > 1:
        linear_prob = torch.nn.parallel.DistributedDataParallel(
            linear_prob, 
            device_ids=[dist.get_global_rank()],
            find_unused_parameters=False
        )

    trainable_params = [p for p in linear_prob.parameters() if p.requires_grad]
    
    if args.lr is None:
        args.lr = args.blr * args.global_bsz / 256
    
    optimizer = torch.optim.Adam(
        trainable_params, 
        lr=args.lr if args.lr else args.blr,
        weight_decay=args.weight_decay
    )

    loss_fn = nn.CrossEntropyLoss()
    
    # Setup metric logger
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('acc', SmoothedValue(window_size=50, fmt='{value:.4f}'))

    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        if hasattr(data_loader_train.sampler, 'set_epoch'):
            data_loader_train.sampler.set_epoch(epoch)
            
        linear_prob.train()

        for step, batch in enumerate(data_loader_train):
            img = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            # Validate and clamp labels to prevent CUDA assertion errors
            labels = labels.long()  # Ensure labels are long integers
            
            # Debug: Check for invalid labels
            if step == 0 and epoch == 0:
                logger.info(f"Label range: min={labels.min().item()}, max={labels.max().item()}")
                logger.info(f"Number of classes: {args.num_classes}")
                invalid_labels = (labels < 0) | (labels >= args.num_classes)
                if invalid_labels.any():
                    logger.warning(f"Found {invalid_labels.sum().item()} invalid labels out of {labels.size(0)}")
                    logger.warning(f"Invalid label values: {labels[invalid_labels].unique()}")

            logits = linear_prob(img)   # [b, num_classes]
            loss = loss_fn(logits, labels)
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == labels).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Aggregate metrics across processes
            loss_reduced = dist.all_reduce_mean(loss)
            acc_reduced = dist.all_reduce_mean(acc)
            
            metric_logger.update(
                loss=loss_reduced,
                acc=acc_reduced,
                lr=optimizer.param_groups[0]["lr"]
            )
            
            if is_main_process and step % args.print_freq == 0:
                logger.info(
                    f"Epoch: [{epoch}/{args.epochs}] Step: [{step}/{len(data_loader_train)}] "
                    f"Loss: {loss_reduced:.4f} Acc: {acc_reduced:.4f} "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

        if epoch % args.eval_freq == 0:
            val_acc = evaluate(linear_prob, data_loader_val, device)
            if is_main_process:
                logger.info(f"Epoch {epoch} - Validation accuracy: {val_acc:.4f}")
    
    return 0


def get_args_parser():
    parser = argparse.ArgumentParser("Reconstruction model training", add_help=False)

    # basic training parameters
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per GPU for training")

    # model parameters
    parser.add_argument("--model", default="detok_BB", type=str)
    parser.add_argument("--token_channels", default=16, type=int)
    parser.add_argument("--pretrained_model_name_or_path", default="", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--last_layer_feature", action="store_true")
    parser.add_argument("--num_register_tokens", default=0, type=int)
    parser.add_argument("--aux_cls_token", action="store_true")
    parser.add_argument("--pooling_cls_token", action="store_true")
    parser.add_argument("--disable_kl", action="store_true")

    parser.add_argument("--mask_ratio", default=0.0, type=float)
    parser.add_argument("--gamma", default=0.0, type=float, help="noise standard deviation for training")

    parser.add_argument("--checkpoint_path", default=None, type=str)

    # logging parameters
    parser.add_argument("--output_dir", default="./work_dirs/linear_prob")
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=1)


    # evaluation parameters
    parser.add_argument("--eval_bsz", type=int, default=256)

    # optimization parameters
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--blr", type=float, default=1e-1)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_sched", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_rate", type=float, default=0.25)
    parser.add_argument("--ema_rate", default=0.999, type=float)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=3.0)
    parser.add_argument("--grad_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="Beta2 for AdamW optimizer")

    # dataset parameters
    parser.add_argument("--use_cached_tokens", action="store_true")
    parser.add_argument("--data_path", default="./data/imagenet/train", type=str)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--class_of_interest", default=[207, 360, 387, 974, 88, 979, 417, 279], type=int, nargs="+")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # system parameters
    parser.add_argument("--seed", default=0, type=int)

    # wandb parameters
    parser.add_argument("--project", default="linear_prob_DeTok", type=str)
    parser.add_argument("--entity", default="YOUR_WANDB_ENTITY", type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--enable_wandb", action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    exit_code = main(args)
    sys.exit(exit_code)