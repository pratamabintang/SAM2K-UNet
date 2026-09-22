import os
import sys
import time
import yaml
import random
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import LandslideDataset
from SAM2UNet import SAM2UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAM2-UNet with Dual-Branch Topography Architecture")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML configuration file")
    # Command-line overrides
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--hiera_path", type=str, default=None, help="Override SAM2 pretrained checkpoint path")
    parser.add_argument("--topo_backbone", type=str, default=None, help="Override topography backbone (timm)")
    parser.add_argument("--use_kan", type=lambda x: (str(x).lower() == 'true'), default=None, help="Enable/disable KAN decoder")
    parser.add_argument("--run_name", type=str, default=None, help="Override experiment/run name")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 1024):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("LandslideSAM2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Structure loss: Weighted Binary Cross-Entropy + Weighted IoU."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def compute_iou(pred: torch.Tensor, mask: torch.Tensor, threshold: float = 0.5) -> float:
    """Computes binary IoU score between prediction and ground truth."""
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    inter = (pred_bin * mask).sum().item()
    union = (pred_bin + mask).clamp(0, 1).sum().item()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / (union + 1e-7)


def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Runs evaluation on validation split."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            target = batch["label"].to(device)
            out, out1, out2 = model(image)
            loss0 = structure_loss(out, target)
            loss1 = structure_loss(out1, target)
            loss2 = structure_loss(out2, target)
            loss = loss0 + loss1 + loss2

            total_loss += loss.item()
            total_iou += compute_iou(out, target)
            count += 1

    return {
        "val_loss": total_loss / max(1, count),
        "val_iou": total_iou / max(1, count),
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    # Apply CLI overrides if present
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["dataset"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.hiera_path is not None:
        config["model"]["hiera_path"] = args.hiera_path
    if args.topo_backbone is not None:
        config["model"]["topo_backbone"] = args.topo_backbone
    if args.use_kan is not None:
        config["model"]["use_kan"] = args.use_kan
    if args.run_name is not None:
        config["experiment"]["name"] = args.run_name

    # Set seeds
    seed = config["experiment"].get("seed", 1024)
    seed_everything(seed)

    # Create run output directory under runs/
    runs_dir = config["experiment"].get("runs_dir", "runs")
    exp_name = config["experiment"].get("name", "sam2_unet")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(runs_dir, f"{exp_name}_{timestamp}")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Save exact copy of configuration in run folder for reproducibility
    saved_config_path = os.path.join(run_dir, "config.yaml")
    with open(saved_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Setup logger
    logger = setup_logger(os.path.join(run_dir, "train.log"))
    logger.info(f"=== Starting Training Session ===")
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Run Output Directory: {run_dir}")
    logger.info(f"Active Config saved to: {saved_config_path}")

    # Prepare dataset & data loaders
    ds_cfg = config["dataset"]
    modalities = ds_cfg.get("modalities", ["IMAGE", "DTM"])
    data_dir = ds_cfg.get("data_dir", "datasets/landslide")
    blacklist_path = ds_cfg.get("blacklist_path", "datasets/landslide/black_list.txt")
    target_size = ds_cfg.get("size", 352)
    batch_size = ds_cfg.get("batch_size", 12)
    num_workers = ds_cfg.get("num_workers", 4)

    logger.info(f"Selected Modalities: {modalities}")
    logger.info(f"Spatial Target Size: {target_size}x{target_size}, Batch Size: {batch_size}")

    train_dataset = LandslideDataset(
        data_dir=data_dir,
        split="train",
        size=target_size,
        modalities=modalities,
        blacklist_path=blacklist_path,
        mode="train",
    )
    val_dataset = LandslideDataset(
        data_dir=data_dir,
        split="val",
        size=target_size,
        modalities=modalities,
        blacklist_path=blacklist_path,
        mode="val",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Compute Device: {device}")

    # num_workers fallback for Windows / CPU if needed
    train_workers = num_workers if (device.type == "cuda" and os.name != "nt") else min(num_workers, 2)
    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        pin_memory=pin_memory,
        drop_last=True if len(train_dataset) > batch_size else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_workers,
        pin_memory=pin_memory,
    )

    logger.info(f"Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")

    # Determine topography input channels
    topo_in_chans = sum(1 for m in modalities if m != "IMAGE")
    logger.info(f"Derived Topography Input Channels: {topo_in_chans}")

    # Build Model
    m_cfg = config["model"]
    hiera_path = m_cfg.get("hiera_path", None)
    topo_backbone = m_cfg.get("topo_backbone", "convnext_tiny")
    pretrained_topo = m_cfg.get("pretrained_topo", True)
    use_kan = m_cfg.get("use_kan", True)

    logger.info(f"Initializing SAM2UNet (Topo: {topo_backbone}, Pretrained: {pretrained_topo}, KAN Decoder: {use_kan})...")
    model = SAM2UNet(
        checkpoint_path=hiera_path if (hiera_path and os.path.exists(hiera_path)) else None,
        topo_in_chans=max(1, topo_in_chans),
        topo_backbone=topo_backbone,
        pretrained_topo=pretrained_topo,
        use_kan=use_kan,
    )
    model.to(device)

    # Optimizer & Scheduler
    t_cfg = config["training"]
    epochs = t_cfg.get("epochs", 20)
    lr = t_cfg.get("lr", 1.0e-3)
    weight_decay = t_cfg.get("weight_decay", 5.0e-4)
    min_lr = t_cfg.get("min_lr", 1.0e-7)
    save_interval = t_cfg.get("save_interval", 5)
    eval_interval = t_cfg.get("eval_interval", 1)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Total Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = opt.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)

    best_val_iou = -1.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0

        for i, batch in enumerate(train_loader):
            image = batch["image"].to(device)
            target = batch["label"].to(device)

            optimizer.zero_grad()
            out, out1, out2 = model(image)
            loss0 = structure_loss(out, target)
            loss1 = structure_loss(out1, target)
            loss2 = structure_loss(out2, target)
            loss = loss0 + loss1 + loss2

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step_count += 1

            if (i + 1) % 20 == 0 or (i + 1) == len(train_loader):
                logger.info(
                    f"Epoch [{epoch}/{epochs}] Step [{i+1}/{len(train_loader)}] - "
                    f"Batch Loss: {loss.item():.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, step_count)
        logger.info(f"Epoch [{epoch}/{epochs}] Finished - Average Train Loss: {avg_train_loss:.4f}")

        # Evaluation phase
        if epoch % eval_interval == 0:
            val_metrics = evaluate(model, val_loader, device)
            val_loss = val_metrics["val_loss"]
            val_iou = val_metrics["val_iou"]
            logger.info(f"Validation - Loss: {val_loss:.4f} | IoU: {val_iou:.4f}")

            # Save best checkpoint
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_ckpt_path = os.path.join(checkpoints_dir, "best.pth")
                torch.save(model.state_dict(), best_ckpt_path)
                logger.info(f"[*] New best validation IoU ({best_val_iou:.4f})! Saved to {best_ckpt_path}")

        # Save latest checkpoint
        latest_ckpt_path = os.path.join(checkpoints_dir, "latest.pth")
        torch.save(model.state_dict(), latest_ckpt_path)

        # Save periodic snapshot
        if epoch % save_interval == 0 or epoch == epochs:
            snap_path = os.path.join(checkpoints_dir, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), snap_path)
            logger.info(f"Saved snapshot to {snap_path}")

    total_time = (time.time() - start_time) / 60
    logger.info(f"=== Training Completed in {total_time:.2f} minutes ===")
    logger.info(f"Best Validation IoU: {best_val_iou:.4f}")
    logger.info(f"Artifacts and checkpoints are preserved at: {run_dir}")


if __name__ == "__main__":
    main()