import os
import json
import yaml
import argparse
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from dataset import LandslideDataset
from SAM2UNet import SAM2UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and test SAM2-UNet on Landslide Dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Split to evaluate")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save predictions and metrics")
    parser.add_argument("--save_masks", type=lambda x: (str(x).lower() == 'true'), default=True, help="Save prediction PNGs")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization probability threshold")
    parser.add_argument("--batch_size", type=int, default=1, help="Evaluation batch size")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_metrics(pred_prob: np.ndarray, gt_binary: np.ndarray, threshold: float = 0.5):
    """
    Computes standard segmentation metrics:
    IoU, Dice/F1, Precision, Recall, MAE.
    """
    pred_bin = (pred_prob >= threshold).astype(np.float32)
    gt = gt_binary.astype(np.float32)

    intersection = (pred_bin * gt).sum()
    union = (pred_bin + gt).clip(0, 1).sum()
    tp = intersection
    fp = (pred_bin * (1 - gt)).sum()
    fn = ((1 - pred_bin) * gt).sum()

    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2.0 * intersection + 1e-7) / (pred_bin.sum() + gt.sum() + 1e-7)
    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)
    mae = np.mean(np.abs(pred_prob - gt))

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "mae": float(mae),
    }


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    # Determine save directory
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        # Default save inside the run folder if checkpoint is in runs/
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        run_parent = os.path.dirname(ckpt_dir)
        save_dir = os.path.join(run_parent, f"{args.split}_results")

    pred_dir = os.path.join(save_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    print(f"Results and predictions will be saved to: {save_dir}")

    # Dataset & Dataloader
    ds_cfg = config["dataset"]
    modalities = ds_cfg.get("modalities", ["IMAGE", "DTM"])
    data_dir = ds_cfg.get("data_dir", "datasets/landslide")
    blacklist_path = ds_cfg.get("blacklist_path", "datasets/landslide/black_list.txt")
    target_size = ds_cfg.get("size", 352)

    dataset = LandslideDataset(
        data_dir=data_dir,
        split=args.split,
        size=target_size,
        modalities=modalities,
        blacklist_path=blacklist_path,
        mode="test",
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Evaluating {len(dataset)} samples from [{args.split}] split with modalities: {modalities}")

    # Model initialization
    topo_in_chans = sum(1 for m in modalities if m != "IMAGE")
    m_cfg = config["model"]
    topo_backbone = m_cfg.get("topo_backbone", "convnext_tiny")
    use_kan = m_cfg.get("use_kan", True)

    model = SAM2UNet(
        checkpoint_path=None,
        topo_in_chans=max(1, topo_in_chans),
        topo_backbone=topo_backbone,
        pretrained_topo=False,
        use_kan=use_kan,
    )
    print(f"Loading checkpoint weights from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    all_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating [{args.split}]"):
            images = batch["image"].to(device)
            labels = batch["label"]
            names = batch["name"]

            # Forward pass: take main output head
            preds, _, _ = model(images)

            # Upsample prediction back to original ground truth resolution (512, 512)
            orig_h = labels.shape[2]
            orig_w = labels.shape[3]
            preds_upsampled = F.interpolate(preds, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            pred_probs = torch.sigmoid(preds_upsampled).cpu().numpy()

            for i in range(len(names)):
                name = names[i]
                prob = pred_probs[i, 0]  # Shape (H, W)
                gt = labels[i, 0].numpy()  # Shape (H, W), values {0, 1}

                m = calculate_metrics(prob, gt, threshold=args.threshold)
                all_metrics.append(m)

                if args.save_masks:
                    # Save probability mask scaled to [0, 255]
                    out_img = (prob * 255.0).clip(0, 255).astype(np.uint8)
                    save_path = os.path.join(pred_dir, f"{name}.png")
                    Image.fromarray(out_img).save(save_path)

    # Compute mean metrics across all samples
    avg_metrics = {
        key: float(np.mean([m[key] for m in all_metrics]))
        for key in all_metrics[0].keys()
    }

    # Print summary
    print("\n" + "=" * 50)
    print(f"EVALUATION METRICS SUMMARY [{args.split.upper()}] (N={len(all_metrics)})")
    print("=" * 50)
    for k, v in avg_metrics.items():
        print(f"  {k.upper():<12}: {v:.4f}")
    print("=" * 50)

    # Save metrics to text and JSON
    metrics_txt = os.path.join(save_dir, "metrics.txt")
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"Evaluation on Split: {args.split}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Modalities: {modalities}\n")
        f.write(f"Threshold: {args.threshold}\n\n")
        for k, v in avg_metrics.items():
            f.write(f"{k.upper()}: {v:.4f}\n")

    metrics_json = os.path.join(save_dir, "metrics.json")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "checkpoint": args.checkpoint,
                "modalities": modalities,
                "threshold": args.threshold,
                "metrics": avg_metrics,
            },
            f,
            indent=2,
        )

    print(f"Metrics saved to:\n  - {metrics_txt}\n  - {metrics_json}")


if __name__ == "__main__":
    main()
