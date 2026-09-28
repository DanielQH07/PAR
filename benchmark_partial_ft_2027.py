"""Measure last-two-block fine-tuning cost on the current GPU."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from extract_2027_backbone_features import ChallengeImages, MEAN, STD


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=('dinov3_vitb16', 'clip_b16', 'siglip2_b16'), required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--train-csv', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(605)
    processor_path = args.weights / 'preprocessor_config.json'
    processor = json.loads(processor_path.read_text()) if processor_path.exists() else {}
    mean, std = processor.get('image_mean', MEAN), processor.get('image_std', STD)
    data = ChallengeImages(args.train_csv, args.image_root, mean, std)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)
    full = AutoModel.from_pretrained(args.weights, local_files_only=True)
    if args.backbone == 'dinov3_vitb16':
        vision = full
        blocks = vision.layer
        tail = vision.norm
    else:
        vision = full.vision_model
        del full
        blocks = vision.encoder.layers
        tail = vision.post_layernorm
    vision.requires_grad_(False)
    for block in blocks[-2:]:
        block.requires_grad_(True)
    tail.requires_grad_(True)
    if args.backbone == 'siglip2_b16':
        vision.head.requires_grad_(True)
    dimension = vision.config.hidden_size if args.backbone == 'dinov3_vitb16' else vision.config.hidden_size
    head = torch.nn.Sequential(torch.nn.LayerNorm(dimension),
                               torch.nn.Linear(dimension, 40))
    device = torch.device('cuda')
    vision.to(device).train()
    head.to(device).train()
    positive = np.asarray([row[1:] for row in data.rows], dtype=np.float32).sum(axis=0)
    pos_weight = torch.tensor((len(data) - positive) / np.maximum(positive, 1),
                              device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in vision.parameters() if p.requires_grad], 'lr': 1e-5},
        {'params': head.parameters(), 'lr': 1e-3},
    ], weight_decay=0)
    scaler = torch.amp.GradScaler('cuda')
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = None
    count = 0
    images_seen = 0
    for step, (images, labels) in enumerate(loader):
        if step >= args.steps:
            break
        if step == min(10, args.steps // 5):
            torch.cuda.synchronize()
            started = time.monotonic()
            count = 0
        with torch.autocast('cuda'):
            output = vision(pixel_values=images.to(device, non_blocking=True))
            logits = head(output.pooler_output)
            loss = criterion(logits, labels.to(device, non_blocking=True))
            loss = loss / args.accumulation
        scaler.scale(loss).backward()
        if (step + 1) % args.accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        images_seen += len(images)
        if started is not None:
            count += 1
        if (step + 1) % 20 == 0:
            print(f'step {step + 1}/{args.steps}', flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    seconds_per_batch = elapsed / count
    result = {
        'backbone': args.backbone, 'batch_size': args.batch_size,
        'accumulation': args.accumulation,
        'effective_batch_size': args.batch_size * args.accumulation,
        'steps': args.steps, 'timed_steps': count,
        'seconds_per_batch': seconds_per_batch,
        'images_per_second': args.batch_size / seconds_per_batch,
        'estimated_train_epoch_seconds': len(data) / args.batch_size * seconds_per_batch,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'trainable_backbone_parameters': sum(p.numel() for p in vision.parameters() if p.requires_grad),
        'input_geometry': 'aspect-preserving resize + pad 224x224',
        'augmentation': 'none: timing benchmark only',
        'validation_included': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
