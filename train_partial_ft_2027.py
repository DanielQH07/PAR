"""Fine-tune the last two vision blocks for UPAR 2027 Track 1."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from evaluate_cached_head import score
from extract_2027_backbone_features import ChallengeImages, MEAN, STD


def make_model(kind, weights, device):
    full = AutoModel.from_pretrained(weights, local_files_only=True)
    if kind == 'dinov3_vitb16':
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
    if kind == 'siglip2_b16':
        vision.head.requires_grad_(True)
    dimension = vision.config.hidden_size
    head = torch.nn.Sequential(torch.nn.LayerNorm(dimension),
                               torch.nn.Linear(dimension, 40))
    return vision.to(device), head.to(device)


def evaluate(vision, head, loader, device):
    vision.eval()
    head.eval()
    probabilities = []
    targets = []
    with torch.inference_mode():
        for images, labels in loader:
            with torch.autocast('cuda'):
                pooled = vision(pixel_values=images.to(device, non_blocking=True)).pooler_output
                logits = head(pooled)
            probabilities.append(logits.float().sigmoid().cpu().numpy())
            targets.append(labels.numpy())
    probs = np.concatenate(probabilities)
    truth = np.concatenate(targets)
    return score(truth, probs), truth, probs


def write_attribute_metrics(path, names, truth, probs):
    pred = probs >= 0.5
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('attribute', 'positive_frequency', 'accuracy', 'balanced_accuracy',
                         'precision', 'recall', 'F1', 'threshold', 'ECE'))
        for i, name in enumerate(names):
            y, p, z = truth[:, i], probs[:, i], pred[:, i]
            tp = np.sum((y == 1) & z)
            fp = np.sum((y == 0) & z)
            fn = np.sum((y == 1) & ~z)
            tn = np.sum((y == 0) & ~z)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            neg_recall = tn / max(tn + fp, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            bins = np.minimum((p * 10).astype(int), 9)
            ece = sum(np.mean(bins == b) * abs(float(y[bins == b].mean()) -
                      float(p[bins == b].mean())) for b in range(10) if np.any(bins == b))
            writer.writerow((name, float(y.mean()), float((tp + tn) / len(y)),
                             float((recall + neg_recall) / 2), float(precision),
                             float(recall), float(f1), 0.5, float(ece)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=('dinov3_vitb16', 'clip_b16', 'siglip2_b16'), required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--train-csv', type=Path, required=True)
    parser.add_argument('--val-csv', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--val-batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=605)
    parser.add_argument('--backbone-lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--hflip-prob', type=float, default=0.5)
    parser.add_argument('--limit-train', type=int, default=0)
    parser.add_argument('--limit-val', type=int, default=0)
    parser.add_argument('--limit-held', type=int, default=0)
    parser.add_argument('--held-out-domain', choices=('Market1501', 'PA100k', 'PETA'))
    parser.add_argument('--resume', action='store_true',
                        help='Resume the latest epoch checkpoint in output-dir')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda')
    processor_path = args.weights / 'preprocessor_config.json'
    processor = json.loads(processor_path.read_text()) if processor_path.exists() else {}
    mean, std = processor.get('image_mean', MEAN), processor.get('image_std', STD)
    train_set = ChallengeImages(args.train_csv, args.image_root, mean, std,
                                limit=args.limit_train, hflip_prob=args.hflip_prob)
    val_set = ChallengeImages(args.val_csv, args.image_root, mean, std,
                              limit=args.limit_val)
    held_set = None
    if args.held_out_domain:
        held_set = ChallengeImages(args.val_csv, args.image_root, mean, std)
        held_set.rows = [row for row in held_set.rows
                         if row[0].startswith(args.held_out_domain + '/')]
        if args.limit_held:
            held_set.rows = held_set.rows[:args.limit_held]
        train_set.rows = [row for row in train_set.rows
                          if not row[0].startswith(args.held_out_domain + '/')]
        val_set.rows = [row for row in val_set.rows
                        if not row[0].startswith(args.held_out_domain + '/')]
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)
    held_loader = (DataLoader(held_set, batch_size=args.val_batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0)
                   if held_set is not None else None)
    vision, head = make_model(args.backbone, args.weights, device)
    positive = np.asarray([row[1:] for row in train_set.rows], dtype=np.float32).sum(axis=0)
    pos_weight = torch.tensor((len(train_set) - positive) / np.maximum(positive, 1),
                              device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW([
        {'params': [p for p in vision.parameters() if p.requires_grad], 'lr': args.backbone_lr},
        {'params': head.parameters(), 'lr': args.head_lr},
    ], weight_decay=0)
    scaler = torch.amp.GradScaler('cuda')
    torch.cuda.reset_peak_memory_stats()
    best_avg = -1.0
    best_epoch = 0
    history = []
    start_epoch = 1
    last_checkpoint = args.output_dir / 'last_checkpoint.pth'
    if args.resume:
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
        vision.load_state_dict(checkpoint['vision'])
        head.load_state_dict(checkpoint['head'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_avg, best_epoch = checkpoint['best_avg'], checkpoint['best_epoch']
        history = checkpoint['history']
    overall_start = time.monotonic()
    for epoch in range(start_epoch, args.epochs + 1):
        vision.train()
        head.train()
        started = time.monotonic()
        total_loss = 0.0
        for images, labels in train_loader:
            with torch.autocast('cuda'):
                pooled = vision(pixel_values=images.to(device, non_blocking=True)).pooler_output
                loss = criterion(head(pooled), labels.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(images)
        torch.cuda.synchronize()
        train_seconds = time.monotonic() - started
        val_start = time.monotonic()
        metrics, truth, probs = evaluate(vision, head, val_loader, device)
        val_seconds = time.monotonic() - val_start
        row = {'epoch': epoch, 'train_loss': total_loss / len(train_set),
               'train_seconds': train_seconds, 'val_seconds': val_seconds, **metrics}
        history.append(row)
        print(f"epoch {epoch}: train {train_seconds:.1f}s val {val_seconds:.1f}s "
              f"mA {metrics['mA']:.4f} F1 {metrics['Inst_F1']:.4f} "
              f"Avg {metrics['Avg']:.4f}", flush=True)
        if metrics['Avg'] > best_avg:
            best_avg, best_epoch = metrics['Avg'], epoch
            torch.save({'vision': vision.state_dict(), 'head': head.state_dict(),
                        'epoch': epoch, 'metrics': metrics, 'backbone': args.backbone},
                       args.output_dir / 'best_model.pth')
            np.save(args.output_dir / 'best_val_probs.npy', probs)
            np.save(args.output_dir / 'best_val_labels.npy', truth)
            with args.val_csv.open(newline='') as stream:
                names = next(csv.reader(stream))[1:]
            write_attribute_metrics(args.output_dir / 'per_attribute.csv', names, truth, probs)
        torch.save({'vision': vision.state_dict(), 'head': head.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': epoch, 'best_avg': best_avg, 'best_epoch': best_epoch,
                    'history': history}, last_checkpoint)
        (args.output_dir / 'progress.json').write_text(json.dumps({
            'completed_epochs': epoch, 'best_epoch': best_epoch, 'best_avg': best_avg,
            'history': history, 'peak_vram_gib': torch.cuda.max_memory_allocated() / 2**30,
        }, indent=2) + '\n')
    result = {'backbone': args.backbone, 'weights': str(args.weights.resolve()),
              'train_csv': str(args.train_csv.resolve()), 'val_csv': str(args.val_csv.resolve()),
              'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
              'val_batch_size': args.val_batch_size, 'backbone_lr': args.backbone_lr,
              'head_lr': args.head_lr, 'hflip_prob': args.hflip_prob,
              'limit_train': args.limit_train, 'limit_val': args.limit_val,
              'limit_held': args.limit_held,
              'loss': 'weighted_bce', 'best_epoch': best_epoch, 'best_avg': best_avg,
              'held_out_domain': args.held_out_domain,
              'train_images': len(train_set), 'selection_val_images': len(val_set),
              'total_seconds': time.monotonic() - overall_start,
              'peak_vram_gib': torch.cuda.max_memory_allocated() / 2**30,
              'history': history}
    if held_loader is not None:
        best = torch.load(args.output_dir / 'best_model.pth', map_location=device,
                          weights_only=True)
        vision.load_state_dict(best['vision'])
        head.load_state_dict(best['head'])
        held_metrics, held_labels, held_probs = evaluate(vision, head, held_loader, device)
        result['heldout_target_images'] = len(held_set)
        result['heldout_target_metrics'] = held_metrics
        np.save(args.output_dir / 'heldout_target_probs.npy', held_probs)
        np.save(args.output_dir / 'heldout_target_labels.npy', held_labels)
        print(f"held-out {args.held_out_domain}: Avg {held_metrics['Avg']:.4f}", flush=True)
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
