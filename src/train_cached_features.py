"""Cache frozen C2T-Net features and train its linear attribute head.

This is a fast head-only experiment. It does not update the Swin/EVA backbone.
"""

import argparse
import ctypes
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from configs import cfg, update_config
from dataset.augmentation import get_transform
from dataset.pedes_attr.pedes import PedesAttrUPAR
from utils.metrics.pedestrian_metrics import get_pedestrian_metrics
from models import base_block  # registers the linear classifier
from models.backbone import swin_transformer2  # registers the Swin backbone
from models.base_block import FeatClassifier
from models.model_factory import build_backbone, build_classifier


def load_frozen_model(config, checkpoint, device):
    backbone, _ = build_backbone(config.BACKBONE.TYPE, config.BACKBONE.MULTISCALE)
    classifier = build_classifier(config.CLASSIFIER.NAME)(
        nattr=40, c_in=2048, bn=config.CLASSIFIER.BN,
        pool=config.CLASSIFIER.POOLING, scale=config.CLASSIFIER.SCALE,
    )
    model = FeatClassifier(backbone, classifier).to(device)
    if device.type == 'cuda':
        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    saved = torch.load(checkpoint, map_location=device)
    state = saved['state_dicts'] if 'state_dicts' in saved else saved
    state = {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    del saved, state
    gc.collect()
    if device.type == 'cuda':
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def cache_split(config, split, model, transform, out_dir, batch_size, workers, limit, device, amp):
    if split == 'train':
        csv_path = Path(config.DATASET.PHASE1_ROOT_PATH) / 'train/train.csv'
    else:
        csv_path = Path(config.DATASET.PHASE2_ROOT_PATH) / 'annotations/val_gt/val_gt.csv'
    dataset = PedesAttrUPAR(
        config, str(csv_path), transform=transform,
        root_path=config.DATASET.ROOT_PATH, train=(split == 'train'),
    )
    if limit:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    count = len(dataset)
    features = np.lib.format.open_memmap(
        out_dir / (split + '_features.npy'), mode='w+', dtype=np.float32, shape=(count, 2048)
    )
    labels = np.lib.format.open_memmap(
        out_dir / (split + '_labels.npy'), mode='w+', dtype=np.float32, shape=(count, 40)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=(device.type == 'cuda'))
    offset = 0
    started = time.monotonic()
    with torch.inference_mode():
        for images, targets, _ in tqdm(loader, desc='Cache ' + split):
            with torch.cuda.amp.autocast(enabled=amp and device.type == 'cuda'):
                vectors = model.backbone(images.to(device, non_blocking=True))
            size = vectors.shape[0]
            features[offset:offset + size] = vectors.float().cpu().numpy()
            labels[offset:offset + size] = targets.numpy()
            offset += size
    features.flush()
    labels.flush()
    print('{}: {} images, {:.1f}s, {:.1f} images/s'.format(
        split, count, time.monotonic() - started, count / (time.monotonic() - started)))
    return count


def extract(args):
    update_config(cfg, args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, deterministic_transform = get_transform(cfg)
    model = load_frozen_model(cfg, args.checkpoint, device)
    torch.save(model.classifier.logits[0].state_dict(), out_dir / 'classifier_init.pth')
    counts = {}
    for split in ('train', 'val'):
        counts[split] = cache_split(cfg, split, model, deterministic_transform, out_dir,
                                   args.batch_size, args.workers, args.limit, device, args.amp)
    metadata = {
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'config': str(Path(args.cfg).resolve()),
        'feature_dim': 2048,
        'counts': counts,
        'transform': 'deterministic validation transform; frozen backbone',
        'amp': args.amp,
    }
    (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


def evaluate(head, features, labels, batch_size, device):
    predictions = np.empty((len(features), 40), dtype=np.float32)
    head.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            inputs = torch.from_numpy(np.array(features[start:end])).to(device)
            predictions[start:end] = head(inputs).sigmoid().cpu().numpy()
    truth = np.array(labels)
    truth[truth == -1] = 0
    return get_pedestrian_metrics(truth, predictions)


def train_head(args):
    out_dir = Path(args.output_dir)
    run_dir = out_dir / 'head_runs' / args.run_id if args.run_id else out_dir
    if args.run_id:
        run_dir.mkdir(parents=True, exist_ok=False)
    train_features = np.load(out_dir / 'train_features.npy', mmap_mode='r')
    train_labels = np.load(out_dir / 'train_labels.npy', mmap_mode='r')
    val_features = np.load(out_dir / 'val_features.npy', mmap_mode='r')
    val_labels = np.load(out_dir / 'val_labels.npy', mmap_mode='r')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    head = torch.nn.Linear(train_features.shape[1], train_labels.shape[1]).to(device)
    head.load_state_dict(torch.load(out_dir / 'classifier_init.pth', map_location=device))
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    ratio = torch.from_numpy(np.array(train_labels).mean(axis=0)).to(device)
    baseline = evaluate(head, val_features, val_labels, args.batch_size, device)
    print('baseline val mA={:.4f}, F1={:.4f}'.format(baseline.ma, baseline.instance_f1))
    rng = np.random.default_rng(args.seed)
    best_score = (baseline.ma + baseline.instance_f1) / 2 if args.selection_metric == 'Avg' else baseline.ma
    best_epoch = 0
    best_metrics = {'mA': float(baseline.ma), 'Inst_F1': float(baseline.instance_f1),
                    'Avg': float((baseline.ma + baseline.instance_f1) / 2)}
    torch.save({'classifier': head.state_dict(), 'epoch': 0,
                'val_ma': float(baseline.ma), 'val_avg': best_metrics['Avg']},
               run_dir / 'best_cached_head.pth')
    for epoch in range(args.epochs):
        order = rng.permutation(len(train_features))
        started = time.monotonic()
        head.train()
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            indexes = order[start:start + args.batch_size]
            inputs = torch.from_numpy(np.array(train_features[indexes])).to(device)
            targets = torch.from_numpy(np.array(train_labels[indexes])).to(device)
            logits = head(inputs)
            positive = (targets > 0.5).float()
            weights = torch.exp(positive * (1 - ratio) + (1 - positive) * ratio)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                logits, targets, reduction='none') * weights).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indexes)
        metrics = evaluate(head, val_features, val_labels, args.batch_size, device)
        avg = (metrics.ma + metrics.instance_f1) / 2
        print('epoch {}: {:.1f}s, train loss {:.4f}, val mA {:.4f}, F1 {:.4f}, Avg {:.4f}'.format(
            epoch + 1, time.monotonic() - started, total_loss / len(order),
            metrics.ma, metrics.instance_f1, avg), flush=True)
        score = avg if args.selection_metric == 'Avg' else metrics.ma
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = {'mA': float(metrics.ma), 'Inst_F1': float(metrics.instance_f1),
                            'Avg': float(avg)}
            torch.save({'classifier': head.state_dict(), 'epoch': epoch + 1,
                        'val_ma': float(metrics.ma), 'val_avg': float(avg)},
                       run_dir / 'best_cached_head.pth')
    (run_dir / 'run_metadata.json').write_text(json.dumps({
        'seed': args.seed, 'epochs': args.epochs, 'learning_rate': args.lr,
        'batch_size': args.batch_size, 'selection_metric': args.selection_metric,
        'best_epoch': best_epoch, 'best_metrics': best_metrics,
        'cache_dir': str(out_dir.resolve()),
    }, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('extract', 'train'))
    parser.add_argument('--cfg', default='config/upar_local.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--output-dir', default='data/feature_cache')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0, help='Limit each split for a smoke test')
    parser.add_argument('--amp', action='store_true', help='Use CUDA mixed precision during extraction')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=605)
    parser.add_argument('--run-id', default=None, help='Unique directory name under head_runs')
    parser.add_argument('--selection-metric', choices=('Avg', 'mA'), default='Avg')
    args = parser.parse_args()
    if args.mode == 'extract':
        extract(args)
    else:
        train_head(args)


if __name__ == '__main__':
    main()
