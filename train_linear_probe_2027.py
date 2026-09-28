"""Train the same LayerNorm + Linear head on cached 2027 backbone features."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from evaluate_cached_head import score


def predict(model, features, batch_size, device):
    result = np.empty((len(features), 40), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            x = torch.from_numpy(np.asarray(features[start:end]).copy()).to(device)
            result[start:end] = model(x).sigmoid().cpu().numpy()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss', choices=('weighted_bce', 'bce'), default='weighted_bce')
    parser.add_argument('--feature-slice', choices=('all', 'first_half', 'second_half'),
                        default='all')
    parser.add_argument('--seed', type=int, default=605)
    parser.add_argument('--train-csv', type=Path)
    parser.add_argument('--val-csv', type=Path)
    parser.add_argument('--held-out-domain', choices=('Market1501', 'PA100k', 'PETA'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_x = np.load(args.cache_dir / 'train_features.npy', mmap_mode='r')
    train_y = np.load(args.cache_dir / 'train_labels.npy', mmap_mode='r')
    val_x = np.load(args.cache_dir / 'val_features.npy', mmap_mode='r')
    val_y = np.asarray(np.load(args.cache_dir / 'val_labels.npy', mmap_mode='r'))
    if args.feature_slice != 'all':
        assert train_x.shape[1] == val_x.shape[1] and train_x.shape[1] % 2 == 0
        halfway = train_x.shape[1] // 2
        selected = slice(0, halfway) if args.feature_slice == 'first_half' else slice(halfway, None)
        train_x, val_x = train_x[:, selected], val_x[:, selected]
    assert train_y.shape[1] == val_y.shape[1] == 40
    assert np.isin(train_y, (0, 1)).all() and np.isin(val_y, (0, 1)).all()
    train_indices = np.arange(len(train_x))
    target_x = target_y = None
    if args.held_out_domain:
        if not args.train_csv or not args.val_csv:
            parser.error('--train-csv and --val-csv are required for LODO')
        def paths(path):
            with path.open(newline='') as stream:
                reader = csv.reader(stream)
                next(reader)
                return [row[0] for row in reader]
        train_paths, val_paths = paths(args.train_csv), paths(args.val_csv)
        assert len(train_paths) == len(train_x) and len(val_paths) == len(val_x)
        train_mask = np.fromiter((not p.startswith(args.held_out_domain + '/')
                                  for p in train_paths), bool)
        source_mask = np.fromiter((not p.startswith(args.held_out_domain + '/')
                                   for p in val_paths), bool)
        train_indices = np.flatnonzero(train_mask)
        target_x, target_y = val_x[~source_mask], val_y[~source_mask]
        val_x, val_y = val_x[source_mask], val_y[source_mask]
    model = torch.nn.Sequential(torch.nn.LayerNorm(train_x.shape[1]),
                                torch.nn.Linear(train_x.shape[1], 40)).to(device)
    positive = np.asarray(train_y[train_indices]).sum(axis=0)
    pos_weight = torch.tensor((len(train_indices) - positive) / np.maximum(positive, 1),
                              dtype=torch.float32, device=device)
    loss_fn = (torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
               if args.loss == 'weighted_bce' else torch.nn.BCEWithLogitsLoss())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    history = []
    best_avg = -1.0
    best_epoch = 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        order = rng.permutation(train_indices)
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            x = torch.from_numpy(np.asarray(train_x[index]).copy()).to(device)
            y = torch.from_numpy(np.asarray(train_y[index]).copy()).to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(index)
        metrics = score(val_y, predict(model, val_x, args.batch_size, device))
        entry = {'epoch': epoch, 'train_loss': total / len(order), **metrics}
        history.append(entry)
        print(f"epoch {epoch}: loss={entry['train_loss']:.4f} "
              f"mA={metrics['mA']:.4f} F1={metrics['Inst_F1']:.4f} "
              f"Avg={metrics['Avg']:.4f}", flush=True)
        if metrics['Avg'] > best_avg:
            best_avg, best_epoch = metrics['Avg'], epoch
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'metrics': metrics, 'head_type': 'layernorm_linear'},
                       args.output_dir / 'best_head.pth')
    summary = {'seed': args.seed, 'epochs': args.epochs, 'batch_size': args.batch_size,
               'lr': args.lr, 'loss': args.loss,
               'best_epoch': best_epoch, 'best_avg': best_avg,
               'seconds': time.monotonic() - started,
               'peak_vram_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
               'cache_dir': str(args.cache_dir.resolve()), 'history': history,
               'feature_slice': args.feature_slice,
               'held_out_domain': args.held_out_domain,
               'train_images': len(train_indices), 'selection_val_images': len(val_y)}
    if target_x is not None:
        best = torch.load(args.output_dir / 'best_head.pth', map_location=device,
                          weights_only=True)
        model.load_state_dict(best['model'])
        summary['heldout_target_images'] = len(target_y)
        summary['heldout_target_metrics'] = score(
            target_y, predict(model, target_x, args.batch_size, device))
        print('held-out target:', json.dumps(summary['heldout_target_metrics']), flush=True)
    (args.output_dir / 'train_summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
