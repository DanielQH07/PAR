"""Cache raw EVA and Swin branch features from the published C2T checkpoint.

This is a frozen checkpoint decomposition, not independently retrained ablations.
"""

import argparse
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
from train_cached_features import load_frozen_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config/upar_local.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--label-cache', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--disable-casa', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    update_config(cfg, args)
    device = torch.device('cuda')
    model = load_frozen_model(cfg, args.checkpoint, device)
    backbone = model.backbone
    if args.disable_casa:
        backbone.tablocks = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in backbone.tablocks])
    captured = {}

    def save_eva(_module, inputs):
        captured['eva'] = inputs[0][:, 0]

    def save_swin(_module, inputs):
        captured['swin'] = inputs[0][:, 0]

    hooks = [backbone.cross_att_eva_swin.register_forward_pre_hook(save_eva),
             backbone.cross_att_swin_eva.register_forward_pre_hook(save_swin)]
    _, transform = get_transform(cfg)
    split_paths = {
        'train': Path(cfg.DATASET.PHASE1_ROOT_PATH) / 'train/train.csv',
        'val': Path(cfg.DATASET.PHASE2_ROOT_PATH) / 'annotations/val_gt/val_gt.csv',
    }
    dimension = 1024 if args.disable_casa else 2048
    metadata = {'checkpoint': str(Path(args.checkpoint).resolve()),
                'disable_casa': args.disable_casa, 'dimension': dimension,
                'components': 'Swin before cross-fusion; CASA bypassed' if args.disable_casa
                else 'Swin+CASA before cross-fusion (first 1024); EVA CLS before cross-fusion (last 1024)',
                'amp': True, 'splits': {}}
    for split, path in split_paths.items():
        dataset = PedesAttrUPAR(cfg, str(path), transform=transform,
                                root_path=cfg.DATASET.ROOT_PATH, train=split == 'train')
        if args.limit:
            dataset = Subset(dataset, range(min(args.limit, len(dataset))))
        original_labels = np.load(args.label_cache / f'{split}_labels.npy', mmap_mode='r')
        assert len(dataset) <= len(original_labels)
        cache = np.lib.format.open_memmap(args.output_dir / f'{split}_features.npy',
                                           mode='w+', dtype=np.float32,
                                           shape=(len(dataset), dimension))
        os.symlink((args.label_cache / f'{split}_labels.npy').resolve(),
                   args.output_dir / f'{split}_labels.npy')
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=args.workers > 0)
        start = time.monotonic()
        offset = 0
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for images, _labels, _names in tqdm(loader, desc=f'C2T branch {split}'):
                with torch.cuda.amp.autocast():
                    backbone(images.to(device, non_blocking=True))
                    swin = captured['swin']
                    if args.disable_casa:
                        features = swin
                    else:
                        eva = captured['eva']
                        features = torch.cat((swin, eva), dim=1)
                end = offset + len(images)
                cache[offset:end] = features.float().cpu().numpy()
                offset = end
                captured.clear()
        cache.flush()
        metadata['splits'][split] = {'images': len(dataset),
                                     'seconds': time.monotonic() - start,
                                     'peak_vram_gib': torch.cuda.max_memory_allocated() / 2**30,
                                     'csv': str(path.resolve())}
    for hook in hooks:
        hook.remove()
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
