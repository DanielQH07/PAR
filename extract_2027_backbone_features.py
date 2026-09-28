"""Offline, aspect-preserving feature extraction for 2027 linear probes."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ChallengeImages(Dataset):
    def __init__(self, csv_path, image_root, mean, std, limit=0, hflip_prob=0.0):
        with open(csv_path, newline='') as stream:
            rows = list(csv.reader(stream))
        self.header, self.rows = rows[0], rows[1:]
        if limit:
            self.rows = self.rows[:limit]
        self.root = Path(image_root)
        self.hflip_prob = hflip_prob
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.pad_color = tuple(round(channel * 255) for channel in mean)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row[0]) as image:
            image = image.convert('RGB')
            scale = min(224 / image.width, 224 / image.height)
            new_size = (min(224, max(1, round(image.width * scale))),
                        min(224, max(1, round(image.height * scale))))
            image = image.resize(new_size, Image.Resampling.BICUBIC)
            if random.random() < self.hflip_prob:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            canvas = Image.new('RGB', (224, 224), self.pad_color)
            canvas.paste(image, ((224 - image.width) // 2, (224 - image.height) // 2))
            array = np.asarray(canvas, dtype=np.uint8).copy()
        x = torch.from_numpy(array).permute(2, 0, 1).float().div_(255)
        return (x - self.mean) / self.std, np.asarray(row[1:], dtype=np.float32)


def load_model(kind, weights, device):
    if kind in ('dinov3_vitb16', 'dinov3_convnext_b'):
        from transformers import AutoModel
        model = AutoModel.from_pretrained(weights, local_files_only=True,
                                          attn_implementation='sdpa')
        dimension = model.config.hidden_size

        def features(images):
            return model(pixel_values=images).pooler_output
    elif kind == 'convnext_b':
        import timm
        model = timm.create_model('convnext_base.fb_in22k_ft_in1k',
                                  pretrained=False, num_classes=0)
        missing, unexpected = model.load_state_dict(
            load_file(str(weights / 'model.safetensors'), device='cpu'), strict=False)
        if missing or set(unexpected) != {'head.fc.weight', 'head.fc.bias'}:
            raise RuntimeError(f'Unexpected ConvNeXt weight keys: {missing}, {unexpected}')
        dimension = model.num_features

        def features(images):
            return model.forward_features(images).mean(dim=(-2, -1))
    elif kind == 'swin_b':
        from transformers import AutoModel
        model = AutoModel.from_pretrained(weights, local_files_only=True)
        dimension = model.config.hidden_size

        def features(images):
            return model(pixel_values=images).pooler_output
    elif kind in ('clip_b16', 'siglip2_b16'):
        from transformers import AutoModel
        full_model = AutoModel.from_pretrained(weights, local_files_only=True)
        dimension = full_model.config.vision_config.hidden_size
        model = full_model.vision_model
        del full_model

        def features(images):
            return model(pixel_values=images).pooler_output
    else:
        raise ValueError(kind)
    model.to(device).eval()
    model.requires_grad_(False)
    return model, features, dimension


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=('dinov3_vitb16', 'dinov3_convnext_b', 'convnext_b',
                                                'swin_b', 'clip_b16', 'siglip2_b16'),
                        required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--train-csv', type=Path, required=True)
    parser.add_argument('--val-csv', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    processor_path = args.weights / 'preprocessor_config.json'
    processor = json.loads(processor_path.read_text()) if processor_path.exists() else {}
    mean = processor.get('image_mean', MEAN)
    std = processor.get('image_std', STD)
    if isinstance(mean, (int, float)):
        mean = [mean] * 3
    if isinstance(std, (int, float)):
        std = [std] * 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')
    model, extract, dimension = load_model(args.backbone, args.weights, device)
    metadata = {'backbone': args.backbone, 'weights': str(args.weights.resolve()),
                'feature_dim': dimension, 'geometry': 'aspect-preserving resize + centered pad 224x224',
                'normalization': {'mean': mean, 'std': std}, 'amp': device.type == 'cuda',
                'batch_size': args.batch_size, 'workers': args.workers, 'splits': {}}
    for split, csv_path in (('train', args.train_csv), ('val', args.val_csv)):
        dataset = ChallengeImages(csv_path, args.image_root, mean, std, args.limit)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=device.type == 'cuda', persistent_workers=args.workers > 0)
        x_cache = np.lib.format.open_memmap(args.output_dir / f'{split}_features.npy',
                                             mode='w+', dtype=np.float32,
                                             shape=(len(dataset), dimension))
        y_cache = np.lib.format.open_memmap(args.output_dir / f'{split}_labels.npy',
                                             mode='w+', dtype=np.float32,
                                             shape=(len(dataset), 40))
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        offset = 0
        started = time.monotonic()
        with torch.inference_mode():
            for images, labels in tqdm(loader, desc=f'{args.backbone} {split}'):
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    vectors = extract(images.to(device, non_blocking=True))
                end = offset + len(images)
                x_cache[offset:end] = vectors.float().cpu().numpy()
                y_cache[offset:end] = labels.numpy()
                offset = end
        x_cache.flush()
        y_cache.flush()
        elapsed = time.monotonic() - started
        metadata['splits'][split] = {'images': len(dataset), 'seconds': elapsed,
                                     'peak_vram_bytes': torch.cuda.max_memory_allocated()
                                     if device.type == 'cuda' else None,
                                     'csv': str(csv_path.resolve())}
        print(f'{split}: {len(dataset)} images in {elapsed:.1f}s', flush=True)
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
