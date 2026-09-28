"""Apply CLIP's trained visual projection to cached pre-projection features."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8192)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = AutoModel.from_pretrained(args.weights, local_files_only=True)
    projection = model.visual_projection.cpu().eval()
    metadata = json.loads((args.source_dir / 'metadata.json').read_text())
    metadata['backbone'] = 'clip_b16_projected'
    metadata['source_cache'] = str(args.source_dir.resolve())
    metadata['feature_dim'] = projection.out_features
    metadata['projection'] = 'trained CLIP visual_projection, no L2 normalization'
    for split in ('train', 'val'):
        source = np.load(args.source_dir / f'{split}_features.npy', mmap_mode='r')
        output = np.lib.format.open_memmap(args.output_dir / f'{split}_features.npy',
                                            mode='w+', dtype=np.float32,
                                            shape=(len(source), projection.out_features))
        with torch.inference_mode():
            for start in range(0, len(source), args.batch_size):
                end = min(start + args.batch_size, len(source))
                output[start:end] = projection(
                    torch.from_numpy(np.asarray(source[start:end]).copy())).numpy()
        output.flush()
        os.symlink((args.source_dir / f'{split}_labels.npy').resolve(),
                   args.output_dir / f'{split}_labels.npy')
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
