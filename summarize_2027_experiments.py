"""Regenerate the experiment registry and frozen-backbone bake-off table."""

import csv
import json
import statistics
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path('experiments')
RESULTS = ROOT / 'results'
MODELS = [
    ('ConvNeXt-B', 'convnext_b', 'data/feature_cache_convnext_full', 'lodo_convnext'),
    ('Swin-B', 'swin_b', 'data/feature_cache_swin_full', 'lodo_swin'),
    ('CLIP-B/16 preproj', 'clip_b16_preproj', 'data/feature_cache_clip_full', 'lodo_clip_preproj'),
    ('CLIP-B/16 projected', 'clip_b16_projected', 'data/feature_cache_clip_projected_full', None),
    ('DINOv3 ViT-B/16', 'dinov3_vitb16', 'data/feature_cache_dino_full_v2', 'lodo_dinov3'),
    ('SigLIP2-B/16', 'siglip2_b16', 'data/feature_cache_siglip2_full', 'lodo_siglip2'),
]
DOMAINS = ('Market1501', 'PA100k', 'PETA')


def read_json(path):
    return json.loads(path.read_text())


def portable_path(value):
    if not value:
        return value
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        if 'upar2027_official' in path.parts:
            index = path.parts.index('upar2027_official')
            return (Path('..') / Path(*path.parts[index:])).as_posix()
        return path.name


def main():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    fieldnames = ['id', 'date', 'git_commit', 'git_dirty', 'model', 'pretrain',
                  'input_size', 'loss', 'lr', 'batch', 'epochs', 'seed', 'mA',
                  'Inst_F1', 'Avg', 'ECE', 'peak_vram_gib', 'train_hours',
                  'config_path', 'checkpoint_path', 'notes']
    registry = []
    bakeoff = []
    for display, tag, cache_path, lodo_prefix in MODELS:
        cache = Path(cache_path)
        metadata = read_json(cache / 'metadata.json')
        runs = []
        for seed in (605, 606, 607):
            run = RESULTS / f'lp_{tag}_seed{seed}_50ep'
            if not (run / 'train_summary.json').exists():
                continue
            summary = read_json(run / 'train_summary.json')
            best = summary['history'][summary['best_epoch'] - 1]
            runs.append(best)
            config = {
                'experiment_id': run.name, 'git_commit': commit, 'git_dirty': dirty,
                'model': display,
                'model_weights': f"weights/{Path(metadata['weights']).name}",
                'train_csv': portable_path(metadata['splits']['train']['csv']),
                'val_csv': portable_path(metadata['splits']['val']['csv']),
                'geometry': metadata['geometry'],
                'normalization': metadata['normalization'],
                'extraction_amp': metadata['amp'],
                'head': 'LayerNorm -> Linear(40)', 'head_loss': summary['loss'],
                'lr': summary['lr'], 'batch_size': summary['batch_size'],
                'epochs': summary['epochs'], 'seed': summary['seed'],
                'best_epoch': summary['best_epoch'], 'threshold': 0.5,
                'model_selection': 'pooled validation Avg',
            }
            (run / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
            eval_csv = RESULTS / (run.name + '_eval') / 'per_attribute.csv'
            ece = float(pd.read_csv(eval_csv)['ECE'].mean()) if eval_csv.exists() else ''
            registry.append({
                'id': run.name, 'date': str(date.today()), 'git_commit': commit,
                'git_dirty': int(dirty), 'model': display,
                'pretrain': f"weights/{Path(metadata['weights']).name}",
                'input_size': '224x224 padded',
                'loss': summary['loss'], 'lr': summary['lr'],
                'batch': summary['batch_size'], 'epochs': summary['epochs'],
                'seed': summary['seed'], 'mA': best['mA'],
                'Inst_F1': best['Inst_F1'], 'Avg': best['Avg'], 'ECE': ece,
                'peak_vram_gib': metadata['splits']['train']['peak_vram_bytes'] / 2**30,
                'train_hours': summary['seconds'] / 3600,
                'config_path': str(run / 'config.yaml'),
                'checkpoint_path': str(run / 'best_head.pth'),
                'notes': 'Frozen backbone; deterministic resize/pad; cached features',
            })
        if not runs:
            continue
        lodo = {}
        if lodo_prefix:
            for domain in DOMAINS:
                path = RESULTS / f'{lodo_prefix}_hold_{domain}_50ep' / 'train_summary.json'
                if path.exists():
                    lodo[domain] = read_json(path)['heldout_target_metrics']['Avg']
        values = [run['Avg'] for run in runs]
        bakeoff.append({
            'model': display, 'pooled_avg_mean': statistics.mean(values),
            'pooled_avg_std': statistics.stdev(values) if len(values) > 1 else 0,
            'seeds': len(values), 'market_lodo': lodo.get('Market1501', ''),
            'pa100k_lodo': lodo.get('PA100k', ''), 'peta_lodo': lodo.get('PETA', ''),
            'mean_lodo': statistics.mean(lodo.values()) if len(lodo) == 3 else '',
            'worst_lodo': min(lodo.values()) if len(lodo) == 3 else '',
            'extract_seconds': sum(split['seconds'] for split in metadata['splits'].values()),
            'peak_vram_gib': metadata['splits']['train']['peak_vram_bytes'] / 2**30,
        })
    for run in sorted(RESULTS.glob('lodo_*_50ep')):
        path = run / 'train_summary.json'
        if not path.exists():
            continue
        summary = read_json(path)
        metrics = summary['heldout_target_metrics']
        registry.append({
            'id': run.name, 'date': str(date.today()), 'git_commit': commit,
            'git_dirty': int(dirty), 'model': run.name.split('_hold_')[0][5:],
            'pretrain': '', 'input_size': '224x224 padded', 'loss': summary['loss'],
            'lr': summary['lr'], 'batch': summary['batch_size'],
            'epochs': summary['epochs'], 'seed': summary['seed'],
            'mA': metrics['mA'], 'Inst_F1': metrics['Inst_F1'],
            'Avg': metrics['Avg'], 'ECE': '', 'peak_vram_gib': '',
            'train_hours': summary['seconds'] / 3600,
            'config_path': str(run / 'config.yaml'),
            'checkpoint_path': str(run / 'best_head.pth'),
            'notes': f"Held out {summary['held_out_domain']}; selected on source validation",
        })
        (run / 'config.yaml').write_text(yaml.safe_dump({
        'experiment_id': run.name, 'git_commit': commit, 'git_dirty': dirty,
        'cache_dir': Path(summary['cache_dir']).name,
        'held_out_domain': summary['held_out_domain'],
            'model_selection': 'source-domain validation Avg',
            'target_evaluation': 'held-out domain validation',
            'head': 'LayerNorm -> Linear(40)', 'loss': summary['loss'],
            'lr': summary['lr'], 'batch_size': summary['batch_size'],
            'epochs': summary['epochs'], 'seed': summary['seed'],
            'best_epoch': summary['best_epoch'], 'threshold': 0.5,
        }, sort_keys=False))
    for run in sorted(RESULTS.glob('lp_c2t_fused_seed*_50ep')):
        summary = read_json(run / 'train_summary.json')
        best = summary['history'][summary['best_epoch'] - 1]
        eval_csv = RESULTS / (run.name + '_eval') / 'per_attribute.csv'
        (run / 'config.yaml').write_text(yaml.safe_dump({
            'experiment_id': run.name, 'git_commit': commit, 'git_dirty': dirty,
            'backbone': 'published C2T-Net, frozen',
            'cache_dir': portable_path(summary['cache_dir']),
            'head': 'LayerNorm -> Linear(40)',
            'loss': summary['loss'], 'lr': summary['lr'],
            'batch_size': summary['batch_size'], 'epochs': summary['epochs'],
            'seed': summary['seed'], 'best_epoch': summary['best_epoch'],
            'threshold': 0.5,
        }, sort_keys=False))
        registry.append({
            'id': run.name, 'date': str(date.today()), 'git_commit': commit,
            'git_dirty': int(dirty), 'model': 'C2T-Net frozen + new head',
            'pretrain': 'published C2T best_model.pth', 'input_size': '256x128',
            'loss': summary['loss'], 'lr': summary['lr'],
            'batch': summary['batch_size'], 'epochs': summary['epochs'],
            'seed': summary['seed'], 'mA': best['mA'],
            'Inst_F1': best['Inst_F1'], 'Avg': best['Avg'],
            'ECE': float(pd.read_csv(eval_csv)['ECE'].mean()) if eval_csv.exists() else '',
            'peak_vram_gib': '', 'train_hours': summary['seconds'] / 3600,
            'config_path': str(run / 'config.yaml'),
            'checkpoint_path': str(run / 'best_head.pth'),
            'notes': 'Published frozen checkpoint; new LayerNorm head and weighted BCE',
        })
    for run in sorted(RESULTS.glob('partial_*_seed*_10ep')):
        path = run / 'metrics.json'
        if not path.exists():
            continue
        summary = read_json(path)
        best = summary['history'][summary['best_epoch'] - 1]
        config = {
            'experiment_id': run.name, 'git_commit': commit, 'git_dirty': dirty,
            'backbone': summary['backbone'],
            'weights': f"weights/{Path(summary['weights']).name}",
            'train_csv': portable_path(summary['train_csv']),
            'val_csv': portable_path(summary['val_csv']),
            'input_geometry': 'aspect-preserving resize + pad 224x224',
            'train_augmentation': f"horizontal flip probability {summary['hflip_prob']}",
            'unfrozen': 'last two vision blocks + final norm/pool + LayerNorm→Linear(40)',
            'loss': summary['loss'], 'backbone_lr': summary['backbone_lr'],
            'head_lr': summary['head_lr'], 'batch_size': summary['batch_size'],
            'epochs': summary['epochs'], 'seed': summary['seed'],
            'best_epoch': summary['best_epoch'], 'model_selection': 'pooled validation Avg',
            'threshold': 0.5,
        }
        (run / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        ece = float(pd.read_csv(run / 'per_attribute.csv')['ECE'].mean())
        registry.append({
            'id': run.name, 'date': str(date.today()), 'git_commit': commit,
            'git_dirty': int(dirty), 'model': summary['backbone'] + ' partial FT',
            'pretrain': f"weights/{Path(summary['weights']).name}",
            'loss': summary['loss'], 'lr': f"{summary['backbone_lr']}/{summary['head_lr']}",
            'batch': summary['batch_size'], 'epochs': summary['epochs'],
            'seed': summary['seed'], 'mA': best['mA'],
            'Inst_F1': best['Inst_F1'], 'Avg': best['Avg'], 'ECE': ece,
            'peak_vram_gib': summary['peak_vram_gib'],
            'train_hours': summary['total_seconds'] / 3600,
            'config_path': str(run / 'config.yaml'),
            'checkpoint_path': str(run / 'best_model.pth'),
            'notes': 'Last two vision blocks unfrozen; validation selected by Avg',
        })
    for run in sorted(RESULTS.glob('c2t_*_2027val*')):
        path = run / 'metrics.json'
        if not path.exists() or run.name.endswith('_eval'):
            continue
        report = read_json(path)
        metrics = report['metrics']
        registry.append({
            'id': run.name, 'date': str(date.today()), 'git_commit': commit,
            'git_dirty': int(dirty), 'model': 'C2T-Net published checkpoint',
            'pretrain': 'published C2T best_model.pth', 'input_size': '256x128',
            'loss': 'weighted BCE' if 'head_' in run.name else '',
            'lr': '1e-4' if 'head_' in run.name else '',
            'batch': '1024' if 'head_' in run.name else '',
            'epochs': 10 if 'head_' in run.name else 0,
            'seed': run.name.split('seed_')[-1] if 'seed_' in run.name else '',
            'mA': metrics['mA'], 'Inst_F1': metrics['Inst_F1'],
            'Avg': metrics['Avg'], 'ECE': float(pd.read_csv(run / 'per_attribute.csv')['ECE'].mean()),
            'peak_vram_gib': '', 'train_hours': '',
            'config_path': 'configs/upar_local.yaml',
                'checkpoint_path': portable_path(report['checkpoint_path']),
            'notes': 'Published checkpoint; not independent 2027 training',
        })
    with (ROOT / 'registry.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(registry)
    with (ROOT / 'backbone_bakeoff.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bakeoff[0]))
        writer.writeheader()
        writer.writerows(bakeoff)
    print(f'Wrote {len(registry)} registry rows and {len(bakeoff)} backbone rows')


if __name__ == '__main__':
    main()
