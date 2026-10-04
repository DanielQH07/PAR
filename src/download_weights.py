"""Download the pretrained backbones used by the current recipe into the layout train.py / package_submission.py expect.
  weights/clip_b16        OpenAI CLIP ViT-B/16 (HF format, also the CLIP tokenizer)  - legacy members, OpenAI teacher
  weights_own/metaclip_b16  MetaCLIP B/16 fullcc2.5b (HF CLIPModel)                  - the submitted backbone (v11+)
  weights_own/pe_core_b16   Perception Encoder PE-Core B/16 (timm state dict)          - distillation teacher (v14)
Usage (repository root): python src/download_weights.py [clip metaclip pe]   (default: all)"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

WANT = set(sys.argv[1:]) or {'clip', 'metaclip', 'pe'}
if 'clip' in WANT:
    snapshot_download('openai/clip-vit-base-patch16', local_dir='weights/clip_b16',
                      allow_patterns=['config.json', 'pytorch_model.bin', 'preprocessor_config.json', 'vocab.json',
                                      'merges.txt', 'tokenizer_config.json', 'special_tokens_map.json'])
if 'metaclip' in WANT:
    snapshot_download('facebook/metaclip-b16-fullcc2.5b', local_dir='weights_own/metaclip_b16',
                      allow_patterns=['config.json', 'pytorch_model.bin'])
if 'pe' in WANT:
    import timm
    import torch
    out = Path('weights_own/pe_core_b16')
    out.mkdir(parents=True, exist_ok=True)
    torch.save(timm.create_model('vit_pe_core_base_patch16_224.fb', pretrained=True).state_dict(), out / 'pytorch_model.bin')
print('done:', sorted(WANT))
