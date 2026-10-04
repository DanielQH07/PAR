"""Convert an open_clip ViT-B/16 checkpoint (visual tower) to the Hugging Face CLIP key layout used by
clip_vit.CLIPVision.load_hf_state and src/package_submission.py (vision_model.* + visual_projection.weight).
Usage: python src/convert_openclip_vision.py open_clip_pytorch_model.bin out_dir/
Writes out_dir/pytorch_model.bin (vision tower only)."""

import sys
from pathlib import Path

import torch


def convert(sd):
    sd = {k.removeprefix('module.'): v for k, v in sd.items()}
    out = {'vision_model.embeddings.class_embedding': sd['visual.class_embedding'],
           'vision_model.embeddings.position_embedding.weight': sd['visual.positional_embedding'],
           'vision_model.embeddings.patch_embedding.weight': sd['visual.conv1.weight'],
           'vision_model.pre_layrnorm.weight': sd['visual.ln_pre.weight'],
           'vision_model.pre_layrnorm.bias': sd['visual.ln_pre.bias'],
           'vision_model.post_layernorm.weight': sd['visual.ln_post.weight'],
           'vision_model.post_layernorm.bias': sd['visual.ln_post.bias'],
           'visual_projection.weight': sd['visual.proj'].t().contiguous()}  # open_clip: x @ proj
    n = 1 + max(int(k.split('.')[3]) for k in sd if k.startswith('visual.transformer.resblocks.'))
    for i in range(n):
        s, h = f'visual.transformer.resblocks.{i}.', f'vision_model.encoder.layers.{i}.'
        q, k, v = sd[s + 'attn.in_proj_weight'].chunk(3)
        qb, kb, vb = sd[s + 'attn.in_proj_bias'].chunk(3)
        out.update({h + 'self_attn.q_proj.weight': q, h + 'self_attn.k_proj.weight': k, h + 'self_attn.v_proj.weight': v,
                    h + 'self_attn.q_proj.bias': qb, h + 'self_attn.k_proj.bias': kb, h + 'self_attn.v_proj.bias': vb,
                    h + 'self_attn.out_proj.weight': sd[s + 'attn.out_proj.weight'],
                    h + 'self_attn.out_proj.bias': sd[s + 'attn.out_proj.bias'],
                    h + 'layer_norm1.weight': sd[s + 'ln_1.weight'], h + 'layer_norm1.bias': sd[s + 'ln_1.bias'],
                    h + 'layer_norm2.weight': sd[s + 'ln_2.weight'], h + 'layer_norm2.bias': sd[s + 'ln_2.bias'],
                    h + 'mlp.fc1.weight': sd[s + 'mlp.c_fc.weight'], h + 'mlp.fc1.bias': sd[s + 'mlp.c_fc.bias'],
                    h + 'mlp.fc2.weight': sd[s + 'mlp.c_proj.weight'], h + 'mlp.fc2.bias': sd[s + 'mlp.c_proj.bias']})
    return {k: v.contiguous() for k, v in out.items()}


if __name__ == '__main__':
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    out = convert(torch.load(src, map_location='cpu', weights_only=True))
    torch.save(out, dst / 'pytorch_model.bin')
    print(f'{len(out)} tensors -> {dst / "pytorch_model.bin"}')
