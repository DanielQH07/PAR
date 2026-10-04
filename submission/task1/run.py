"""UPAR 2027 Track 1 submission: ensemble of CLIP ViT-B/16 attribute models.

The ingestion program imports this file, calls load_model() once and then
predict_batch(samples). Only torch, numpy and Pillow are needed; the CLIP vision
tower lives in clip_vit.py. All members share the frozen pre-trained CLIP weights
(shared.pth); each member stores only its fine-tuned tail and head (member_*.pth).
Each member's logits (optionally averaged with the h-flipped image) are shifted by a
precomputed contextual-calibration offset; member probabilities are averaged.

Robustness (a Codabench worker turned out to be CPU-only, 6 cores; a previous run was
killed without a traceback after ~13 min, most likely out of memory):
  * CPU: only the first member is built (at its "cpu_size"), no TTA, 2 decode threads,
    periodic gc + malloc_trim, resident memory logged; GPU: the full ensemble;
  * on CPU the ViT linear layers are dynamically quantised to int8;
  * members are in priority order; on the first batch the per-image cost is measured
    and flip-TTA, then trailing members, are dropped to fit TIME_BUDGET_S;
  * hard deadline: once HARD_DEADLINE_S have passed since load_model(), remaining
    images get the training label prior, so predictions.csv is always produced;
  * diagnostics go to stderr so they appear in the ingestion log.
"""
from __future__ import annotations

import ctypes
import gc
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clip_vit import CLIP_MEAN, CLIP_STD, PARModel  # noqa: E402

BATCH_SIZE = 64
TIME_BUDGET_S = float(os.environ.get("UPAR_TIME_BUDGET_S", 2400))  # platform limit is 3600 s total
HARD_DEADLINE_S = float(os.environ.get("UPAR_HARD_DEADLINE_S", 3500))  # v9 took 3076 s on the scorer; ~100 s left for container start + writing before the 3600 s platform limit
DEFAULT_NUM_IMAGES = 29248
_STATE: dict[str, Any] = {}


def _log(message: str) -> None:
    print(f"[run.py] {message}", file=sys.stderr, flush=True)


def _cpu_quota() -> int:
    """CPUs actually available: cgroup quota (docker --cpus) or affinity, not the host count."""
    count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            count = min(count, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, count)


def _quantise(vision: torch.nn.Module, size: tuple[int, int]) -> torch.nn.Module:
    """int8 dynamic quantisation of the Linear layers for CPU speed; falls back to fp32
    if no quantised engine is available or the quantised model fails a test forward."""
    engines = torch.backends.quantized.supported_engines
    engine = next((e for e in ("fbgemm", "x86", "qnnpack") if e in engines), None)
    if engine is None:
        _log(f"no quantised engine in {engines}; staying in fp32")
        return vision
    try:
        torch.backends.quantized.engine = engine
        quantised = torch.ao.quantization.quantize_dynamic(vision, {torch.nn.Linear}, dtype=torch.qint8)
        with torch.inference_mode():
            probe = torch.zeros(1, 3, *size)
            reference, test = vision(probe)[0], quantised(probe)[0]
        if not torch.isfinite(test).all():
            raise RuntimeError("non-finite output")
        _log(f"int8 dynamic quantisation on ({engine}); probe max|diff| {(reference - test).abs().max():.3f}")
        return quantised
    except Exception as error:  # noqa: BLE001 - never fail the run because of an optimisation
        _log(f"quantisation failed ({error!r}); staying in fp32")
        return vision


def load_model() -> None:
    started = time.monotonic()
    config = json.loads((HERE / "members.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("UPAR_THREADS", 0)) or _cpu_quota())
    _log(f"python {platform.python_version()} torch {torch.__version__} device {device}"
         + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
         + f" cpus {os.cpu_count()} threads {torch.get_num_threads()}")
    shared = {k: v.float() for k, v in
              torch.load(HERE / "shared.pth", map_location="cpu", weights_only=True).items()}
    specs = config["members"] if device.type == "cuda" else config["members"][:1]
    members = []
    for member in specs:
        size = tuple(member["size"] if device.type == "cuda" else member.get("cpu_size", member["size"]))
        model = PARModel(size, member["head"])
        model.vision.load_hf_state(shared)
        tail = torch.load(HERE / member["file"], map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict({k: v.float() for k, v in tail.items()}, strict=False)
        head_missing = [k for k in missing if k.startswith("head.")]
        if unexpected or head_missing:
            raise RuntimeError(f"{member['file']}: unexpected {unexpected}, missing {head_missing}")
        model = model.eval()
        if device.type == "cpu" and os.environ.get("UPAR_QUANT"):  # opt-in: slower on some CPUs
            model.vision = _quantise(model.vision, size)
        key = "correction" if device.type == "cuda" else ("cpu_correction" if "cpu_correction" in member else "correction")
        correction = torch.tensor(member.get(key, [0.0] * 40), dtype=torch.float32)
        members.append((model.to(device), size, correction.to(device)))
    del shared
    gc.collect()
    # bf16 on Ampere+; fp16 on older GPUs (T4/V100) where bf16 is not supported
    amp_dtype = (torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
                 else torch.float16)
    names = config["attribute_names"]
    prior = np.asarray(config.get("prior", [0.5] * 40), dtype=np.float64)
    _STATE.update(members=members, device=device, amp_dtype=amp_dtype, amp=device.type == "cuda",
                  prior=prior, started=started, deadline_hit=False,
                  tta=bool(config.get("tta")) and device.type == "cuda", names=names, batches_done=0, index={n: i for i, n in enumerate(names)},
                  pool=None, pool_pid=None, budget_checked=False)
    _log(f"loaded {len(members)}/{len(config['members'])} members {[m[1] for m in members]}, "
         f"tta={_STATE['tta']} in {time.monotonic() - started:.1f}s, rss {_rss_mb()} MB")


def _rss_mb() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def _release_memory() -> None:
    gc.collect()
    try:  # return freed heap pages to the OS (glibc); harmless elsewhere
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _pool() -> ThreadPoolExecutor:
    """One thread pool per process (a pool inherited through fork would deadlock)."""
    if _STATE["pool_pid"] != os.getpid():
        workers = 8 if _STATE["device"].type == "cuda" else 2  # CPU: leave cores to torch
        _STATE["pool"] = ThreadPoolExecutor(max(1, min(workers, _cpu_quota())))
        _STATE["pool_pid"] = os.getpid()
    return _STATE["pool"]


def _resolve(sample: dict[str, Any]) -> str:
    for candidate in (sample.get("image_path"), sample.get("image")):
        if candidate and os.path.exists(candidate):
            return candidate
    image = sample.get("image", "")
    for root in ("/app/input_data/images", "input_data/images", "/app/input_data"):
        if os.path.exists(os.path.join(root, image)):
            return os.path.join(root, image)
    return sample.get("image_path") or image  # let PIL raise a clear error


def _canvases(path: str, sizes: list[tuple[int, int]]) -> list[np.ndarray]:
    """Decode once; aspect-preserving bicubic resize into each HxW canvas padded with
    the CLIP mean. Returns uint8 HxWx3 arrays (normalisation happens batched)."""
    out = []
    with Image.open(path) as image:
        big_h, big_w = max(h for h, _ in sizes), max(w for _, w in sizes)
        if image.format == "JPEG" and image.width >= 4 * big_w and image.height >= 4 * big_h:
            image.draft("RGB", (2 * big_w, 2 * big_h))  # DCT-domain downscale, still >= 2x target
        image = image.convert("RGB")
        for h, w in sizes:
            scale = min(w / image.width, h / image.height)
            new = (min(w, max(1, round(image.width * scale))), min(h, max(1, round(image.height * scale))))
            resized = image.resize(new, Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (w, h), tuple(round(c * 255) for c in CLIP_MEAN))
            canvas.paste(resized, ((w - new[0]) // 2, (h - new[1]) // 2))
            out.append(np.asarray(canvas, dtype=np.uint8))
    return out


def _num_images(samples: list[dict[str, Any]]) -> int:
    """Read num_images from input_data/task.json next to the images, if present."""
    parts = Path(samples[0].get("image_path") or "").parts
    if "images" in parts:
        task = Path(*parts[:len(parts) - parts[::-1].index("images") - 1]) / "task.json"
        try:
            return int(json.loads(task.read_text())["num_images"])
        except Exception:  # noqa: BLE001 - fall back to the documented test-set size
            pass
    return DEFAULT_NUM_IMAGES


def _member_logits(model, batch, correction, tta):
    with torch.inference_mode():
        try:
            with torch.autocast(_STATE["device"].type, dtype=_STATE["amp_dtype"], enabled=_STATE["amp"]):
                logits = model(batch).float()
                if tta:
                    logits = (logits + model(batch.flip(-1)).float()) / 2
        except RuntimeError as error:  # AMP kernel problem on this GPU: retry in fp32 from now on
            if not _STATE["amp"]:
                raise
            _log(f"autocast failed ({error}); switching to fp32")
            _STATE["amp"] = False
            return _member_logits(model, batch, correction, tta)
    return logits - correction


def _fit_budget(samples, batches, prep_per_image) -> None:
    """Time each member on the first batch and drop work that would exceed the budget."""
    device = _STATE["device"]
    for model, size, correction in _STATE["members"]:  # warm-up (CUDA init, cudnn autotune)
        _member_logits(model, batches[size][:2], correction, tta=False)
    costs = []
    for model, size, correction in _STATE["members"]:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.monotonic()
        _member_logits(model, batches[size], correction, tta=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        costs.append((time.monotonic() - t0) / len(samples))
    total = _num_images(samples)
    tta, keep = _STATE["tta"], len(costs)
    # image decoding/resizing is paid once per image regardless of the member count
    projected = lambda k, flip: total * (prep_per_image + sum(costs[:k]) * (2 if flip else 1))  # noqa: E731
    while projected(keep, tta) > TIME_BUDGET_S and (tta or keep > 1):
        if tta:
            tta = False
        else:
            keep -= 1
    _log(f"per-image cost: preprocessing {prep_per_image * 1000:.1f} ms, "
         f"members {[round(c * 1000, 1) for c in costs]} ms; "
         f"{total} images -> keeping {keep}/{len(costs)} members, tta={tta}, "
         f"projected {projected(keep, tta) / 60:.1f} min (budget {TIME_BUDGET_S / 60:.0f} min)")
    _STATE["members"] = _STATE["members"][:keep]
    _STATE["tta"] = tta
    _STATE["budget_checked"] = True


def predict_batch(samples: list[dict[str, Any]]) -> list[list[float]]:
    if not samples:
        return []
    if not _STATE:
        load_model()
    if time.monotonic() - _STATE["started"] > HARD_DEADLINE_S:
        if not _STATE["deadline_hit"]:
            _log("hard deadline reached: returning the label prior for the remaining images")
            _STATE["deadline_hit"] = True
        return _as_rows(np.tile(_STATE["prior"], (len(samples), 1)), samples)
    device = _STATE["device"]
    sizes = sorted({size for _, size, _ in _STATE["members"]})
    prep_started = time.monotonic()
    paths = [_resolve(s) for s in samples]
    canvases = list(_pool().map(lambda p: _canvases(p, sizes), paths))
    mean = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1) * 255
    std = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1) * 255
    batches = {}
    for i, size in enumerate(sizes):
        x = torch.from_numpy(np.stack([c[i] for c in canvases])).to(device)
        batches[size] = (x.permute(0, 3, 1, 2).float() - mean) / std
    if not _STATE["budget_checked"]:
        _fit_budget(samples, batches, (time.monotonic() - prep_started) / len(samples))
    members = _STATE["members"]
    probs = torch.zeros((), device=device)
    for model, size, correction in members:
        probs = probs + _member_logits(model, batches[size], correction, _STATE["tta"]).sigmoid()
    probs = (probs / len(members)).clamp(0.0, 1.0).cpu().numpy().astype(np.float64)
    del batches, canvases
    _STATE["batches_done"] += 1
    if _STATE["batches_done"] % 20 == 0:
        _release_memory()
        _log(f"{_STATE['batches_done']} batches, {time.monotonic() - _STATE['started']:.0f}s, "
             f"rss {_rss_mb()} MB")
    return _as_rows(probs, samples)


def _as_rows(probs: np.ndarray, samples: list[dict[str, Any]]) -> list[list[float]]:
    index = _STATE["index"]
    out = []
    for row, sample in zip(probs, samples):
        names = sample.get("attribute_names") or _STATE["names"]
        if len(names) == 40 and all(n in index for n in names):
            out.append([float(row[index[n]]) for n in names])
        else:  # unexpected naming: the Data page fixes the order, so fall back to it
            if not _STATE.get("warned"):
                _log(f"attribute names not recognised ({list(names)[:3]}...), using positional order")
                _STATE["warned"] = True
            out.append([float(v) for v in row])
    return out


def predict_image(sample: dict[str, Any]) -> list[float]:
    return predict_batch([sample])[0]
