import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@dataclass
class ClipScoreResult:
    file: str
    prompt: str
    score: float


def _get_device() -> torch.device:
    try:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return torch.device("cpu")


def load_clip(device: Optional[torch.device] = None) -> Tuple[CLIPModel, CLIPProcessor, torch.device]:
    """Load CLIP model/processor consistently used across the project."""
    dev = device or _get_device()
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, dev


@torch.no_grad()
def compute_clip_score(model: CLIPModel, processor: CLIPProcessor, image: Image.Image, text: str, device: Optional[torch.device] = None) -> float:
    dev = device or _get_device()
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    # Move tensor inputs to device when available
    inputs = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = model(**inputs)
    score = float(out.logits_per_image.squeeze().item())
    return score


def _safe_load_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def gather_meta_files(meta_dir: str) -> List[str]:
    files = []
    for name in os.listdir(meta_dir):
        if not name.lower().endswith(".meta.json"):
            continue
        files.append(os.path.join(meta_dir, name))
    return sorted(files)


def evaluate_from_meta(meta_dir: str, limit: Optional[int] = None) -> Dict:
    """
    Compute CLIPScore over images referenced by meta files in a directory.

    The meta JSON is expected to contain at least:
    - saved: path to the PNG image
    - original_prompt or prompt: text used for generation

    Returns a dictionary with aggregated statistics and per-file results.
    """
    meta_files = gather_meta_files(meta_dir)
    if limit is not None:
        meta_files = meta_files[:max(0, int(limit))]

    model, processor, device = load_clip()

    results: List[ClipScoreResult] = []
    for mf in meta_files:
        data = _safe_load_json(mf)
        if not isinstance(data, dict):
            continue

        img_path = data.get("saved") or data.get("file")
        prompt = data.get("original_prompt") or data.get("prompt") or data.get("enhanced_prompt")
        sim = data.get("similarity")

        if not img_path or not prompt:
            # Skip incomplete meta entries
            continue
        if not os.path.isfile(img_path):
            # Try relative normalization
            norm = os.path.join(os.path.dirname(mf), os.path.basename(img_path))
            if os.path.isfile(norm):
                img_path = norm
            else:
                continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        try:
            # Recompute CLIPScore to ensure consistency
            score = compute_clip_score(model, processor, image, prompt, device=device)
        except Exception:
            # Fallback to stored similarity if available
            if isinstance(sim, (int, float)):
                score = float(sim)
            else:
                continue

        results.append(ClipScoreResult(file=img_path, prompt=prompt, score=score))

    # Aggregate statistics
    scores = [r.score for r in results]
    if not scores:
        return {
            "count": 0,
            "scores": [],
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "above_thresholds": {}
        }

    import math
    mean = float(sum(scores) / len(scores))
    median = float(sorted(scores)[len(scores) // 2]) if scores else None
    # Population std for reporting; robust to small N using two-pass formula
    var = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = float(math.sqrt(var))
    mn = float(min(scores))
    mx = float(max(scores))

    thresholds = [8.0, 12.0, 16.0, 20.0]
    above = {str(t): round(100.0 * sum(1 for s in scores if s >= t) / len(scores), 2) for t in thresholds}

    return {
        "count": len(scores),
        "mean": round(mean, 4),
        "median": round(median, 4),
        "std": round(std, 4),
        "min": round(mn, 4),
        "max": round(mx, 4),
        "above_thresholds": above,
        "scores": [
            {"file": r.file, "prompt": r.prompt, "score": round(r.score, 4)} for r in results
        ],
    }


def save_report(report: Dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)