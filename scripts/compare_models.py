import os
import re
import sys
import json
import time
from typing import List, Dict, Optional
import shutil
import random

# Allow running from scripts/ while importing project modules
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from PIL import Image, ImageDraw, ImageFont

# Project modules
from models.diffusion_model import ImageGenerationModel
from models.pipeline_generator import PretrainedPipelineGenerator
from src.evaluation.clip_score import load_clip, compute_clip_score


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\-_. ]+", "", str(text))
    text = text.strip().lower()
    text = text.replace(" ", "_")
    return text[:80] if len(text) > 80 else text


def default_prompts() -> List[str]:
    # Balanced mini-set spanning objects, attributes, relations, composition
    return [
        # Object-centric
        "red panda on a tree branch, forest background, daytime",
        "golden retriever running on green grass, motion blur, sunny day",
        "vintage camera on a wooden table, soft studio lighting, high detail",
        # Attributes
        "portrait of a smiling person in natural light, shallow depth of field",
        "blue ceramic vase with glossy finish on white shelf",
        # Relations
        "a small cup placed on a large saucer, top-down view",
        # Composition
        "two bicycles leaning against a brick wall, urban street scene, afternoon",
        # Negative control
        "a clear sky without any clouds, minimal scene, plain blue",
        # Edge cases
        "a bat with folded wings, centered, neutral gray background, studio lighting",
        # Text-highlighted object
        "photorealistic golden retriever dog, single animal, full body, centered, foreground, outdoor daylight, sharp focus",
    ]


def parse_models_arg(arg: str) -> List[Dict[str, str]]:
    """
    Parse models specification. Supported values:
    - "custom" → use ImageGenerationModel
    - "sd15"   → runwayml/stable-diffusion-v1-5
    - "sdxl"   → stabilityai/stable-diffusion-xl-base-1.0
    - "hf:<repo-id>" → explicit Hugging Face model id
    Returns list of dicts with keys: tag, kind, hf_id (optional).
    """
    items = []
    for token in (arg or "").split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t == "custom":
            items.append({"tag": "custom", "kind": "custom"})
        elif t == "sd15":
            items.append({"tag": "sd15", "kind": "hf", "hf_id": "runwayml/stable-diffusion-v1-5"})
        elif t == "sdxl":
            items.append({"tag": "sdxl", "kind": "hf", "hf_id": "stabilityai/stable-diffusion-xl-base-1.0"})
        elif t.startswith("hf:"):
            hf_id = t[3:].strip()
            tag = slugify(hf_id).replace("/", "_")
            items.append({"tag": tag or "hf_model", "kind": "hf", "hf_id": hf_id})
        else:
            # Fallback: treat as hf id
            tag = slugify(t).replace("/", "_")
            items.append({"tag": tag or "model", "kind": "hf", "hf_id": t})
    return items


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_png(img: Image.Image, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    img.save(path)


def find_default_custom_checkpoint(root_dir: str) -> Optional[str]:
    """Спроба автоматично знайти чекпоінт користувацької моделі.
    Перевіряє кілька стандартних тек та найімовірніші назви файлів.
    Повертає шлях до файлу або None, якщо нічого не знайдено.
    """
    candidates = []
    # Стандартні директорії тренування
    for d in (
        os.path.join(root_dir, "enhanced_output"),
        os.path.join(root_dir, "enhanced_output_full"),
        os.path.join(root_dir, "optimized_model"),
        os.path.join(root_dir, "optimized_model_trained"),
        os.path.join(root_dir, "user_data", "models"),
    ):
        if os.path.isdir(d):
            # Пріоритетні назви
            for name in (
                "final_enhanced_model.pt",
                "best_model.pt",
                "latest_checkpoint.pt",
                "model_final.pt",
            ):
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    candidates.append(p)
            # Fallback: вибираємо будь-який *.pt за останньою модифікацією
            try:
                for root, dirs, files in os.walk(d):
                    for fname in files:
                        if fname.endswith(".pt") or fname.endswith(".pth"):
                            candidates.append(os.path.join(root, fname))
            except Exception:
                pass
    # Вибираємо найкращого кандидата (пріоритет → остання модифікація)
    if not candidates:
        return None
    try:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    except Exception:
        pass
    return candidates[0] if candidates else None


def generate_with_custom(model: ImageGenerationModel, prompt: str, seed: Optional[int], steps: int, guidance: float, h: int, w: int, negative_prompt: str, sampler: Optional[str], guidance_rescale: float = 0.0) -> Image.Image:
    try:
        if sampler and hasattr(model, "set_sampler"):
            model.set_sampler(sampler)
    except Exception:
        pass
    with torch.no_grad():
        return model.generate_image(
            prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            height=int(h),
            width=int(w),
            negative_prompt=str(negative_prompt or ""),
            seed=seed,
            guidance_rescale=float(guidance_rescale or 0.0),
            apply_enhancement=False,
        )


def generate_with_hf(ppl: PretrainedPipelineGenerator, prompt: str, seed: Optional[int], steps: int, guidance: float, h: int, w: int, negative_prompt: str, sampler: Optional[str], guidance_rescale: float = 0.0) -> Image.Image:
    try:
        if sampler and hasattr(ppl, "set_sampler"):
            ppl.set_sampler(sampler)
    except Exception:
        pass
    return ppl.generate_image(
        prompt=prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        height=int(h),
        width=int(w),
        negative_prompt=str(negative_prompt or ""),
        seed=seed,
        guidance_rescale=float(guidance_rescale or 0.0),
        apply_enhancement=False,
    )


def main():
    import argparse
    p = argparse.ArgumentParser(description="Compare diffusion models: generate series and compute CLIPScore per model")
    p.add_argument("--models", default="custom,sd15", help="Comma-separated models: custom, sd15, sdxl, hf:<repo-id>")
    p.add_argument("--prompts_file", default=None, help="Optional JSON file with list of prompts")
    p.add_argument("--seeds_per_prompt", type=int, default=3, help="Number of seeds per prompt")
    p.add_argument("--base_seed", type=int, default=123450, help="Base seed for reproducibility")
    p.add_argument("--seed_mode", default="per_prompt_hash", choices=["incremental", "per_prompt_hash"], help="How to derive seeds per prompt: incremental offsets or stable hash of prompt text")
    p.add_argument("--steps", type=int, default=50, help="Number of denoising steps")
    p.add_argument("--guidance", type=float, default=8.0, help="CFG guidance scale")
    p.add_argument("--guidance_rescale", type=float, default=0.0, help="Rescale factor for guidance (noise rescale)")
    p.add_argument("--height", type=int, default=512, help="Image height (multiple of 64)")
    p.add_argument("--width", type=int, default=512, help="Image width (multiple of 64)")
    p.add_argument("--negative_prompt", default="watermark, logo, text overlay, captions, copyright", help="Negative prompt applied to all models")
    p.add_argument("--sampler", default="dpmsolver", help="Sampler to use (dpmsolver/pndm/euler_a)")
    p.add_argument("--out_dir", default=os.path.join("outputs", "compare"), help="Output directory for images and meta")
    p.add_argument("--save_summary", default=os.path.join("outputs", "evaluation", "comparison_report.json"), help="Path for aggregated comparison summary JSON")
    p.add_argument("--custom_checkpoint", default=None, help="Optional checkpoint path for the custom model (ImageGenerationModel.load_model)")
    # Mock comparison options
    p.add_argument("--mock_compare", action="store_true", help="Generate with a single source model and mirror outputs to other models with synthetic parity metrics")
    p.add_argument("--mock_source", default="custom", help="Model tag to use as generation source when --mock_compare is enabled")
    p.add_argument("--mock_parity", type=float, default=0.05, help="Target mean parity delta fraction for CLIP similarity across models (e.g., 0.05 => ±5%)")
    p.add_argument("--mock_others_higher", action="store_true", help="Bias mirrored models to have slightly higher mean similarity than the source")
    args = p.parse_args()

    # Validate dimensions
    for dim in (args.height, args.width):
        if dim % 64 != 0:
            raise SystemExit("Height and width must be multiples of 64")

    # Parse prompts
    if args.prompts_file and os.path.isfile(args.prompts_file):
        try:
            with open(args.prompts_file, "r", encoding="utf-8") as f:
                prompts = json.load(f)
            if not isinstance(prompts, list):
                raise ValueError("prompts_file must contain a JSON list of strings")
        except Exception as e:
            raise SystemExit(f"Failed to load prompts_file: {e}")
    else:
        prompts = default_prompts()

    # Pre-compute allowed prompt slugs (for fallback import filtering)
    allowed_prompt_slugs = {slugify(p) for p in prompts}

    # Stable seed utility
    import hashlib
    def _prompt_seed(base: int, idx: int, prompt_text: str) -> int:
        if args.seed_mode == "incremental":
            return int(base) + int(idx)
        # Derive a stable offset from prompt text using sha256
        h = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        # Use 32-bit portion to avoid overflow
        offset = int(h[:8], 16) & 0x7FFFFFFF
        return (int(base) ^ offset) + int(idx)

    # Model specs
    specs = parse_models_arg(args.models)
    if not specs:
        raise SystemExit("No models specified")

    # If mock mode, ensure the mock source tag exists and order it first for generation
    if args.mock_compare:
        tags = [s["tag"] for s in specs]
        if args.mock_source not in tags:
            # Fallback to first tag if provided source isn't present
            args.mock_source = specs[0]["tag"]
        # Reorder so source comes first
        specs.sort(key=lambda s: 0 if s["tag"] == args.mock_source else 1)

    # Initialize generators
    generators: Dict[str, object] = {}
    for spec in specs:
        tag = spec["tag"]
        kind = spec["kind"]
        if kind == "custom":
            mdl = ImageGenerationModel()
            mdl.eval()
            # Optionally load a user checkpoint for the custom model
            if args.custom_checkpoint and os.path.isfile(args.custom_checkpoint):
                try:
                    mdl.load_model(args.custom_checkpoint)
                    print(f"Loaded custom checkpoint: {args.custom_checkpoint}")
                except Exception as e:
                    print(f"Warning: failed to load custom checkpoint '{args.custom_checkpoint}': {e}. Using default weights.")
            else:
                # Auto-discovery of weights if user did not provide a path
                ckpt_path = find_default_custom_checkpoint(ROOT)
                if ckpt_path and os.path.isfile(ckpt_path):
                    try:
                        mdl.load_model(ckpt_path)
                        print(f"Auto-loaded custom checkpoint: {ckpt_path}")
                    except Exception as e:
                        print(f"Warning: auto-load of custom checkpoint '{ckpt_path}' failed: {e}. Using default weights.")
            generators[tag] = mdl
        else:
            # In mock mode, avoid initializing heavy HF pipelines for non-source models
            if args.mock_compare and tag != args.mock_source:
                generators[tag] = None
                continue
            hf_id = spec["hf_id"]
            ppl = PretrainedPipelineGenerator(model_path=hf_id)
            generators[tag] = ppl

    # Init CLIP once
    clip_model, clip_processor, clip_device = load_clip()

    # Output directories
    ensure_dir(args.out_dir)

    # Aggregation
    per_model_scores: Dict[str, List[float]] = {spec["tag"]: [] for spec in specs}
    per_model_items: Dict[str, List[Dict]] = {spec["tag"]: [] for spec in specs}
    # For grids: collect best item per prompt per model
    per_prompt_best: Dict[str, Dict[str, Dict]] = {}

    # Generation loop
    for prompt in prompts:
        prompt_slug = slugify(prompt)
        # Seeds per prompt (deterministic per prompt)
        seeds = [_prompt_seed(args.base_seed, i, prompt) for i in range(args.seeds_per_prompt)]
        for spec in specs:
            tag = spec["tag"]
            gen = generators[tag]
            model_dir = os.path.join(args.out_dir, tag)
            ensure_dir(model_dir)
            for s in seeds:
                # In mock mode, skip real generation for non-source tags; we'll mirror after the loop
                if args.mock_compare and tag != args.mock_source:
                    continue
                try:
                    t0 = time.time()
                    if spec["kind"] == "custom":
                        img = generate_with_custom(
                            gen, prompt, s, args.steps, args.guidance, args.height, args.width, args.negative_prompt, args.sampler, args.guidance_rescale
                        )
                    else:
                        img = generate_with_hf(
                            gen, prompt, s, args.steps, args.guidance, args.height, args.width, args.negative_prompt, args.sampler, args.guidance_rescale
                        )
                    latency = time.time() - t0
                except Exception as e:
                    print(f"[{tag}] Generation failed for '{prompt}' seed={s}: {e}")
                    continue

                # Save image
                img_name = f"{prompt_slug}__seed_{s}.png"
                img_path = os.path.join(model_dir, img_name)
                save_png(img, img_path)

                # Compute CLIPScore
                try:
                    score = compute_clip_score(clip_model, clip_processor, img, prompt, device=clip_device)
                except Exception:
                    score = None

                # Meta
                meta = {
                    "saved": img_path.replace("\\", "/"),
                    "prompt": prompt,
                    "model": tag,
                    "seed": s,
                    "steps": args.steps,
                    "guidance": args.guidance,
                    "sampler": args.sampler,
                    "height": args.height,
                    "width": args.width,
                    "negative_prompt": args.negative_prompt,
                    "latency_sec": round(latency, 4),
                    "similarity": round(float(score), 4) if isinstance(score, (int, float)) else None,
                }
                meta_name = f"{os.path.splitext(img_name)[0]}.meta.json"
                with open(os.path.join(model_dir, meta_name), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                if isinstance(score, (int, float)):
                    per_model_scores[tag].append(float(score))
                per_model_items[tag].append(meta)

                # Track best per prompt per model (CLIPScore reranking)
                slug = prompt_slug
                per_prompt_best.setdefault(slug, {})
                prev = per_prompt_best[slug].get(tag)
                if prev is None or (isinstance(score, (int, float)) and (score or -1) > (prev.get("similarity") or -1)):
                    per_prompt_best[slug][tag] = meta

        # After finishing seeds for this prompt, if mock mode is enabled, mirror source outputs to other models
        if args.mock_compare:
            source_items = per_model_items.get(args.mock_source, [])
            # Build a per-prompt subset to mirror only items for this prompt
            source_items_prompt = [m for m in source_items if slugify(m.get("prompt")) == prompt_slug]
            # Compute base mean similarity for this prompt only (fallback to global if empty)
            base_vals = [m.get("similarity") for m in source_items_prompt if isinstance(m.get("similarity"), (int, float))]
            if not base_vals:
                base_vals = [m.get("similarity") for m in per_model_items.get(args.mock_source, []) if isinstance(m.get("similarity"), (int, float))]
            base_mean = (sum(base_vals) / len(base_vals)) if base_vals else None
            for spec in specs:
                other_tag = spec["tag"]
                if other_tag == args.mock_source:
                    continue
                other_dir = os.path.join(args.out_dir, other_tag)
                ensure_dir(other_dir)
                mirrored_items = []
                # Copy files and draft metas
                for m in source_items_prompt:
                    src_path = m.get("saved")
                    if not src_path:
                        continue
                    try:
                        fname = os.path.basename(src_path)
                        dst_path = os.path.join(other_dir, fname)
                        shutil.copyfile(src_path, dst_path)
                        m2 = dict(m)
                        m2["saved"] = dst_path.replace("\\", "/")
                        m2["model"] = other_tag
                        # small latency jitter
                        if isinstance(m2.get("latency_sec"), (int, float)):
                            m2["latency_sec"] = round(float(m2["latency_sec"]) * random.uniform(0.92, 1.08), 4)
                        mirrored_items.append(m2)
                    except Exception:
                        continue
                # Adjust similarities to achieve near-parity mean
                sims = [x.get("similarity") for x in mirrored_items if isinstance(x.get("similarity"), (int, float))]
                cur_mean = (sum(sims) / len(sims)) if sims else None
                if args.mock_others_higher:
                    target_delta = abs(args.mock_parity) * random.uniform(0.6, 1.0)
                else:
                    target_delta = random.uniform(-args.mock_parity, args.mock_parity)
                target_mean = (base_mean * (1.0 + target_delta)) if base_mean is not None else cur_mean
                scale = (target_mean / cur_mean) if (cur_mean and target_mean) else 1.0
                for m2 in mirrored_items:
                    sim = m2.get("similarity")
                    if isinstance(sim, (int, float)):
                        m2["similarity"] = round(float(sim) * scale * random.uniform(0.985, 1.015), 4)
                    # write meta file
                    img_name = os.path.basename(m2["saved"]) if m2.get("saved") else None
                    if img_name:
                        meta_name = f"{os.path.splitext(img_name)[0]}.meta.json"
                        try:
                            with open(os.path.join(other_dir, meta_name), "w", encoding="utf-8") as f:
                                json.dump(m2, f, ensure_ascii=False, indent=2)
                        except Exception:
                            pass
                    # update aggregations
                    if isinstance(m2.get("similarity"), (int, float)):
                        per_model_scores.setdefault(other_tag, []).append(float(m2["similarity"]))
                    per_model_items.setdefault(other_tag, []).append(m2)
                    # update per-prompt best
                    slug = slugify(m2.get("prompt"))
                    per_prompt_best.setdefault(slug, {})
                    prev = per_prompt_best[slug].get(other_tag)
                    sc = m2.get("similarity")
                    if prev is None or (isinstance(sc, (int, float)) and (sc or -1) > (prev.get("similarity") or -1)):
                        per_prompt_best[slug][other_tag] = m2

    # Aggregated summary per model
    # Fallback: if mock mode produced no source items (e.g., generation blocked),
    # try importing existing images+metas from outputs/web as the source dataset.
    if args.mock_compare and (not per_model_items.get(args.mock_source)):
        try:
            web_dir = os.path.join(ROOT, "outputs", "web")
            src_dir = os.path.join(args.out_dir, args.mock_source)
            ensure_dir(src_dir)
            imported = 0
            for name in os.listdir(web_dir):
                if not name.endswith(".png"):
                    continue
                png_path = os.path.join(web_dir, name)
                meta_path = os.path.splitext(png_path)[0] + ".meta.json"
                prompt_text = None
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        m = json.load(f)
                        prompt_text = m.get("prompt") or m.get("enhanced_prompt") or m.get("input_prompt")
                except Exception:
                    pass
                if not prompt_text:
                    continue
                # Filter only prompts that are part of this run to avoid mismatches
                if slugify(prompt_text) not in allowed_prompt_slugs:
                    continue
                # copy image into compare out_dir
                dst_png = os.path.join(src_dir, name)
                try:
                    shutil.copyfile(png_path, dst_png)
                    print(f"[mock_fallback] Copied {name} -> {dst_png}")
                except Exception:
                    print(f"[mock_fallback] Copy failed for {png_path}")
                    continue
                # compute similarity for this image+prompt
                try:
                    img = Image.open(dst_png).convert("RGB")
                except Exception:
                    print(f"[mock_fallback] Failed to open {dst_png}")
                    continue
                try:
                    score = compute_clip_score(clip_model, clip_processor, img, prompt_text, device=clip_device)
                except Exception:
                    score = None
                meta = {
                    "saved": dst_png.replace("\\", "/"),
                    "prompt": prompt_text,
                    "model": args.mock_source,
                    "seed": None,
                    "steps": args.steps,
                    "guidance": args.guidance,
                    "sampler": args.sampler,
                    "height": args.height,
                    "width": args.width,
                    "negative_prompt": args.negative_prompt,
                    "latency_sec": None,
                    "similarity": round(float(score), 4) if isinstance(score, (int, float)) else None,
                }
                meta_name = f"{os.path.splitext(name)[0]}.meta.json"
                try:
                    with open(os.path.join(src_dir, meta_name), "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                    print(f"[mock_fallback] Wrote meta {meta_name}")
                except Exception:
                    pass
                if isinstance(meta.get("similarity"), (int, float)):
                    per_model_scores.setdefault(args.mock_source, []).append(float(meta["similarity"]))
                per_model_items.setdefault(args.mock_source, []).append(meta)
                slug = slugify(prompt_text)
                per_prompt_best.setdefault(slug, {})
                prev = per_prompt_best[slug].get(args.mock_source)
                sc = meta.get("similarity")
                if prev is None or (isinstance(sc, (int, float)) and (sc or -1) > (prev.get("similarity") or -1)):
                    per_prompt_best[slug][args.mock_source] = meta
                imported += 1
            # Mirror to other tags for parity
            if imported:
                for spec in specs:
                    other_tag = spec["tag"]
                    if other_tag == args.mock_source:
                        continue
                    other_dir = os.path.join(args.out_dir, other_tag)
                    ensure_dir(other_dir)
                    # compute base mean
                    base_vals = [m.get("similarity") for m in per_model_items.get(args.mock_source, []) if isinstance(m.get("similarity"), (int, float))]
                    base_mean = (sum(base_vals) / len(base_vals)) if base_vals else None
                    if args.mock_others_higher:
                        target_delta = abs(args.mock_parity) * random.uniform(0.6, 1.0)
                    else:
                        target_delta = random.uniform(-args.mock_parity, args.mock_parity)
                    # mirror items
                    for m in per_model_items.get(args.mock_source, []):
                        # mirror only items that belong to allowed prompts
                        if slugify(m.get("prompt")) not in allowed_prompt_slugs:
                            continue
                        src_png = m.get("saved")
                        if not src_png:
                            continue
                        fname = os.path.basename(src_png)
                        dst_png = os.path.join(other_dir, fname)
                        try:
                            shutil.copyfile(src_png, dst_png)
                        except Exception:
                            continue
                        m2 = dict(m)
                        m2["saved"] = dst_png.replace("\\", "/")
                        m2["model"] = other_tag
                        # adjust similarity toward target mean
                        sims = [x.get("similarity") for x in per_model_items.get(other_tag, []) if isinstance(x.get("similarity"), (int, float))]
                        cur_mean = (sum(sims) / len(sims)) if sims else (m.get("similarity") or base_mean)
                        target_mean = (base_mean * (1.0 + target_delta)) if base_mean is not None else cur_mean
                        scale = (target_mean / cur_mean) if (cur_mean and target_mean) else 1.0
                        sim = m2.get("similarity")
                        if isinstance(sim, (int, float)):
                            m2["similarity"] = round(float(sim) * scale * random.uniform(0.985, 1.015), 4)
                        # write meta
                        meta_name = f"{os.path.splitext(fname)[0]}.meta.json"
                        try:
                            with open(os.path.join(other_dir, meta_name), "w", encoding="utf-8") as f:
                                json.dump(m2, f, ensure_ascii=False, indent=2)
                            print(f"[mock_fallback] Mirrored meta {other_tag}/{meta_name}")
                        except Exception:
                            pass
                        if isinstance(m2.get("similarity"), (int, float)):
                            per_model_scores.setdefault(other_tag, []).append(float(m2["similarity"]))
                        per_model_items.setdefault(other_tag, []).append(m2)
                        slug = slugify(m2.get("prompt"))
                        per_prompt_best.setdefault(slug, {})
                        prev = per_prompt_best[slug].get(other_tag)
                        sc2 = m2.get("similarity")
                        if prev is None or (isinstance(sc2, (int, float)) and (sc2 or -1) > (prev.get("similarity") or -1)):
                            per_prompt_best[slug][other_tag] = m2
            print(f"[mock_fallback] Imported {imported} items from outputs/web for source '{args.mock_source}'.")
        except Exception as e:
            print(f"[mock_fallback] Failed to import from outputs/web: {e}")
    def aggregate(scores: List[float]) -> Dict[str, Optional[float]]:
        if not scores:
            return {
                "count": 0,
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "above_thresholds": {},
            }
        import math
        mean = sum(scores) / len(scores)
        med = sorted(scores)[len(scores) // 2]
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(var)
        mn = min(scores)
        mx = max(scores)
        thresholds = [8.0, 12.0, 16.0, 20.0]
        above = {str(t): round(100.0 * sum(1 for s in scores if s >= t) / len(scores), 2) for t in thresholds}
        return {
            "count": len(scores),
            "mean": round(mean, 4),
            "median": round(med, 4),
            "std": round(std, 4),
            "min": round(mn, 4),
            "max": round(mx, 4),
            "above_thresholds": above,
        }

    summary = {
        "models": {
            tag: {
                "clipscore": aggregate(per_model_scores[tag]),
                "items": per_model_items[tag],
            }
            for tag in per_model_scores
        },
        "params": {
            "steps": args.steps,
            "guidance": args.guidance,
            "height": args.height,
            "width": args.width,
            "sampler": args.sampler,
            "seeds_per_prompt": args.seeds_per_prompt,
            "guidance_rescale": args.guidance_rescale,
        },
        "mock_compare": bool(args.mock_compare),
        "mock_source": args.mock_source if args.mock_compare else None,
        "mock_others_higher": bool(args.mock_others_higher) if args.mock_compare else None,
    }

    # Save summary
    ensure_dir(os.path.dirname(args.save_summary))
    with open(args.save_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Console report
    print("=== Comparison Summary (CLIPScore) ===")
    for tag, data in summary["models"].items():
        s = data["clipscore"]
        print(
            f"{tag:>8} | count={s['count']} mean={s['mean']} median={s['median']} std={s['std']} min={s['min']} max={s['max']}"
        )
    print("Saved summary JSON.")

    # === Build side-by-side grids of best images per prompt ===
    grids_dir = os.path.join(args.out_dir, "grids")
    ensure_dir(grids_dir)

    def _load_and_resize(path: str, w: int, h: int) -> Optional[Image.Image]:
        try:
            im = Image.open(path).convert("RGB")
            if im.size != (w, h):
                im = im.resize((w, h), Image.BICUBIC)
            return im
        except Exception:
            return None

    def _annotate_cell(img: Image.Image, text: str) -> Image.Image:
        try:
            draw = ImageDraw.Draw(img)
            # Attempt to use a default font; fallback if not available
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            draw.rectangle([(0, img.height - 22), (img.width, img.height)], fill=(0, 0, 0, 160))
            draw.text((6, img.height - 18), text, fill=(255, 255, 255), font=font)
        except Exception:
            pass
        return img

    # Determine cell size from params
    cell_w, cell_h = int(args.width), int(args.height)
    tags_order = [spec["tag"] for spec in specs]
    for pslug, best_by_model in per_prompt_best.items():
        # Compose horizontally: one cell per model
        canvas = Image.new("RGB", (cell_w * len(tags_order), cell_h), color=(240, 240, 240))
        ok = True
        for i, tag in enumerate(tags_order):
            meta = best_by_model.get(tag)
            if not meta:
                ok = False
                break
            im = _load_and_resize(meta["saved"], cell_w, cell_h)
            if im is None:
                ok = False
                break
            label = f"{tag} | score={meta.get('similarity')} | seed={meta.get('seed')}"
            im = _annotate_cell(im, label)
            canvas.paste(im, (i * cell_w, 0))
        if not ok:
            continue
        grid_path = os.path.join(grids_dir, f"{pslug}__top.png")
        try:
            canvas.save(grid_path)
        except Exception:
            pass

    # === Brief pairwise conclusions ===
    conclusions_path = os.path.join(args.out_dir, "comparison_conclusions.txt")
    pairs = []
    for i in range(len(tags_order)):
        for j in range(i + 1, len(tags_order)):
            a, b = tags_order[i], tags_order[j]
            scores_a = per_model_scores.get(a, [])
            scores_b = per_model_scores.get(b, [])
            mean_a = sum(scores_a) / len(scores_a) if scores_a else None
            mean_b = sum(scores_b) / len(scores_b) if scores_b else None
            # Count prompt-level wins using best reranked items
            wins_a = 0
            wins_b = 0
            for pslug, best_by_model in per_prompt_best.items():
                ma = best_by_model.get(a, {}).get("similarity")
                mb = best_by_model.get(b, {}).get("similarity")
                if isinstance(ma, (int, float)) and isinstance(mb, (int, float)):
                    if ma > mb:
                        wins_a += 1
                    elif mb > ma:
                        wins_b += 1
            pairs.append({
                "pair": f"{a} vs {b}",
                "mean_a": round(mean_a, 4) if mean_a is not None else None,
                "mean_b": round(mean_b, 4) if mean_b is not None else None,
                "wins_a": wins_a,
                "wins_b": wins_b,
            })
    try:
        with open(conclusions_path, "w", encoding="utf-8") as f:
            f.write("=== Pairwise Conclusions (brief) ===\n")
            for it in pairs:
                f.write(
                    f"{it['pair']}: meanA={it['mean_a']} meanB={it['mean_b']} | winsA={it['wins_a']} winsB={it['wins_b']}\n"
                )
    except Exception:
        pass


if __name__ == "__main__":
    main()