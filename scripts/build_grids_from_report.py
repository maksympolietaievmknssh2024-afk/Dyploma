import os
import json
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def slugify(text: str) -> str:
    import re
    text = re.sub(r"[^a-zA-Z0-9\-_. ]+", "", str(text))
    text = text.strip().lower()
    text = text.replace(" ", "_")
    return text[:80] if len(text) > 80 else text


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
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.rectangle([(0, img.height - 22), (img.width, img.height)], fill=(0, 0, 0, 160))
        draw.text((6, img.height - 18), text, fill=(255, 255, 255), font=font)
    except Exception:
        pass
    return img


def build_grids(report_path: str, out_dir: str, cell_w: int = 512, cell_h: int = 512) -> None:
    # Load report
    with open(report_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    models = summary.get("models", {})
    tags_order = list(models.keys())
    if len(tags_order) < 2:
        raise SystemExit("Report must contain at least two models to build grids")

    grids_dir = os.path.join(out_dir, "grids")
    ensure_dir(grids_dir)

    # Collect best-by-prompt across models
    per_prompt_best: Dict[str, Dict[str, Dict]] = {}
    for tag, data in models.items():
        items: List[Dict] = data.get("items", [])
        for it in items:
            prompt = it.get("prompt")
            sim = it.get("similarity")
            if not prompt or sim is None:
                # Skip missing data
                continue
            pslug = slugify(prompt)
            per_prompt_best.setdefault(pslug, {})
            prev = per_prompt_best[pslug].get(tag)
            # Keep the highest similarity per prompt per model
            if prev is None or (prev.get("similarity") or -1) < sim:
                per_prompt_best[pslug][tag] = it

    # Compose grids for prompts that have entries for all models
    built = 0
    for pslug, best_by_model in per_prompt_best.items():
        if not all(tag in best_by_model for tag in tags_order):
            continue
        canvas = Image.new("RGB", (cell_w * len(tags_order), cell_h), color=(240, 240, 240))
        ok = True
        for i, tag in enumerate(tags_order):
            meta = best_by_model.get(tag)
            im = _load_and_resize(meta.get("saved"), cell_w, cell_h)
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
            built += 1
        except Exception:
            pass

    # Write brief pairwise conclusions based on means and wins
    conclusions_path = os.path.join(out_dir, "comparison_conclusions.txt")
    pairs = []
    # Compute means from report
    means = {tag: models[tag].get("clipscore", {}).get("mean") for tag in tags_order}
    wins = {tag: 0 for tag in tags_order}
    for pslug, best_by_model in per_prompt_best.items():
        # Compare first two tags only for wins (extendable to n-way)
        if len(tags_order) >= 2:
            a, b = tags_order[0], tags_order[1]
            sa = best_by_model.get(a, {}).get("similarity")
            sb = best_by_model.get(b, {}).get("similarity")
            if isinstance(sa, (int, float)) and isinstance(sb, (int, float)):
                if sa > sb:
                    wins[a] += 1
                elif sb > sa:
                    wins[b] += 1

    if len(tags_order) >= 2:
        a, b = tags_order[0], tags_order[1]
        pairs.append({
            "pair": f"{a} vs {b}",
            "mean_a": means.get(a),
            "mean_b": means.get(b),
            "wins_a": wins.get(a, 0),
            "wins_b": wins.get(b, 0),
        })

    try:
        with open(conclusions_path, "w", encoding="utf-8") as f:
            f.write("=== Pairwise Conclusions (brief) ===\n")
            for it in pairs:
                f.write(
                    f"{it['pair']}: meanA={it['mean_a']} meanB={it['mean_b']} | winsA={it['wins_a']} winsB={it['wins_b']}\n"
                )
            f.write(f"\nBuilt {built} grids in: {grids_dir}\n")
    except Exception:
        pass


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build side-by-side grids from an existing comparison report")
    ap.add_argument("--report", default=os.path.join("outputs", "evaluation", "comparison_report.json"))
    ap.add_argument("--out_dir", default=os.path.join("outputs", "compare"))
    ap.add_argument("--cell_w", type=int, default=512)
    ap.add_argument("--cell_h", type=int, default=512)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    build_grids(args.report, args.out_dir, cell_w=args.cell_w, cell_h=args.cell_h)


if __name__ == "__main__":
    main()