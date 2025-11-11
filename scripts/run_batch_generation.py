import os
import re
import json
import time
import base64
from typing import List, Dict

try:
    import requests
except ImportError:
    raise SystemExit("Please install requests: pip install requests")


BASE_URL = os.environ.get("WEB_URL", "http://127.0.0.1:5001")
GENERATE_URL = f"{BASE_URL}/generate"
OUTPUT_DIR = os.path.join("outputs", "web")


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\-_. ]+", "", text)
    text = text.strip().lower()
    text = text.replace(" ", "_")
    return text[:60] if len(text) > 60 else text


def save_image_png(base64_png: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(base64.b64decode(base64_png))


def run_batch(items: List[Dict]) -> List[Dict]:
    results = []
    ts = time.strftime("%Y%m%d_%H%M%S")
    for idx, it in enumerate(items, start=1):
        payload = {
            "prompt": it["prompt"],
            "negative_prompt": it.get("negative_prompt", ""),
            "negative_preset": it.get("negative_preset", "standard"),
            "generation_speed": it.get("generation_speed", "balanced"),
            "seed": it.get("seed"),
            "num_candidates": it.get("num_candidates", 3),
            "sampler": it.get("sampler", "dpmsolver"),
            "guidance_rescale": it.get("guidance_rescale", 0.3),
            "use_enhancement": it.get("use_enhancement", False),
        }

        try:
            resp = requests.post(GENERATE_URL, json=payload, timeout=600)
            data = resp.json()
        except Exception as e:
            results.append({"error": str(e), "payload": payload})
            continue

        if data.get("error"):
            results.append({"error": data["error"], "payload": payload})
            continue

        # File naming
        slug = slugify(it["prompt"]) or f"item_{idx}"
        sampler_used = data.get("sampler_used") or payload["sampler"]
        img_name = f"batch_{ts}_{slug}_{sampler_used}.png"
        img_path = os.path.join(OUTPUT_DIR, img_name)
        save_image_png(data["image"], img_path)

        meta = {
            "file": img_path,
            "prompt": it["prompt"],
            "negative_prompt": payload["negative_prompt"],
            "negative_preset": payload["negative_preset"],
            "generation_speed": payload["generation_speed"],
            "seed": data.get("used_seed", payload.get("seed")),
            "num_candidates": payload["num_candidates"],
            "sampler": payload["sampler"],
            "sampler_used": data.get("sampler_used"),
            "guidance_rescale": data.get("guidance_rescale", payload.get("guidance_rescale")),
            "use_enhancement": payload["use_enhancement"],
            "similarity": data.get("similarity"),
        }
        results.append(meta)

    # Write aggregated metadata
    agg_path = os.path.join(OUTPUT_DIR, f"batch_{ts}_summary.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump({"items": results, "base_url": BASE_URL}, f, ensure_ascii=False, indent=2)

    print(f"Saved summary to: {agg_path}")
    for r in results:
        if "file" in r:
            print(f"✓ {r['file']} | sampler={r.get('sampler_used') or r.get('sampler')} | seed={r.get('seed')} | sim={r.get('similarity')}")
        else:
            print(f"× Error: {r.get('error')} | payload={r.get('payload')}")
    return results


def main():
    negative = "watermark, logo, text overlay, captions, copyright"
    base_seed = 123450

    batch_items = [
        {
            "prompt": "sunny beach with palm trees, clear sky, turquoise water",
            "negative_prompt": negative,
            "sampler": "dpmsolver",
            "seed": base_seed + 1,
        },
        {
            "prompt": "portrait of a smiling person in natural light, shallow depth of field",
            "negative_prompt": negative + ", extra fingers, deformed, watermark",
            "sampler": "pndm",
            "seed": base_seed + 2,
        },
        {
            "prompt": "golden retriever running on green grass, motion blur, sunny day",
            "negative_prompt": negative + ", text, logo",
            "sampler": "euler_a",
            "seed": base_seed + 3,
        },
        {
            "prompt": "vintage camera on a wooden table, soft studio lighting, high detail",
            "negative_prompt": negative,
            "sampler": "dpmsolver",
            "seed": base_seed + 4,
        },
    ]

    # Defaults for all items
    for it in batch_items:
        it.setdefault("negative_preset", "standard")
        it.setdefault("generation_speed", "balanced")
        it.setdefault("num_candidates", 3)
        it.setdefault("guidance_rescale", 0.3)
        it.setdefault("use_enhancement", False)

    run_batch(batch_items)


if __name__ == "__main__":
    main()