import os
import json
import argparse
from datetime import datetime
from statistics import mean, median, pstdev


def collect_meta(model_dir: str):
    sims = []
    lats = []
    items = []
    for name in os.listdir(model_dir):
        if not name.endswith(".meta.json"):
            continue
        path = os.path.join(model_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sim = data.get("similarity")
            lat = data.get("latency_sec")
            if isinstance(sim, (int, float)):
                sims.append(float(sim))
            if isinstance(lat, (int, float)):
                lats.append(float(lat))
            items.append({
                "file": data.get("saved") or path,
                "prompt": data.get("prompt"),
                "similarity": sim,
                "latency_sec": lat,
            })
        except Exception:
            # Skip malformed entries
            continue
    return items, sims, lats


def stats(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "std": round(pstdev(values), 4) if len(values) > 1 else 0.0,
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def summarize(compare_dir: str):
    models_summary = {}
    model_dirs = [d for d in os.listdir(compare_dir) if os.path.isdir(os.path.join(compare_dir, d))]
    for tag in sorted(model_dirs):
        mdir = os.path.join(compare_dir, tag)
        items, sims, lats = collect_meta(mdir)
        models_summary[tag] = {
            "similarity": stats(sims),
            "latency_sec": stats(lats),
            "items": items,
        }
    return models_summary


def main():
    ap = argparse.ArgumentParser(description="Summarize compare outputs into JSON report")
    ap.add_argument("--compare-dir", default=os.path.join("outputs", "compare"))
    ap.add_argument("--output", default=os.path.join("outputs", "evaluation", "comparison_report.json"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary = {
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_dir": args.compare_dir,
        "models": summarize(args.compare_dir),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Print a compact overview to stdout
    for tag, data in summary["models"].items():
        sim = data["similarity"]["mean"]
        lat = data["latency_sec"]["mean"]
        print(f"{tag}: mean_similarity={sim} | mean_latency={lat}s | n={data['similarity']['count']}")


if __name__ == "__main__":
    main()