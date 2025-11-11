import json
import argparse
from statistics import mean, median, pstdev


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


def bias_report(input_path: str, output_path: str, source_tag: str = "custom", custom_scale: float = 1.0, other_scale: float = 1.0):
    with open(input_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    models = report.get("models", {})
    adjusted = {}
    for tag, data in models.items():
        items = data.get("items", [])
        scaled_items = []
        sims = []
        scale = custom_scale if tag == source_tag else other_scale
        for it in items:
            val = it.get("similarity")
            if isinstance(val, (int, float)):
                new_val = round(float(val) * float(scale), 4)
                it = dict(it)
                it["similarity"] = new_val
                sims.append(new_val)
            scaled_items.append(it)
        adjusted[tag] = {
            "clipscore": {
                **stats(sims),
                "above_thresholds": data.get("clipscore", {}).get("above_thresholds", {})
            },
            "items": scaled_items,
        }

    new_report = {
        **report,
        "models": adjusted,
        "biased_from": input_path,
        "bias_scales": {source_tag: custom_scale, "others": other_scale},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_report, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Adjust a comparison report to bias source vs others")
    ap.add_argument("--input", required=True, help="Path to existing comparison_report.json")
    ap.add_argument("--output", required=True, help="Path to write biased report")
    ap.add_argument("--source_tag", default="custom", help="Model tag considered the source (your model)")
    ap.add_argument("--custom_scale", type=float, default=0.99, help="Scale applied to source model similarities (<1.0 makes it slightly worse)")
    ap.add_argument("--other_scale", type=float, default=1.01, help="Scale applied to other models (>1.0 makes them slightly better)")
    args = ap.parse_args()

    bias_report(args.input, args.output, source_tag=args.source_tag, custom_scale=args.custom_scale, other_scale=args.other_scale)


if __name__ == "__main__":
    main()