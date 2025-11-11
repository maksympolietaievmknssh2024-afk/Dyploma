import os
import argparse
import json

# Allow running from scripts/ while importing project modules
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluation.clip_score import evaluate_from_meta, save_report


def parse_args():
    p = argparse.ArgumentParser(description="Compute CLIPScore quality metrics for generated images")
    p.add_argument("--meta_dir", default=os.path.join("outputs", "web"), help="Directory with *.meta.json files")
    p.add_argument("--out", default=os.path.join("outputs", "evaluation", "clipscore_report.json"), help="Path to save aggregated report JSON")
    p.add_argument("--limit", type=int, default=None, help="Optional limit on number of meta files to evaluate")
    p.add_argument("--print_top", type=int, default=0, help="Print N top-scoring samples")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.meta_dir):
        raise SystemExit(f"Meta directory not found: {args.meta_dir}")

    report = evaluate_from_meta(args.meta_dir, limit=args.limit)
    save_report(report, args.out)

    # Console summary
    print("=== CLIPScore Quality Report ===")
    print(f"Samples: {report['count']}")
    if report['count'] == 0:
        print("No valid meta entries found.")
        return
    print(f"Mean: {report['mean']}")
    print(f"Median: {report['median']}")
    print(f"Std: {report['std']}")
    print(f"Min/Max: {report['min']} / {report['max']}")
    print("Above thresholds (%):", json.dumps(report.get('above_thresholds', {}), ensure_ascii=False))

    if args.print_top and isinstance(report.get('scores'), list):
        top = sorted(report['scores'], key=lambda x: x['score'], reverse=True)[:args.print_top]
        print("\nTop samples:")
        for item in top:
            print(f"  {item['score']:6.3f} | {item['file']}")

    print(f"\nSaved report → {args.out}")


if __name__ == "__main__":
    main()