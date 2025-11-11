import argparse
import os
import shutil
from pathlib import Path

TARGETS = [
    "output",
    "enhanced_output",
    "semantic_model_output",
    "seed_test_results",
    "test_results",
    os.path.join("data", "cifar-10-batches-py"),
    os.path.join("data", "cifar-10-python.tar.gz"),
    os.path.join("data", "processed"),
]

# File patterns to remove (heavy checkpoints, logs)
FILE_PATTERNS = [
    "*.pth",
    "*.pt",
    "*.ckpt",
    "*.safetensors",
    "training.log",
]

def get_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return round(path.stat().st_size / (1024 * 1024), 2)
    total = 0
    for p in path.rglob('*'):
        if p.is_file():
            total += p.stat().st_size
    return round(total / (1024 * 1024), 2)

def remove_path(path: Path, dry_run: bool) -> None:
    size_mb = get_size_mb(path)
    if not path.exists():
        print(f"SKIP: {path} (not found)")
        return
    action = "WOULD REMOVE" if dry_run else "REMOVING"
    print(f"{action}: {path} ({size_mb} MB)")
    if dry_run:
        return
    try:
        if path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
        print(f"DONE: {path}")
    except Exception as e:
        print(f"ERROR removing {path}: {e}")

def remove_files_by_pattern(root: Path, pattern: str, dry_run: bool) -> float:
    removed_total = 0.0
    for p in root.rglob(pattern):
        size = get_size_mb(p)
        removed_total += size
        action = "WOULD REMOVE" if dry_run else "REMOVING"
        print(f"{action}: {p} ({size} MB)")
        if dry_run:
            continue
        try:
            p.unlink()
            print(f"DONE: {p}")
        except Exception as e:
            print(f"ERROR removing {p}: {e}")
    return removed_total

def main():
    parser = argparse.ArgumentParser(description="Clean up heavy artifacts from workspace")
    parser.add_argument("--force", action="store_true", help="Actually remove files/directories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")
    args = parser.parse_args()

    # Default to dry-run if neither flag is provided
    dry_run = args.dry_run or (not args.force)

    root = Path(__file__).resolve().parents[1]
    print(f"Workspace root: {root}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'FORCE-DELETE'}\n")

    total_mb = 0.0
    for target in TARGETS:
        path = (root / target).resolve()
        size = get_size_mb(path)
        total_mb += size
        remove_path(path, dry_run=dry_run)

    # Remove heavy files by patterns
    for pattern in FILE_PATTERNS:
        total_mb += remove_files_by_pattern(root, pattern, dry_run=dry_run)

    print(f"\nTOTAL SIZE targeted: {round(total_mb, 2)} MB")
    if dry_run:
        print("Run with --force to actually delete these artifacts.")

if __name__ == "__main__":
    main()