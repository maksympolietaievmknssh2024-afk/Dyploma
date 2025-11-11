#!/usr/bin/env python3
"""
Visualize semantic test results: confusion matrix, per-word accuracy,
summary charts, and result tables.

Reads semantic_test_results/semantic_test_results.json and saves figures to
the same directory.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_confusion_matrix(disambig_tests, out_dir: str):
    """
    Build a 2x2 confusion matrix for disambiguation: pred vs true same/different.
    """
    cm = np.zeros((2, 2), dtype=int)  # rows: pred [same, diff], cols: true [same, diff]

    for t in disambig_tests:
        if 'should_be_different' not in t or 'correctly_distinguished' not in t:
            continue
        true_diff = bool(t.get('should_be_different', True))
        correctly = bool(t.get('correctly_distinguished', False))

        # Predicted different only if it should be different and was correctly distinguished
        pred_diff = true_diff and correctly
        pred_same = not pred_diff

        r = 1 if pred_diff else 0
        c = 1 if true_diff else 0
        cm[r, c] += 1

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["True Same", "True Different"]) 
    plt.yticks([0, 1], ["Pred Same", "Pred Different"]) 
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.title("Disambiguation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "disambiguation_confusion_matrix.png"), bbox_inches='tight')
    plt.close()


def save_per_word_accuracy(disambig_tests, out_dir: str):
    per_word = defaultdict(lambda: {"total": 0, "correct": 0})
    for t in disambig_tests:
        w = t.get("word", "unknown")
        per_word[w]["total"] += 1
        per_word[w]["correct"] += int(bool(t.get("correctly_distinguished", False)))

    labels = list(per_word.keys())
    acc = [per_word[w]["correct"] / per_word[w]["total"] if per_word[w]["total"] else 0.0 for w in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, acc, color="#3b82f6")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Per-Word Disambiguation Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_word_accuracy.png"), bbox_inches='tight')
    plt.close()


def save_disambiguation_summary(disambig_tests, out_dir: str):
    total = len(disambig_tests)
    correct = sum(1 for t in disambig_tests if bool(t.get("correctly_distinguished", False)))
    incorrect = total - correct

    plt.figure(figsize=(5, 4))
    plt.bar(["Correct", "Incorrect"], [correct, incorrect], color=["#10b981", "#ef4444"])
    plt.ylabel("Number of Tests")
    plt.title("Disambiguation Summary")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "disambiguation_summary.png"), bbox_inches='tight')
    plt.close()


def save_results_table(disambig_tests, out_dir: str):
    """Save a CSV table with key fields for each disambiguation test."""
    cols = ["word", "prompt_a", "prompt_b", "similarity", "should_be_different", "correctly_distinguished"]
    path = os.path.join(out_dir, "disambiguation_results_table.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for t in disambig_tests:
            row = [
                str(t.get("word", "")),
                str(t.get("prompt_a", "")).replace("\n", " "),
                str(t.get("prompt_b", "")).replace("\n", " "),
                str(t.get("similarity", "")),
                str(t.get("should_be_different", "")),
                str(t.get("correctly_distinguished", "")),
            ]
            f.write(",".join(row) + "\n")


def build_gallery(generated, out_dir: str, name: str = "gallery.png", max_images: int = 6):
    """
    Build a simple grid gallery from generated image entries.
    Expects entries with keys: image_path and (optionally) prompt.
    """
    entries = [g for g in generated if isinstance(g, dict) and os.path.exists(g.get("image_path", ""))]
    entries = entries[:max_images]
    if not entries:
        return

    images = []
    captions = []
    for e in entries:
        try:
            img = Image.open(e["image_path"]).convert("RGB")
            images.append(img)
            captions.append(str(e.get("prompt", os.path.basename(e["image_path"])) ))
        except Exception:
            continue

    if not images:
        return

    # Create a simple grid: 3 columns
    cols = 3
    rows = int(np.ceil(len(images) / cols))
    w, h = images[0].size
    pad = 10
    # Reserve bottom caption area per image
    caption_h = 40
    grid = Image.new("RGB", (cols * (w + pad) - pad, rows * (h + caption_h + pad) - pad), color=(255, 255, 255))

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = c * (w + pad)
        y = r * (h + caption_h + pad)
        grid.paste(img, (x, y))
        # Caption: draw basic text using PIL
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(grid)
            caption = captions[idx][:60]
            draw.text((x + 5, y + h + 5), caption, fill=(0, 0, 0))
        except Exception:
            pass

    grid.save(os.path.join(out_dir, name))


def main():
    in_path = os.path.join("semantic_test_results", "semantic_test_results.json")
    if not os.path.exists(in_path):
        print(f"File not found: {in_path}. Run test_semantic_understanding.py first.")
        return

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = os.path.dirname(in_path)
    ensure_dir(out_dir)

    disambig_tests = data.get("disambiguation_tests", [])
    generated_images = data.get("generated_images", [])

    # Figures and tables
    save_confusion_matrix(disambig_tests, out_dir)
    save_per_word_accuracy(disambig_tests, out_dir)
    save_disambiguation_summary(disambig_tests, out_dir)
    save_results_table(disambig_tests, out_dir)

    # Gallery from generated images
    build_gallery(generated_images, out_dir, name="generated_gallery.png", max_images=6)

    # Optional: print summary
    total = len(disambig_tests)
    correct = sum(1 for t in disambig_tests if bool(t.get("correctly_distinguished", False)))
    acc = (correct / total) if total else 0.0
    print(f"Disambiguation tests: {total}, accuracy: {acc:.2f}")
    print(f"Saved figures and table to: {out_dir}")


if __name__ == "__main__":
    main()