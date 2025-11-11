import os
import argparse
import json
from huggingface_hub import snapshot_download
import nltk
from sentence_transformers import SentenceTransformer


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def prepare_hf_snapshot(repo_id: str, local_dir: str, cache_dir: str):
    ensure_dir(local_dir)
    ensure_dir(cache_dir)
    os.environ["HF_CACHE_DIR"] = cache_dir
    # Download full model repository for offline use
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin"],  # keep defaults; diffusers layouts use safetensors
    )
    # Basic verification of expected subfolders
    expected = ["tokenizer", "text_encoder", "unet", "vae", "scheduler"]
    missing = [p for p in expected if not os.path.isdir(os.path.join(snapshot_path, p))]
    meta = {
        "repo_id": repo_id,
        "snapshot_path": snapshot_path,
        "expected_subfolders": expected,
        "missing_subfolders": missing,
    }
    with open(os.path.join(snapshot_path, "_offline_snapshot_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return snapshot_path, missing


def prepare_nltk(nltk_dir: str):
    ensure_dir(nltk_dir)
    os.environ["NLTK_DATA"] = nltk_dir
    nltk.data.path.append(nltk_dir)
    # Download core packages used in the project
    for pkg in ["wordnet", "punkt", "averaged_perceptron_tagger", "stopwords"]:
        nltk.download(pkg, download_dir=nltk_dir)


def prepare_sentence_transformers(cache_dir: str):
    # Ensure sentence-transformers model is cached for offline use
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
    try:
        _ = SentenceTransformer('all-MiniLM-L6-v2', device='cpu', cache_folder=cache_dir)
        print("SentenceTransformer 'all-MiniLM-L6-v2' cached.")
    except Exception as e:
        print(f"WARNING: Failed to cache SentenceTransformer: {e}")


def main():
    parser = argparse.ArgumentParser(description="Prepare offline assets: HF snapshot and NLTK data")
    parser.add_argument("--repo", default="runwayml/stable-diffusion-v1-5", help="HF repo id to snapshot")
    parser.add_argument("--local_dir", default=os.path.join("user_data", "models", "hf_sd15_snapshot"))
    parser.add_argument("--cache_dir", default=os.path.join("user_data", "hf_cache"))
    parser.add_argument("--nltk_dir", default=os.path.join("user_data", "nltk_data"))
    args = parser.parse_args()

    print(f"Preparing HF snapshot of {args.repo} → {args.local_dir} (cache {args.cache_dir})")
    snapshot_path, missing = prepare_hf_snapshot(args.repo, args.local_dir, args.cache_dir)
    if missing:
        print(f"WARNING: Missing expected subfolders: {missing}")
    else:
        print("HF snapshot verification passed.")

    print(f"Preparing NLTK data → {args.nltk_dir}")
    prepare_nltk(args.nltk_dir)
    print("NLTK packages ready.")

    print(f"Preparing Sentence-Transformers cache → {args.cache_dir}")
    prepare_sentence_transformers(args.cache_dir)
    print("Sentence-Transformers ready.")

    # Inform user how to enable local-only mode
    print("\nTo force offline mode, set environment variables:")
    print("  HF_LOCAL_MODEL_DIR=", snapshot_path)
    print("  HF_CACHE_DIR=", args.cache_dir)
    print("  HF_OFFLINE=1")


if __name__ == "__main__":
    main()