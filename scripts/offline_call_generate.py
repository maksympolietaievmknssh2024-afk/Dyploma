import os
import json
import sys
from base64 import b64decode


def main() -> None:
    # Read payload path
    if len(sys.argv) > 1:
        payload_path = sys.argv[1]
    else:
        payload_path = os.path.join("scripts", "payloads", "retriever_strict.json")
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Ensure pipeline mode to match server behavior
    os.environ.setdefault("USE_PIPELINE", "1")
    # Respect any external HF cache settings already configured by app.py

    # Import flask app and use test_client to call the endpoint internally
    root_dir = os.path.dirname(os.path.dirname(__file__))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from app import app  # noqa: E402

    out_dir = os.path.join("outputs", "web")
    os.makedirs(out_dir, exist_ok=True)

    with app.test_client() as client:
        resp = client.post("/generate", json=payload)
        try:
            result = resp.get_json(force=True)
        except Exception:
            result = None

    if not isinstance(result, dict):
        print("ERROR: invalid response type")
        with open(os.path.join(out_dir, "last_response.json"), "w", encoding="utf-8") as f:
            json.dump({"error": "Invalid response type"}, f, indent=2)
        sys.exit(1)

    if result.get("error"):
        print("ERROR:", result.get("error"))
        with open(os.path.join(out_dir, "last_response.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        sys.exit(1)

    image_b64 = result.get("image")
    if not image_b64:
        print("ERROR: no image data returned")
        with open(os.path.join(out_dir, "last_response.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        sys.exit(1)

    # Derive output name similar to CLI script
    prompt = str(payload.get("prompt", "image")).strip()
    sampler = str(payload.get("sampler", "auto"))
    name = "cli_{}_{}.png".format(
        "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in prompt.lower())[:60],
        sampler,
    )
    out_path = os.path.join(out_dir, name)
    with open(out_path, "wb") as f:
        f.write(b64decode(image_b64))

    meta = {
        "saved": out_path.replace("\\", "/"),
        "similarity": result.get("similarity"),
        "seed": result.get("used_seed"),
        "sampler": sampler,
        "sampler_used": result.get("sampler_used"),
        "guidance_rescale": result.get("guidance_rescale"),
        "prompt": prompt,
        "negative_preset": payload.get("negative_preset"),
        "generation_speed": payload.get("generation_speed"),
        "use_enhancement": payload.get("use_enhancement"),
        "num_candidates": payload.get("num_candidates"),
    }
    meta_name = "{}.meta.json".format(os.path.splitext(name)[0])
    with open(os.path.join(out_dir, meta_name), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", out_path)
    print("SamplerUsed:", result.get("sampler_used"))
    print("GuidanceRescale:", result.get("guidance_rescale"))
    print("Similarity:", result.get("similarity"))


if __name__ == "__main__":
    main()