import os
import argparse
import torch
from PIL import Image
import io
import base64
import time
import logging
from flask import Flask, render_template, request, jsonify
from transformers import CLIPModel, CLIPProcessor
import re

# Import our custom modules
from models.diffusion_model import ImageGenerationModel
from models.pipeline_generator import PretrainedPipelineGenerator
from utils.text_processing import TextProcessor
from src.training.auto_retraining_system import AutoRetrainingSystem

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Initialize the model and text processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
text_processor = TextProcessor()
clip_model = None
clip_processor = None

# Initialize auto-retraining system
auto_retraining = AutoRetrainingSystem(
    data_collection_dir="user_data",
    min_samples_for_retraining=50,  # Менше зразків для тестування
    retraining_interval_hours=12,   # Частіша перевірка
    max_dataset_size=5000
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- NEW: Ensure runtime dirs and environment defaults ---
def ensure_runtime_dirs_and_env():
    # HF cache directory
    hf_cache_dir = os.environ.get("HF_CACHE_DIR") or os.path.join("user_data", "hf_cache")
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_CACHE_DIR"] = hf_cache_dir

    # Optional: Hugging Face offline toggle remains user-controlled (HF_OFFLINE)

    # Local snapshot directory for HF model (if user prepares it)
    hf_local_model_dir = os.environ.get("HF_LOCAL_MODEL_DIR", "")
    if hf_local_model_dir:
        # If set, ensure it exists to avoid silent failures
        try:
            os.makedirs(hf_local_model_dir, exist_ok=True)
        except Exception:
            pass

    # NLTK data directory
    nltk_dir = os.environ.get("NLTK_DATA") or os.path.join("user_data", "nltk_data")
    os.makedirs(nltk_dir, exist_ok=True)
    os.environ["NLTK_DATA"] = nltk_dir
    # Enforce English prompts by default unless explicitly disabled
    if os.environ.get("ENFORCE_ENGLISH") is None:
        os.environ["ENFORCE_ENGLISH"] = "1"
    # Prefer pretrained Diffusers pipeline by default for stable results
    if os.environ.get("USE_PIPELINE") is None:
        os.environ["USE_PIPELINE"] = "1"
    logger.info(
        f"Runtime dirs set: HF_CACHE_DIR={hf_cache_dir}, NLTK_DATA={nltk_dir}, HF_LOCAL_MODEL_DIR={hf_local_model_dir or 'None'}, "
        f"ENFORCE_ENGLISH={os.environ.get('ENFORCE_ENGLISH')}, USE_PIPELINE={os.environ.get('USE_PIPELINE')}"
    )

# Initialize runtime env immediately
ensure_runtime_dirs_and_env()
# --- END NEW ---
# --- NEW: Ensure model loads when using `flask run` ---
# --- NEW: Replace before_first_request with lazy initializer ---
def ensure_model_initialized():
    global model
    if model is None:
        model_path = os.environ.get("MODEL_PATH", None)
        use_pipeline = os.environ.get("USE_PIPELINE", "0") in ("1", "true", "True")
        hf_cache_dir = os.environ.get("HF_CACHE_DIR", None)
        try:
            load_model(model_path, use_pipeline=use_pipeline, hf_cache_dir=hf_cache_dir)
            logger.info("Model initialized lazily on first request")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
    # Керуємо моніторингом: вимикаємо лише у режимі пайплайна
    try:
        use_pipeline = os.environ.get("USE_PIPELINE", "0") in ("1", "true", "True")
        if use_pipeline:
            auto_retraining.stop_monitoring_process()
            logger.info("Auto-retraining monitoring stopped in pipeline mode")
    except Exception as e:
        logger.error(f"Failed to adjust auto-retraining monitoring: {e}")
# --- END NEW ---

# --- NEW: CLIP reranker lazy initializer ---
def get_clip_reranker():
    global clip_model, clip_processor
    if clip_model is None or clip_processor is None:
        try:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info(f"Initialized CLIP reranker on device: {device}")
        except Exception as e:
            logger.error(f"Failed to init CLIP reranker: {e}")
            raise
    return clip_model, clip_processor
# --- END NEW ---

@app.route('/')
def index():
    return render_template('index.html')

# --- NEW: Language enforcement helper ---
def is_english_text(text: str) -> bool:
    if not text:
        return False
    # Reject Cyrillic characters (U+0400–U+04FF)
    if re.search(r"[\u0400-\u04FF]", text):
        return False
    # Ensure prompt has Latin letters
    return re.search(r"[A-Za-z]", text) is not None
# --- END NEW ---

# --- NEW: Helper to detect pipeline mode ---
def is_pipeline_mode() -> bool:
    return os.environ.get("USE_PIPELINE", "0") in ("1", "true", "True")
# --- END NEW ---

# --- NEW: Toggle to enforce English prompts regardless of mode ---
def should_enforce_english() -> bool:
    """Return True if English-only prompts should be enforced.
    Behavior:
    - If env ENFORCE_ENGLISH is '1/true/yes' -> enforce
    - If env ENFORCE_ENGLISH is '0/false/no' -> do not enforce
    - Otherwise (auto) -> enforce only in pipeline mode
    """
    val = os.environ.get("ENFORCE_ENGLISH", "auto").strip().lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return is_pipeline_mode()

@app.route('/generate', methods=['POST'])
def generate():
    ensure_model_initialized()
    data = request.get_json()
    prompt = data.get('prompt', '')
    negative_prompt = data.get('negative_prompt', '')
    negative_preset = data.get('negative_preset', 'standard')
    # Краще за замовчуванням: збалансована якість
    generation_speed = data.get('generation_speed', 'balanced')
    seed = data.get('seed', None)
    # Серверні дефолти для кращої якості
    num_candidates = int(data.get('num_candidates', 5)) if data.get('num_candidates') is not None else 5
    num_candidates = max(1, min(8, num_candidates))
    
    # Увімкнути автопокращення промпта за замовчуванням у пайплайні
    use_enhancement = bool(data.get('use_enhancement', True if is_pipeline_mode() else False))
    # NEW: guidance_rescale support
    try:
        guidance_rescale = float(data.get('guidance_rescale', 0.3) or 0.3)
    except Exception:
        guidance_rescale = 0.0
    # Clamp to [0,1] to avoid destabilizing behavior when users input values like 5 or 8
    if guidance_rescale < 0.0:
        guidance_rescale = 0.0
    elif guidance_rescale > 1.0:
        guidance_rescale = 1.0
    # Семплер за замовчуванням: dpmsolver для стабільності
    sampler = str(data.get('sampler', 'dpmsolver') or 'dpmsolver').strip().lower()
    
    print(f"DEBUG: Received data: {data}")
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    # Перевірка мови: керується ENFORCE_ENGLISH або режимом пайплайна
    if should_enforce_english() and not is_english_text(prompt):
        return jsonify({'error': 'Only English prompts are accepted. Please translate your prompt.'}), 400

    speed_settings = {
        'fast': {'steps': 25, 'guidance': 7.5},
        'balanced': {'steps': 50, 'guidance': 8.0},
        'quality': {'steps': 75, 'guidance': 9.0}
    }
    settings = speed_settings.get(generation_speed, speed_settings['fast'])

    generation_start_time = time.time()

    # UPDATED: apply enhancement only if explicitly requested
    enhanced_prompt = text_processor.enhance_prompt(prompt) if use_enhancement else prompt

    # Auto-augment negative prompt to avoid common subject confusions
    auto_neg_terms = text_processor.suggest_negative_terms_for_prompt(prompt) or []
    # Stronger subject isolation when prompt mentions a dog/retriever
    try:
        lp_for_neg = (prompt or "").lower()
    except Exception:
        lp_for_neg = ""
    if re.search(r"\b(dog|retriever|canine|puppy)\b", lp_for_neg):
        auto_neg_terms += [
            # suppress people/vehicles/crowds which often replace/occlude the dog
            "people", "humans", "human face", "crowd", "multiple people",
            "motorcycle", "bike", "bicycle", "car", "vehicle",
            # avoid occlusions and off-frame crops
            "partial body", "cut off", "cropped", "blurry subject",
            # encourage single clear subject
            "multiple animals", "two dogs", "group"
        ]
    auto_neg_str = ", ".join(auto_neg_terms) if auto_neg_terms else ""
    combined_negative = negative_prompt.strip()
    if auto_neg_str:
        combined_negative = (combined_negative + (", " if combined_negative else "") + auto_neg_str)
    processed_negative_prompt = text_processor.process_negative_prompt(combined_negative, negative_preset)

    objects = text_processor.extract_objects_from_prompt(prompt)

    # Dynamic aspect ratio: favor landscape for dog/running prompts to encourage full-body framing
    gen_width, gen_height = 512, 512
    try:
        lower_prompt = (prompt or "").lower()
    except Exception:
        lower_prompt = ""
    if re.search(r"\b(dog|retriever|canine|puppy)\b", lower_prompt) or re.search(r"\b(run|running|sprint|sprinting)\b", lower_prompt):
        gen_width, gen_height = 768, 512  # multiples of 64; wider frame reduces head-only crops
    elif re.search(r"\b(portrait|headshot|close[- ]?up)\b", lower_prompt):
        gen_width, gen_height = 512, 768

    try:
        text_emb = model.encode_text(enhanced_prompt, apply_enhancement=False)
        text_vec = text_emb.mean(dim=1)
    except Exception as e:
        logger.exception(f"encode_text failed: {e}")
        text_vec = None

    # Subject-aware similarity: boost images that contain the intended subject (e.g., dog)
    def compute_similarity(pil_img):
        clip_m, clip_p = get_clip_reranker()
        # Base similarity against the full prompt
        base_inputs = clip_p(text=[enhanced_prompt], images=[pil_img], return_tensors="pt", padding=True)
        base_inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in base_inputs.items()}
        with torch.no_grad():
            base_out = clip_m(**base_inputs)
            base_sim = float(base_out.logits_per_image.squeeze().item())

        # If prompt mentions a dog, apply a presence bonus
        presence_bonus = 0.0
        try:
            lp = lower_prompt
        except NameError:
            lp = (prompt or "").lower()
        if re.search(r"\b(dog|retriever|canine|puppy)\b", lp):
            subject_text = "golden retriever dog, full body, single subject"
            pres_inputs = clip_p(text=[subject_text], images=[pil_img], return_tensors="pt", padding=True)
            pres_inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in pres_inputs.items()}
            with torch.no_grad():
                pres_out = clip_m(**pres_inputs)
                pres_sim = float(pres_out.logits_per_image.squeeze().item())
            # Soft scaling to avoid overpowering the base prompt
            # Map to a moderate bonus using tanh; empirically stable across CLIP variants
            import math
            presence_bonus = 2.0 * math.tanh(pres_sim / 12.0)
            # Enforce a minimum subject presence; penalize frames without a clear dog
            try:
                threshold = float(os.environ.get("DOG_PRESENCE_THRESHOLD", "8.0"))
            except Exception:
                threshold = 8.0
            if pres_sim < threshold:
                # Apply a strong penalty so non-dog images lose to true dog frames
                presence_bonus -= 6.0

        return base_sim + presence_bonus

    # Optionally set sampler on the model
    try:
        if sampler and sampler != 'auto' and hasattr(model, 'set_sampler'):
            model.set_sampler(sampler)
    except Exception as e:
        logger.warning(f"Failed to set sampler '{sampler}': {e}")
    sampler_used = None
    if hasattr(model, 'get_sampler_name'):
        try:
            sampler_used = model.get_sampler_name()
        except Exception:
            sampler_used = sampler if sampler != 'auto' else None

    # Generate candidates and select best by similarity
    best_image = None
    best_similarity = -1.0
    used_seed = None
    
    # Derive seeds
    base_seed = int(seed) if seed is not None else int(time.time())
    candidate_seeds = [base_seed + i for i in range(num_candidates)]
    
    try:
        # Stage 1: draft candidates for reranking
        draft_steps = max(8, settings['steps'] // 3)
        draft_guidance = max(5.5, settings['guidance'] - 0.5)
        scores = []
        for s in candidate_seeds:
            try:
                img = model.generate_image(
                    enhanced_prompt,
                    negative_prompt=processed_negative_prompt,
                    num_inference_steps=draft_steps,
                    guidance_scale=draft_guidance,
                    height=gen_height,
                    width=gen_width,
                    seed=s,
                    guidance_rescale=guidance_rescale,
                    apply_enhancement=use_enhancement
                )
                sim = compute_similarity(img)
                scores.append((sim, s))
            except Exception as e:
                # Changed to logger.exception to include stack trace
                logger.exception(f"Draft generation failed for seed {s}: {e}")
                continue
        # Fallback if no draft scores
        if not scores:
            logger.warning("No draft candidates succeeded; attempting minimal fallback generation")
            try:
                fallback_img = model.generate_image(
                    enhanced_prompt,
                    negative_prompt="",  # disable negative prompt for stability
                    num_inference_steps=max(6, draft_steps // 2),
                    guidance_scale=max(5.0, draft_guidance - 0.5),
                    height=gen_height,
                    width=gen_width,
                    seed=base_seed,
                    guidance_rescale=guidance_rescale,
                    apply_enhancement=use_enhancement
                )
                sim = compute_similarity(fallback_img)
                scores.append((sim, base_seed))
            except Exception as e:
                logger.exception(f"Fallback generation failed: {e}")
        if not scores:
            raise RuntimeError("All draft candidates failed to generate. Try lowering steps or restarting.")
        scores.sort(reverse=True, key=lambda x: x[0])
        top_k = 1 if num_candidates == 1 else min(3, num_candidates)
        top_seeds = [s for _, s in scores[:top_k]]

        # Stage 2: refine best seeds with full settings
        for s in top_seeds:
            try:
                image = model.generate_image(
                    enhanced_prompt,
                    negative_prompt=processed_negative_prompt,
                    num_inference_steps=settings['steps'],
                    guidance_scale=settings['guidance'],
                    height=gen_height,
                    width=gen_width,
                    seed=s,
                    guidance_rescale=guidance_rescale,
                    apply_enhancement=use_enhancement
                )
                sim = compute_similarity(image)
                if sim > best_similarity:
                    best_similarity = sim
                    best_image = image
                    used_seed = s
            except Exception as e:
                # Include stack trace for refine stage
                logger.exception(f"Refine generation failed for seed {s}: {e}")
                continue
        if best_image is None:
            raise RuntimeError("Failed to refine any candidate images. Adjust parameters and retry.")
        
        # Convert to base64 for sending to frontend
        buffered = io.BytesIO()
        best_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        generation_time = time.time() - generation_start_time
        
        # Collect data for auto-retraining
        auto_retraining.collect_prompt_data(
            original_prompt=prompt,
            enhanced_prompt=enhanced_prompt,
            negative_prompt=processed_negative_prompt,
            detected_objects=objects,
            generation_success=True,
            generation_time=generation_time
        )
        
        logger.info(f"Successfully generated image for prompt: {prompt[:50]}... (took {generation_time:.2f}s, sim={best_similarity:.3f}, seed={used_seed})")
        
        # Optional: compute presence score for the selected image (for debugging)
        subject_presence_score = None
        try:
            if re.search(r"\b(dog|retriever|canine|puppy)\b", (prompt or "").lower()):
                clip_m, clip_p = get_clip_reranker()
                pres_inputs = clip_p(text=["golden retriever dog, full body, single subject"], images=[best_image], return_tensors="pt", padding=True)
                pres_inputs = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in pres_inputs.items()}
                with torch.no_grad():
                    pres_out = clip_m(**pres_inputs)
                    subject_presence_score = float(pres_out.logits_per_image.squeeze().item())
        except Exception:
            subject_presence_score = None

        return jsonify({
            'image': img_str,
            'original_prompt': prompt,
            'enhanced_prompt': enhanced_prompt,
            'negative_prompt_used': processed_negative_prompt,
            'detected_objects': objects,
            'generation_time': round(generation_time, 2),
            'similarity': round(best_similarity, 4),
            'used_seed': used_seed,
            'sampler_used': sampler_used,
            'guidance_rescale': guidance_rescale,
            'width': gen_width,
            'height': gen_height,
            'subject_presence_score': subject_presence_score
        })
    except Exception as e:
        generation_time = time.time() - generation_start_time
        
        # Collect failure data for auto-retraining
        auto_retraining.collect_prompt_data(
            original_prompt=prompt,
            enhanced_prompt=enhanced_prompt,
            negative_prompt=processed_negative_prompt,
            detected_objects=objects,
            generation_success=False,
            generation_time=generation_time
        )
        
        logger.error(f"Failed to generate image for prompt: {prompt[:50]}... Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Endpoint for collecting user feedback"""
    data = request.get_json()
    prompt = data.get('prompt', '')
    feedback = data.get('feedback', '')
    rating = data.get('rating')
    
    if not prompt or not feedback:
        return jsonify({'error': 'Prompt and feedback are required'}), 400
    
    try:
        auto_retraining.collect_user_feedback(prompt, feedback, rating)
        logger.info(f"Collected feedback for prompt: {prompt[:30]}...")
        return jsonify({'success': True, 'message': 'Feedback collected successfully'})
    except Exception as e:
        logger.error(f"Error collecting feedback: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Endpoint for getting auto-retraining statistics"""
    try:
        stats = auto_retraining.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/retraining/status', methods=['GET'])
def get_retraining_status():
    """Endpoint for checking if retraining is needed"""
    try:
        needs_retraining = auto_retraining.check_retraining_needed()
        return jsonify({
            'needs_retraining': needs_retraining,
            'stats': auto_retraining.get_statistics()
        })
    except Exception as e:
        logger.error(f"Error checking retraining status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/retraining/start', methods=['POST'])
def start_manual_retraining():
    """Endpoint for manually starting retraining"""
    try:
        success = auto_retraining.start_retraining()
        if success:
            return jsonify({'success': True, 'message': 'Retraining started successfully'})
        else:
            return jsonify({'error': 'Failed to start retraining'}), 500
    except Exception as e:
        logger.error(f"Error starting manual retraining: {e}")
        return jsonify({'error': str(e)}), 500

def _find_latest_checkpoint(search_dir: str = "user_data/models") -> str:
    """Find the most recent checkpoint file (.pt) within retrained model directories.
    Returns an absolute path or None if not found.
    """
    try:
        candidates = []
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for fname in files:
                    if fname.lower().endswith(".pt"):
                        fpath = os.path.join(root, fname)
                        try:
                            mtime = os.path.getmtime(fpath)
                        except Exception:
                            mtime = 0
                        candidates.append((mtime, fpath))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
    except Exception as e:
        try:
            logger.error(f"Failed to search checkpoints: {e}")
        except Exception:
            pass
    return None

def load_model(model_path, use_pipeline: bool = False, hf_cache_dir: str = None):
    global model
    if use_pipeline:
        # Ініціалізація Diffusers‑пайплайна без донавчання
        model = PretrainedPipelineGenerator(
            model_path=model_path,
            device=str(device),
            hf_cache_dir=hf_cache_dir,
        )
        print(f"Using pretrained pipeline: {model_path or 'runwayml/stable-diffusion-v1-5'}")
        return
    else:
        model = ImageGenerationModel(device=device)

    chosen_path = None
    if model_path and os.path.exists(model_path):
        chosen_path = model_path
    else:
        # Try to auto-pick the latest checkpoint from user_data/models
        auto_path = _find_latest_checkpoint()
        if auto_path:
            chosen_path = auto_path

    if chosen_path:
        print(f"Loading model from {chosen_path}")
        try:
            # Спробуємо кілька форматів збереження
            ckpt = None
            try:
                ckpt = torch.load(chosen_path, map_location=device)
            except Exception:
                ckpt = None

            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                # Формат з train_optimized_gpu.py
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                # Наш сумісний формат через ImageGenerationModel.save_model
                model.load_model(chosen_path)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {chosen_path}: {e}")
            print("Falling back to default model weights")
    else:
        print("Using default model weights")

def parse_args():
    parser = argparse.ArgumentParser(description="Web interface for image generation")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--enable_monitoring", action="store_true", help="Enable auto-retraining monitoring")
    parser.add_argument("--use_pipeline", action="store_true", help="Use pretrained diffusers pipeline instead of custom model")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="Optional Hugging Face cache directory (outside project)")
    parser.add_argument("--enforce_english", action="store_true", help="Enforce English-only prompts regardless of mode")
    return parser.parse_args()

@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == "__main__":
    args = parse_args()
    
    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)
    
    # Optional: configure HF cache directory to avoid cleanup deleting weights
    if args.hf_cache_dir:
        try:
            os.makedirs(args.hf_cache_dir, exist_ok=True)
            os.environ["HF_HOME"] = args.hf_cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = args.hf_cache_dir
        except Exception:
            pass

    # Export minimal env for lazy init path when using flask run
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    # Respect default set earlier; only override when flag explicitly provided
    if args.use_pipeline:
        os.environ["USE_PIPELINE"] = "1"
    else:
        os.environ["USE_PIPELINE"] = os.environ.get("USE_PIPELINE", "1")
    if args.hf_cache_dir:
        os.environ["HF_CACHE_DIR"] = args.hf_cache_dir
    # Export English enforcement toggle
    # Keep existing default (1) unless explicitly disabled; if flag passed, set to 1
    os.environ["ENFORCE_ENGLISH"] = "1" if args.enforce_english else os.environ.get("ENFORCE_ENGLISH", "1")

    # Load the model
    load_model(args.model_path, use_pipeline=args.use_pipeline, hf_cache_dir=args.hf_cache_dir)

    # Start auto-retraining monitoring
    if args.enable_monitoring:
        try:
            auto_retraining.start_monitoring()
            logger.info("Auto-retraining monitoring started")
        except Exception as e:
            logger.error(f"Failed to start auto-retraining monitoring: {e}")
    else:
        logger.info("Auto-retraining monitoring is disabled (use --enable_monitoring to turn on)")
    
    # Start the Flask app
    try:
        # Disable the Werkzeug reloader to keep a single stable process for tooling
        app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    finally:
        # Stop monitoring when app shuts down
        auto_retraining.stop_monitoring_process()
        logger.info("Auto-retraining monitoring stopped")