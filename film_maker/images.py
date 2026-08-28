"""
film_maker.images
=================
One reference still per shot, generated as N variants with a vision-model
pick (the graphic-novel workflow's proven quality lever).

  * Z-Image Turbo (same pipeline as the music generator), style-bible prefix
    prepended so every still lands in the locked look.
  * image_variants seeds per shot → get_vision scores each against a
    concrete checklist (subject match, action mid-beat, face/mouth
    readability for dialogue shots, composition, artifacts) → best variant
    is copied to image_{shot_id}.jpg. Variants are kept alongside
    (image_{shot_id}_v{n}.jpg) so you can override the pick by hand: delete
    the chosen file, copy your preferred variant over it, and produce()
    reuses it.
  * Idempotent: shots whose chosen still already exists are skipped, so
    regeneration passes only fill gaps.
"""
from __future__ import annotations

import gc
import os
import secrets
import shutil
from typing import Dict, List, Optional

from .llm import get_vision, logger
from .cinematography import style_image_prefix


def _load_zimage(cfg: Dict):
    import torch
    from diffusers import ZImagePipeline
    pipe = ZImagePipeline.from_pretrained(
        cfg.get("image_model_repo", "ykarout/Z-Image-Turbo-FP8-Full"),
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    return pipe


def _free_cuda():
    """Collect garbage and return every freed block from torch's caching
    allocator to the driver. Must run AFTER the last reference to the
    pipeline is gone -- freed-but-cached blocks otherwise stay RESERVED by
    this process and shrink what the wgp.py subprocess can allocate."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _report_vram(prefix: str):
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("%s GPU memory: %.2f GB allocated, %.2f GB reserved.",
                        prefix,
                        torch.cuda.memory_allocated() / 1e9,
                        torch.cuda.memory_reserved() / 1e9)
    except Exception:
        pass


def _generate(pipe, prompt: str, cfg: Dict, seed: int):
    import torch
    if seed == -1:
        seed = secrets.randbits(32)
    image = pipe(
        prompt=prompt + " No text, no titles, no written words, no watermark, "
                        "no signature, no captions.",
        height=int(cfg.get("h3_height", 704)),
        width=int(cfg.get("h3_width", 1280)),
        num_inference_steps=int(cfg.get("image_steps", 9)),
        guidance_scale=0.0,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    return image, seed


def _pick_best(images: List, shot: Dict, paths: List[str]) -> int:
    """Vision-model pick; deterministic fallback to variant 0."""
    has_dialogue = bool(shot.get("lines"))
    checklist = (
        "Score each numbered image 0-10 against this checklist and pick ONE "
        "winner:\n"
        f"- Matches the intended subject and action: {shot.get('action','')[:220]}\n"
        f"- Composition: {shot.get('shot_scale','')} at "
        f"{shot.get('camera_angle','')}, one clear focal subject\n"
        "- Action reads as caught mid-beat (dynamic), not stiffly posed\n"
        + ("- CRITICAL: any speaking character's face is clearly visible, "
           "well-lit, and the mouth is fully unobstructed (this image drives "
           "lip-synced speech)\n" if has_dialogue else "")
        + "- Anatomy, hands, and faces are clean; no duplicated limbs\n"
          "- No text, captions, watermarks, split panels, or collage\n"
          "Respond with ONLY the winning image number (1-"
        + str(len(images)) + ").")
    reply = get_vision(checklist, images)
    for tok in reply.split():
        tok = tok.strip(".,:#()")
        if tok.isdigit():
            n = int(tok)
            if 1 <= n <= len(images):
                return n - 1
    return 0


def generate_stills(scenes: List[Dict], story: Dict, style_bible: Dict,
                    images_dir: str, cfg: Dict,
                    only_shot_ids: Optional[set] = None) -> None:
    os.makedirs(images_dir, exist_ok=True)
    prefix = style_image_prefix(style_bible)
    n_var = max(1, int(cfg.get("image_variants", 4)))

    todo = []
    for scene in scenes:
        for sh in scene.get("shots", []):
            final = os.path.join(images_dir, f"image_{sh['shot_id']}.jpg")
            sh["still_path"] = final
            if only_shot_ids is not None and \
                    sh["shot_id"] not in only_shot_ids:
                continue
            if not os.path.exists(final):
                todo.append(sh)
    if not todo:
        logger.info("[stills] All stills already exist; nothing to generate.")
        return

    logger.info("[stills] Generating %d still(s) x %d variant(s)...",
                len(todo), n_var)
    pipe = _load_zimage(cfg)
    try:
        for i, sh in enumerate(todo, 1):
            prompt = f"{prefix}. {sh.get('image_prompt','')}"
            variants, vpaths = [], []
            for v in range(n_var):
                img, seed = _generate(pipe, prompt, cfg, -1)
                vpath = os.path.join(images_dir,
                                     f"image_{sh['shot_id']}_v{v+1}.jpg")
                img.save(vpath, dpi=(300, 300))
                variants.append(img)
                vpaths.append(vpath)
            best = _pick_best(variants, sh, vpaths) if n_var > 1 else 0
            shutil.copy(vpaths[best], sh["still_path"])
            logger.info("[stills] %d/%d %s -> variant %d chosen.",
                        i, len(todo), sh["shot_id"], best + 1)
    finally:
        # Unload IN THIS SCOPE (the reference owner). CPU-offload first so
        # weights leave the GPU even if a stray internal reference survives,
        # then drop the reference and flush the caching allocator -- the
        # H3 render subprocess needs the whole card.
        try:
            pipe.to("cpu")
        except Exception:
            pass
        del pipe
        _free_cuda()
        _report_vram("[stills] Z-Image released.")
