"""
film_maker.assembly
===================
Stitch the rendered shot clips into the film, PRESERVING every clip's H3
audio (dialogue + diegetic sound):

  * "cut" mode (default) — concat-demuxer re-encode of all clips in story
    order. Hard cuts read as editing; each clip carries its own soundscape,
    so the cut IS the sound transition.
  * "crossfade" mode — 0.4s (configurable) audio+video crossfade at SCENE
    boundaries only; shots within a scene stay hard-cut.

Every input has been conformed by render.conform_clip (uniform fps, h264,
48kHz stereo AAC), so concat is safe. Missing clips are skipped with a loud
note so a partially rendered film still assembles for review.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Dict, List, Optional

from .llm import logger


def _run(cmd: List[str]):
    return subprocess.run(cmd, check=True, capture_output=True)


def _concat_reencode(paths: List[str], out_path: str) -> str:
    if len(paths) == 1:
        import shutil
        shutil.copy(paths[0], out_path)
        return out_path
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        listfile = f.name
        for p in paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    try:
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
              "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
              "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
              out_path])
    finally:
        os.unlink(listfile)
    return out_path


def _xfade_pair(a: str, b: str, out_path: str, fade: float) -> str:
    """Audio+video crossfade between two already-uniform clips."""
    da = _duration(a)
    offset = max(0.0, da - fade)
    _run(["ffmpeg", "-y", "-i", a, "-i", b,
          "-filter_complex",
          f"[0:v][1:v]xfade=transition=fade:duration={fade}:offset={offset}[v];"
          f"[0:a][1:a]acrossfade=d={fade}[a]",
          "-map", "[v]", "-map", "[a]",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
          "-c:a", "aac", "-b:a", "192k", out_path])
    return out_path


def _duration(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", path],
            check=True, capture_output=True, text=True)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def assemble_film(scenes: List[Dict], dirs: Dict, cfg: Dict,
                  title_slug: str) -> Optional[str]:
    scene_clip_lists: List[List[str]] = []
    missing = []
    for scene in scenes:
        # v6: one clip per scene -- or several PARTS split at shot
        # boundaries (scene_max_task_seconds); parts join at hard cuts.
        parts = [p for p in (scene.get("clip_parts") or [])
                 if p and os.path.exists(p)]
        if parts and len(parts) == len(scene.get("clip_parts") or []):
            scene_clip_lists.append(parts)
            continue
        sp = scene.get("clip_path") or ""
        if sp and os.path.exists(sp):
            scene_clip_lists.append([sp])
            continue
        # Fallback for pre-v6 projects assembled from per-shot clips.
        clips = []
        for sh in scene.get("shots", []):
            p = sh.get("clip_path") or ""
            if p and os.path.exists(p):
                clips.append(p)
            else:
                missing.append(sh.get("shot_id", "?"))
        if clips:
            scene_clip_lists.append(clips)
        elif sp:
            missing.append(f"scene{scene.get('scene_id','?')}")
    if missing:
        logger.warning("[assembly] %d shot(s) missing and skipped: %s",
                       len(missing), ", ".join(missing))
    if not scene_clip_lists:
        logger.error("[assembly] No clips to assemble.")
        return None

    final_path = os.path.join(dirs["output"], f"{title_slug}_final.mp4")
    if cfg.get("keep_previous_finals", True) and os.path.exists(final_path):
        import datetime
        stamp = datetime.datetime.fromtimestamp(
            os.path.getmtime(final_path)).strftime("%Y%m%d_%H%M%S")
        prev = os.path.join(dirs["output"],
                            f"{title_slug}_final_{stamp}.mp4")
        try:
            os.replace(final_path, prev)
            logger.info("[assembly] Previous final kept as %s",
                        os.path.basename(prev))
        except OSError:
            pass
    mode = cfg.get("scene_transition", "cut")
    if mode != "crossfade" or len(scene_clip_lists) == 1:
        flat = [p for group in scene_clip_lists for p in group]
        _concat_reencode(flat, final_path)
    else:
        fade = float(cfg.get("crossfade_seconds", 0.4))
        tmp_dir = os.path.join(dirs["render"], "assembly_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        # Concat each scene hard-cut, then chain scene blocks with crossfades.
        scene_blocks = []
        for i, group in enumerate(scene_clip_lists):
            block = os.path.join(tmp_dir, f"scene_{i:03d}.mp4")
            _concat_reencode(group, block)
            scene_blocks.append(block)
        acc = scene_blocks[0]
        for i, nxt in enumerate(scene_blocks[1:], 1):
            merged = os.path.join(tmp_dir, f"acc_{i:03d}.mp4")
            try:
                _xfade_pair(acc, nxt, merged, fade)
                acc = merged
            except subprocess.CalledProcessError:
                logger.warning("[assembly] Crossfade failed at scene block "
                               "%d; hard-cutting instead.", i)
                _concat_reencode([acc, nxt], merged)
                acc = merged
        import shutil
        shutil.copy(acc, final_path)

    logger.info("[assembly] Final film (with H3 audio): %s (%.1f min)",
                final_path, _duration(final_path) / 60.0)
    logger.info("[assembly] Score it in post: the non-diegetic music track "
                "was deliberately left empty.")
    return final_path
