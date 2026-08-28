"""
film_maker.render
=================
MiniMax H3 rendering through Wan2GP, ported from the proven music-video
adapter (batched wgp.py --process manifest, conda-wrapped subprocess,
stdout-parsed task→file mapping that survives sliding windows, live raw-file
renaming, transient-error retries) with three film-specific changes:

  1. H3'S AUDIO IS KEPT. Every conform/trim/concat preserves the generated
     audio track (the music pipeline stripped it with -an). Clips missing an
     audio stream get a silent AAC bed so concat never fails.

  2. VOICE CHAINING via WAVES. H3's audio reference influences the generated
     voice/timbre, so:
       wave 1 — every silent shot, plus each character's ESTABLISHING shot
                (their first shot in story order where they are the only
                speaker). These render with no audio reference.
       harvest — each establishing clip's audio is extracted into a voice
                bank (voice_bank/<character>.wav), trimmed to the 2-15s
                reference window. Established ONCE and reused verbatim
                everywhere after, so the voice cannot drift through copies.
       wave 2 — every remaining shot renders with its lead speaker's banked
                clip as audio_guide. Multi-speaker shots optionally get a
                concatenated reference of all their speakers' banked voices.
     Disable with cfg["voice_chaining"]=False → single wave, no audio refs.

  3. NO SONG TIMELINE. Shot durations come from the pacing plan: target
     frames = round(duration*fps), rendered at the next valid grid length
     (5+17k), trimmed back to the exact target after render.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Dict, List, Optional

from .llm import logger
from .h3_prompts import lead_speaker, speaking_characters, assemble_h3_prompt


# ---------------------------------------------------------------------------
# Frame grid math (ported)
# ---------------------------------------------------------------------------
def _grid_bounds(cfg: Dict, max_frames: Optional[int] = None):
    off = int(cfg["wgp_frames_offset"])
    step = int(cfg["wgp_frames_steps"])
    k_min = max(1, -(-(int(cfg["wgp_frames_minimum"]) - off) // step))
    max_total = int(max_frames if max_frames is not None else
                    cfg.get("wgp_video_length_max_frames",
                            cfg["wgp_frames_maximum"]))
    k_max = (max_total - off) // step
    return off, step, k_min, k_max


def ceil_to_grid(frames: int, cfg: Dict,
                 max_frames: Optional[int] = None) -> int:
    """Smallest valid H3 grid length (offset + step*k) >= frames, clamped.
    max_frames overrides the single-window cap for long takes."""
    off, step, k_min, k_max = _grid_bounds(cfg, max_frames)
    k = -(-(frames - off) // step)
    k = max(k_min, min(k_max, k))
    return off + step * k


# ---------------------------------------------------------------------------
# ffmpeg helpers (audio-preserving)
# ---------------------------------------------------------------------------
def _run(cmd: List[str]):
    return subprocess.run(cmd, check=True, capture_output=True)


def _has_audio(path: str) -> bool:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type",
             "-of", "default=nokey=1:noprint_wrappers=1", path],
            check=True, capture_output=True, text=True)
        return "audio" in out.stdout
    except Exception:
        return False


def _probe_duration(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", path],
            check=True, capture_output=True, text=True)
        return float(out.stdout.strip())
    except Exception:
        return -1.0


def conform_clip(path: str, target_seconds: float, fps: float) -> str:
    """Force `path` to exactly target_seconds at fps, KEEPING its audio.
    Short clips are padded by holding the last frame (audio padded with
    silence); long clips are trimmed. Clips with no audio stream gain a
    silent AAC track so downstream concat is uniform."""
    if target_seconds <= 0 or not os.path.exists(path):
        return path
    has_aud = _has_audio(path)
    actual = _probe_duration(path)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False,
                                     dir=os.path.dirname(path)) as tmp:
        tmp_path = tmp.name
    try:
        vf = f"fps={fps}"
        if 0 < actual < target_seconds - (1.0 / fps):
            deficit = target_seconds - actual
            vf += f",tpad=stop_mode=clone:stop_duration={deficit:.6f}"
            logger.info("  [conform] %s rendered %.2fs short; holding last "
                        "frame.", os.path.basename(path), deficit)
        if has_aud:
            cmd = ["ffmpeg", "-y", "-i", path,
                   "-vf", vf, "-af", "apad",
                   "-t", f"{target_seconds:.6f}",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                   "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                   tmp_path]
        else:
            cmd = ["ffmpeg", "-y", "-i", path,
                   "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                   "-vf", vf,
                   "-t", f"{target_seconds:.6f}",
                   "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                   "-c:a", "aac", "-b:a", "192k", "-shortest",
                   tmp_path]
        _run(cmd)
        shutil.move(tmp_path, path)
    except subprocess.CalledProcessError as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        tail = e.stderr.decode("utf-8", "ignore")[-300:] if e.stderr else ""
        logger.warning("  [conform] Could not conform %s (%s); leaving as-is.",
                       os.path.basename(path), tail)
    return path


def extract_speaker_frame(clip_path: str, out_png: str) -> Optional[str]:
    """Grab one mid-clip frame as the VISUAL half of a voice reference:
    MiniMax H3 requires at least as many reference images/videos as audio
    references (each voice binds to a face), so every banked voice wav
    gets a companion frame from the SAME establishing clip -- face and
    voice from the same rendered footage. In stills-free mode this frame
    doubles as a rendered-footage identity anchor for later scenes."""
    if not os.path.exists(clip_path):
        return None
    dur = _probe_duration(clip_path)
    mid = max(0.0, (dur or 2.0) / 2.0)
    try:
        _run(["ffmpeg", "-y", "-ss", f"{mid:.2f}", "-i", clip_path,
              "-frames:v", "1", "-q:v", "2", out_png])
        return out_png if os.path.exists(out_png) else None
    except Exception as e:
        logger.warning("[render] Speaker frame grab failed for %s: %s",
                       os.path.basename(clip_path), e)
        return None


def extract_voice_ref(clip_path: str, out_wav: str, min_s: float,
                      max_s: float) -> Optional[str]:
    """Pull the clip's audio as a mono voice reference in [min_s, max_s]."""
    if not os.path.exists(clip_path) or not _has_audio(clip_path):
        return None
    dur = _probe_duration(clip_path)
    take = max(min_s, min(max_s, dur if dur > 0 else max_s))
    try:
        _run(["ffmpeg", "-y", "-i", clip_path, "-vn",
              "-t", f"{take:.3f}", "-ac", "1", "-ar", "44100",
              "-c:a", "pcm_s16le", out_wav])
        return out_wav
    except subprocess.CalledProcessError:
        return None


def _concat_wavs(paths: List[str], out_wav: str, max_s: float) -> Optional[str]:
    """Concatenate several voice refs into one, capped at max_s total (used
    for multi-speaker shots so the reference carries every voice)."""
    if not paths:
        return None
    if len(paths) == 1:
        shutil.copy(paths[0], out_wav)
        return out_wav
    per = max(1.5, max_s / len(paths))
    trimmed = []
    try:
        for i, p in enumerate(paths):
            t = out_wav + f".part{i}.wav"
            _run(["ffmpeg", "-y", "-i", p, "-t", f"{per:.3f}",
                  "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", t])
            trimmed.append(t)
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                         delete=False) as f:
            listfile = f.name
            for t in trimmed:
                f.write(f"file '{os.path.abspath(t)}'\n")
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
              "-c:a", "pcm_s16le", out_wav])
        os.unlink(listfile)
        return out_wav
    except subprocess.CalledProcessError:
        return None
    finally:
        for t in trimmed:
            if os.path.exists(t):
                os.unlink(t)


# ---------------------------------------------------------------------------
# Job planning + voice-chaining waves
# ---------------------------------------------------------------------------
# End-of-scene buttons: a deliberate held beat after the final line, so
# scenes end on an emotion instead of a mid-breath cut.
_SCENE_BUTTONS = {
    "fear": "a held breath and a slow look back toward what should not "
            "be there, the soundscape thinning to a single detail",
    "anger": "a long unblinking stare as a hand sets something down with "
             "exaggerated care, one hard exhale through the nose",
    "grief": "eyes resting on the absent one's belongings a beat too "
             "long, the ambience sinking almost to silence",
    "joy": "a smile finally allowed to win, held to the camera a beat "
           "longer than polite",
    "shame": "a glance away from the lens, hands stilling mid-fidget",
    "love": "eye contact held half a beat past comfortable before one of "
            "them looks down, hiding the smile",
    "wonder": "a slow, silent stare upward, lips parted, the key sound "
              "swelling gently",
    "resolve": "a single decisive nod to no one, jaw set, before the cut",
}


def _speech_seconds(shot: Dict, cfg: Dict) -> float:
    """Estimated PERFORMED time for a shot's dialogue: word count at a
    spoken-performance rate plus a turn gap per line (breath, reaction,
    delivery-cue pauses). H3 honors [/duration], so a window shorter than
    this cuts the character off mid-sentence."""
    lines = shot.get("lines") or []
    if not lines:
        return 0.0
    wps = float(cfg.get("speech_words_per_second", 2.3))
    gap = float(cfg.get("dialogue_turn_gap_seconds", 0.6))
    words = sum(len(str(ln.get("text", "")).split()) for ln in lines)
    return words / max(0.5, wps) + gap * len(lines)


def _ensure_dialogue_fit(shot: Dict, cfg: Dict, scene_last: bool) -> None:
    """Floor a shot's duration to (speech time + settle room), plus the
    scene-button hold on the scene's final shot -- dialogue always
    finishes, with air after it."""
    need = _speech_seconds(shot, cfg)
    if need <= 0 and not scene_last:
        return
    need += float(cfg.get("dialogue_settle_seconds", 1.2))
    if scene_last and cfg.get("scene_button", True):
        need += float(cfg.get("scene_button_seconds", 2.5))
    cur = float(shot.get("duration", 0))
    if need > cur + 0.05:
        shot["duration"] = round(need, 1)
        logger.info("[render] Shot %s: duration %.1fs -> %.1fs so the "
                    "dialogue finishes%s.", shot.get("shot_id"), cur,
                    shot["duration"],
                    " (+ scene button)" if scene_last else "")


def _scene_button_text(scene: Dict) -> str:
    # Authored exit hook (hook architecture) beats the stock button.
    if scene.get("exit_hook"):
        return (f" After the final line, the take holds for a deliberate "
                f"closing beat -- {scene['exit_hook']} -- before it "
                f"ends; no one speaks during this hold.")
    from .behavior_grammar import _resolve_emotions
    keys = _resolve_emotions([scene.get("emotion", ""),
                             scene.get("summary", "")[:120]])
    hook = _SCENE_BUTTONS.get(keys[0] if keys else "resolve",
                              _SCENE_BUTTONS["resolve"])
    return (f" After the final line, the take holds for a deliberate "
            f"closing beat -- {hook} -- before it ends; no one speaks "
            f"during this hold.")


def _fmt_secs(seconds: float) -> str:
    s = round(float(seconds), 1)
    return f"{int(s)}s" if abs(s - int(s)) < 0.05 else f"{s}s"


def _identity_lock_block(scene: Dict, story: Dict, cfg: Dict,
                         wardrobe: Optional[Dict] = None,
                         locations: Optional[Dict] = None) -> str:
    """Text-only identity anchoring for stills-free rendering: the same
    consistency job the Z-Image still does visually, done verbally -- the
    strongest identity anchors (facial identity, hair, build, wardrobe,
    distinctive props) plus the location's look, stated once at the top of
    the scene generation and held ('these details must remain preserved')
    across every window by the model's own cross-window memory."""
    from .cinematography import wardrobe_for
    chars = []
    names_here = set(scene.get("characters") or [])
    scene_id = int(scene.get("scene_id", 1))
    for c in (story.get("characters") or []):
        if names_here and c.get("name") not in names_here:
            continue
        # visual_lock is the canonical 60-100 word character description
        # written to be reproduced VERBATIM in every Z-Image prompt --
        # in stills-free mode it does the same job verbally.
        bits = [c.get("visual_lock") or c.get("screen_tag", "")]
        if wardrobe:
            state = wardrobe_for(c.get("name", ""), scene_id, wardrobe)
            if state:
                bits.append(f"current wardrobe state: {state}")
        desc = "; ".join(str(b).strip() for b in bits
                         if b and str(b).strip())
        if desc:
            chars.append(f"{c['name']}: {desc}")
    loc_line = ""
    if locations:
        loc = locations.get(scene.get("location", ""), {})
        if isinstance(loc, dict):
            loc_line = loc.get("look") or loc.get("description") or ""
    parts = []
    if chars:
        parts.append("Character identity anchors, exact in every shot of "
                     "this scene -- faces, hairstyles, builds, and clothing "
                     "must remain preserved and never drift: "
                     + " | ".join(chars) + ".")
    if loc_line:
        parts.append(f"Setting, constant for the whole scene: {loc_line}.")
    return " ".join(parts)


def _scene_window_lines(scene: Dict, story: Dict, style_medium: str,
                        cfg: Dict, identity_lock: str = "") -> List[str]:
    """One prompt LINE per sliding window for a whole-scene generation,
    directed with the fork's inline window commands (docs/PROMPTS.md):
      * every shot boundary is a NATIVE hard cut -- '[/duration=Xs,
        /new_shot]' -- so cuts happen inside the model with full identity
        memory, not by concatenating separately rendered clips;
      * a shot longer than one window becomes several windows: the first
        carries the cut, continuations carry '[/duration=..,/overlap=N]'
        so multi-frame overlap hands real motion + audio context across
        the join, with each window's text a distinct authored PHASE
        (never re-describing what already happened);
      * the scene's first window has only a duration command -- it starts
        from the Start Image (or from text alone in stills-free mode,
        where the identity lock text leads the line)."""
    fps = float(cfg.get("h3_fps", 24.0))
    single_cap = int(cfg.get("wgp_windowed_window_frames") or
                     cfg.get("wgp_video_length_max_frames",
                             cfg.get("wgp_frames_maximum", 362)))
    overlap = int(cfg.get("wgp_windowed_overlap_frames", 17))
    lines: List[str] = []
    for sh in scene.get("shots", []):
        dur = float(sh.get("duration", cfg.get("min_shot_seconds", 5)))
        frames = max(1, int(round(dur * fps)))
        n_win = 1 if frames <= single_cap else \
            estimate_window_count(frames, cfg)
        phases = sh.get("chain_phases") or []
        shot_lines = sh.get("lines") or []
        placed = set()
        win_secs = dur / n_win
        for w in range(n_win):
            if n_win > 1 and phases:
                pi = min(len(phases) - 1, (w * len(phases)) // n_win)
                ph = phases[pi]
                idxs = [int(j) for j in (ph.get("line_indices") or [])
                        if isinstance(j, (int, float))
                        and 0 <= int(j) < len(shot_lines)
                        and int(j) not in placed]
                placed.update(idxs)
                ov = {"action": ph.get("action", sh.get("action", "")),
                      "motion": ph.get("motion", sh.get("motion", "")),
                      "lines": [shot_lines[j] for j in idxs]}
                if w > 0:
                    ov["action"] = (str(ov["action"]) +
                                    " (the same unbroken take continuing: "
                                    "advance the action forward, never "
                                    "restart or repeat)")
            elif n_win > 1:
                ov = {"lines": shot_lines if w == 0 else []}
                if w > 0:
                    ov["action"] = (str(sh.get("action", "")) +
                                    f" (part {w + 1} of {n_win} of the "
                                    f"same unbroken take: continue "
                                    f"forward, never restart)")
            else:
                ov = None
            p = assemble_h3_prompt(sh, scene, story, style_medium, cfg,
                                   chained=False, overrides=ov)
            flat = " ".join(p.split())
            if not lines:                       # scene's very first window
                cmd = f"[/duration={_fmt_secs(win_secs)}]"
                if scene.get("entry_hook"):
                    flat += (" The take opens already in motion, no "
                             "settling-in: " +
                             " ".join(str(scene["entry_hook"]).split())
                             + ".")
                if identity_lock:
                    flat = " ".join(identity_lock.split()) + " " + flat
            elif w == 0:                        # a cut to a new shot
                cmd = f"[/duration={_fmt_secs(win_secs)},/new_shot]"
            else:                               # long-take continuation
                cmd = (f"[/duration={_fmt_secs(win_secs)},"
                       f"/overlap={overlap}]")
            lines.append(f"{cmd} {flat}")
        if n_win > 1 and phases:
            missing = [j for j in range(len(shot_lines))
                       if j not in placed]
            if missing:
                logger.warning("[render] Scene %s shot %s: %d line(s) "
                               "unplaced by phases.",
                               scene.get("scene_id"), sh.get("shot_id"),
                               len(missing))
    return lines


def build_scene_jobs(scenes: List[Dict], story: Dict, style_bible: Dict,
                     dirs: Dict, cfg: Dict,
                     only_scene_ids: Optional[set] = None,
                     wardrobe: Optional[Dict] = None,
                     locations: Optional[Dict] = None) -> List[Dict]:
    """ONE render job per SCENE: the whole scene is a single continuous
    H3 generation through Wan2GP's sliding-window engine, directed window
    by window (one prompt line per window, native [/new_shot] cuts at
    shot boundaries, multi-frame-overlap continuations inside long
    takes). This REPLACES last-frame chaining entirely: no frame is ever
    extracted from one clip to start another -- identity, motion, and
    audio continuity are carried by the model's own reference memories
    and window overlap, which is exactly what they are for.

    The scene starts from its synthesized Z-Image still (Start Image +
    persistent Ref2VA reference) -- or, with use_reference_stills=False,
    from text alone with a detailed identity-lock block leading window
    1."""
    import secrets
    clips_dir = dirs["clips"]
    os.makedirs(clips_dir, exist_ok=True)
    fps = float(cfg.get("h3_fps", 24.0))
    style_medium = (style_bible or {}).get("medium", "cinematic")
    use_stills = bool(cfg.get("use_reference_stills", True))
    jobs = []
    for scene in scenes:
        shots = scene.get("shots", [])
        if not shots:
            continue
        sid = int(scene["scene_id"])
        out_path = os.path.join(clips_dir, f"scene_{sid:03d}.mp4")
        scene["clip_path"] = os.path.abspath(out_path)
        for sh in shots:
            sh.pop("clip_path", None)   # scene clip supersedes shot clips
        if only_scene_ids and sid not in only_scene_ids:
            continue
        still = shots[0].get("still_path") or os.path.join(
            dirs["images"], f"image_{shots[0]['shot_id']}.jpg")
        if use_stills and not os.path.exists(still):
            logger.warning("Scene %s: missing opening still %s -- "
                           "skipping.", sid, still)
            continue
        # RAM guard: WanGP holds a task's ENTIRE output (frames, latents,
        # H3 audio) in memory until it finishes, so a 100s scene as one
        # task can exhaust system RAM (WSL2 grinding to a halt at 96%+).
        # Long scenes split into PARTS at SHOT boundaries -- a hard cut
        # carries no motion/audio continuity anyway, so parts joined at
        # cuts lose nothing and mid-take stitching stays gone. A single
        # shot longer than the cap stays whole (splitting it WOULD
        # reintroduce mid-take stitching) with a warning.
        for si, sh in enumerate(shots):
            _ensure_dialogue_fit(sh, cfg, scene_last=(si == len(shots) - 1))
        cap_secs = float(cfg.get("scene_max_task_seconds", 45.0))
        parts: List[List[Dict]] = [[]]
        acc = 0.0
        for sh in shots:
            d = float(sh.get("duration", 5))
            if parts[-1] and acc + d > cap_secs:
                parts[-1+ 0].append if False else None
                parts.append([])
                acc = 0.0
            parts[-1].append(sh)
            acc += d
            if d > cap_secs:
                logger.warning("Scene %s shot %s: %.0fs single take "
                               "exceeds scene_max_task_seconds=%.0fs; "
                               "kept whole (watch RAM).",
                               sid, sh.get("shot_id"), d, cap_secs)
        multi = len(parts) > 1
        scene["clip_parts"] = []
        identity_lock = "" if use_stills else _identity_lock_block(
            scene, story, cfg, wardrobe=wardrobe, locations=locations)
        for pi, part_shots in enumerate(parts, 1):
            part_path = (os.path.join(clips_dir,
                                      f"scene_{sid:03d}_p{pi}.mp4")
                         if multi else out_path)
            scene["clip_parts"].append(os.path.abspath(part_path))
            total_secs = sum(float(sh.get("duration", 5))
                             for sh in part_shots)
            total_frames = max(1, int(round(total_secs * fps)))
            render_frames = ceil_to_grid(
                total_frames, cfg,
                max_frames=total_frames
                + int(cfg.get("wgp_frames_steps", 17)))
            sub_scene = dict(scene)
            sub_scene["shots"] = part_shots
            lines = _scene_window_lines(sub_scene, story, style_medium,
                                        cfg, identity_lock=identity_lock)
            if pi == len(parts) and cfg.get("scene_button", True) and lines:
                lines[-1] = (lines[-1].rstrip()
                             + _scene_button_text(scene))
            prompt = str(cfg.get("wgp_windowed_prompt_separator",
                                 "\n")).join(lines)
            # WanGP parses braces in prompts as template variables (its
            # macro system) and SKIPS any task containing an unknown one
            # -- curly braces must never reach the manifest. Square
            # brackets are safe: only [/...] is a command.
            prompt = prompt.replace("{", "(").replace("}", ")")
            speakers = []
            for sh in part_shots:
                for s in speaking_characters(sh):
                    if s not in speakers:
                        speakers.append(s)
            jobs.append({
                "shot_id": (f"scene{sid:03d}_p{pi}" if multi
                            else f"scene{sid:03d}"),
                "scene_id": sid,
                "prompt": prompt,
                "image_ref": (os.path.abspath(still) if use_stills
                              else None),
                # The scene still hard-pins frame 1 only for part 1;
                # later parts open on a CUT, so the still rides along as
                # an identity reference only.
                "_start_image_ok": (pi == 1),
                "audio_ref": None,          # filled per wave
                "num_frames": render_frames,
                "n_windows": len(lines),
                "seed": scene.get("seed") or secrets.randbelow(2**31),
                "duration": total_secs,
                "target_seconds": total_secs,
                "output_path": part_path,
                "lead_speaker": speakers[0] if speakers else "",
                "speakers": speakers,
                "_extra_task_fields": _windowed_task_fields(cfg),
                "_rendered": os.path.exists(part_path),
            })
        if multi:
            logger.info("[render] Scene %s: %.0fs split into %d task "
                        "part(s) at shot boundaries (RAM cap %.0fs).",
                        sid, sum(float(s.get("duration", 5))
                                 for s in shots), len(parts), cap_secs)
    done = sum(1 for j in jobs if j["_rendered"])
    logger.info("[render] %d scene job(s): %d already rendered, %d to go "
                "(%d windows total).", len(jobs), done, len(jobs) - done,
                sum(j["n_windows"] for j in jobs))
    return jobs


def estimate_window_count(total_frames: int, cfg: Dict) -> int:
    """How many sliding windows Wan2GP will generate for a WINDOWED long
    take: the first window covers the window size; each later window
    advances (size - overlap) new frames. Kept here (frame domain) so
    planning-time phase counts can match render-time window counts."""
    size = int(cfg.get("wgp_windowed_window_frames") or
               cfg.get("wgp_video_length_max_frames",
                       cfg["wgp_frames_maximum"]))
    if total_frames <= size:
        return 1
    overlap = int(cfg.get("wgp_windowed_overlap_frames", 17))
    stride = max(1, size - overlap)
    return 1 + -(-(total_frames - size) // stride)


def _windowed_task_fields(cfg: Dict) -> Dict:
    """Extra wgp task fields that turn on per-window prompting for a
    windowed long take: the sliding-window size/overlap (multi-frame
    overlap is what carries MOTION and AUDIO context across the join --
    the thing chain mode's single image_start frame cannot) plus the
    fork's switch that consumes one prompt line per window. All field
    names/values are config knobs because they are fork-dependent --
    verify ONCE with wgp_dry_run=True against a UI \"Export Settings\"
    JSON, exactly as for the chain-mode fields."""
    size = int(cfg.get("wgp_windowed_window_frames") or
               cfg.get("wgp_video_length_max_frames",
                       cfg["wgp_frames_maximum"]))
    overlap = int(cfg.get("wgp_windowed_overlap_frames", 17))
    fields = {
        str(cfg.get("wgp_windowed_size_field",
                    "sliding_window_size")): size,
        str(cfg.get("wgp_windowed_overlap_field",
                    "sliding_window_overlap")): overlap,
    }
    fields.update(dict(cfg.get("wgp_windowed_extra_task_fields") or {}))
    return fields


def _mark_establishing(jobs: List[Dict]) -> None:
    """Stamp each character's ESTABLISHING scene (the first scene where
    they are the only speaker) with _establishing_for -- the clip whose
    rendered audio becomes that character's banked voice reference.
    Characters who never speak alone in a scene simply render unbanked;
    within any one scene the single continuous generation keeps their
    voice consistent by construction."""
    seen = set()
    for j in jobs:
        speakers = [s for s in j["speakers"] if s]
        new_here = [s for s in speakers if s not in seen]
        seen.update(speakers)
        if new_here:
            # This scene's clip is the first place these voices exist --
            # bank it for each of them. (The narrator speaks in nearly
            # every scene, so requiring solo scenes would bank nothing;
            # a mixed clip is a valid H3 reference: it locks every (S#)
            # voice heard in it.)
            j["_establishing_for"] = new_here[0]
            j["_establishing_for_list"] = new_here


def _plan_waves(jobs: List[Dict], cfg: Dict) -> List[List[Dict]]:
    """Wave 1: silent scenes + each character's establishing scene (their
    first solo-voiced scene, whose rendered audio becomes their banked
    voice reference). Wave 2: everything else, with banked voice refs
    attached before submission."""
    if not cfg.get("voice_chaining", True):
        return [jobs]
    wave1, wave2 = [], []
    for j in jobs:
        speakers = [s for s in j["speakers"] if s]
        if not speakers or j.get("_establishing_for"):
            wave1.append(j)
        else:
            wave2.append(j)
    n_est = sum(1 for j in wave1 if j.get("_establishing_for"))
    logger.info("[render] Voice chaining: wave 1 = %d shot(s) (%d "
                "establishing), wave 2 = %d shot(s).",
                len(wave1), n_est, len(wave2))
    return [w for w in (wave1, wave2) if w]


def _harvest_voices(jobs: List[Dict], voice_dir: str, cfg: Dict,
                    bank: Dict[str, str]) -> None:
    os.makedirs(voice_dir, exist_ok=True)
    lo = float(cfg.get("wgp_audio_ref_min_seconds", 2.0))
    hi = float(cfg.get("wgp_audio_ref_max_seconds", 15.0))
    for j in jobs:
        sps = [s for s in (j.get("_establishing_for_list") or
                           ([j["_establishing_for"]]
                            if j.get("_establishing_for") else []))
               if s and s not in bank]
        if not sps or not j.get("_rendered"):
            continue
        first_out = None
        for sp in sps:
            out = os.path.join(
                voice_dir, f"{re.sub(r'[^A-Za-z0-9_-]', '_', sp)}.wav")
            if first_out is None:
                got = extract_voice_ref(j["output_path"], out, lo, hi)
                if not got:
                    break
                first_out = got
                extract_speaker_frame(j["output_path"],
                                      os.path.splitext(out)[0] + ".png")
            else:
                shutil.copyfile(first_out, out)
                src_png = os.path.splitext(first_out)[0] + ".png"
                if os.path.exists(src_png):
                    shutil.copyfile(src_png,
                                    os.path.splitext(out)[0] + ".png")
                got = out
            bank[sp] = got
            logger.info("[render] Voice established for %s -> %s",
                        sp, os.path.basename(got))


def _attach_voice_refs(jobs: List[Dict], bank: Dict[str, str],
                       voice_dir: str, cfg: Dict) -> None:
    hi = float(cfg.get("wgp_audio_ref_max_seconds", 15.0))
    for j in jobs:
        speakers = [s for s in j["speakers"] if s and s in bank]
        if not speakers:
            continue
        if len(speakers) == 1 or \
                cfg.get("voice_ref_mode", "lead") == "lead":
            # ONE clean voice per reference: a concatenated multi-voice
            # wav switches speakers mid-stream, which reliably garbles
            # H3's synthesized speech (gibberish words). The lead
            # speaker's banked wav locks the most important voice; the
            # rest stay coherent via in-scene context.
            lead = j.get("lead_speaker") or speakers[0]
            j["audio_ref"] = bank.get(lead) or bank[speakers[0]]
        else:                                   # legacy "combo" mode
            combo = os.path.join(
                voice_dir, f"combo_{j['shot_id']}.wav")
            j["audio_ref"] = _concat_wavs([bank[s] for s in speakers],
                                          combo, hi) or bank[speakers[0]]
        # H3 validation: visual refs must be >= audio refs. With a scene
        # still that's already satisfied; in stills-free mode, pair the
        # voice with each banked speaker's FRAME from their establishing
        # clip (face + voice from the same rendered footage).
        if j["audio_ref"] and not j.get("image_ref"):
            frames = []
            for s in speakers:
                png = os.path.splitext(bank[s])[0] + ".png"
                if os.path.exists(png) and png not in frames:
                    frames.append(png)
            if frames:
                j["image_refs_extra"] = frames
            else:
                logger.warning("[render] %s: no speaker frame available "
                               "to pair with the voice ref -- dropping "
                               "audio ref to satisfy H3's visual>=audio "
                               "rule.", j["shot_id"])
                j["audio_ref"] = None


# ---------------------------------------------------------------------------
# (last-frame chaining removed in v6: scenes render as ONE generation)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Live raw-file naming -- wgp.py names outputs "{datetime}_seed{seed}_{full
# prompt text}.mp4", which is unreadable and absurdly long. The moment a
# save is detected on stdout, the file is renamed to a stable, monitorable
# scheme tied to the actual shot:
#     scene_04_shot_02_seed1636186535.mp4
#     scene_04_shot_02_window_1of2_seed...mp4   (multi-window long takes)
#     scene_04_shot_02_segment_01_seed...mp4    (chain-mode segments)
# Purely cosmetic for wgp_render/raw while a batch runs; each task's final
# file is then moved to shot_clips/shot_<id>.mp4 as before.
# ---------------------------------------------------------------------------
def _live_name(job: Dict, win_no: int, win_total: int) -> str:
    sid = str(job.get("shot_id", "unknown"))
    seg = ""
    if ".seg" in sid:
        base, s = sid.split(".seg", 1)
        try:
            seg = f"_segment_{int(s) + 1:02d}"
        except ValueError:
            seg = f"_segment_{s}"
        sid = base
    parts = sid.split("_")
    scene_part = parts[0] if parts else "00"
    shot_part = parts[1] if len(parts) > 1 else "01"
    win = f"_window_{win_no}of{win_total}" if win_total and win_total > 1 else ""
    seed = job.get("seed")
    seed_part = f"_seed{seed}" if seed is not None else ""
    return f"scene_{scene_part}_shot_{shot_part}{seg}{win}{seed_part}.mp4"


def _rename_live(path: str, job: Dict, win_no: int, win_total: int) -> str:
    new_path = os.path.join(os.path.dirname(path),
                            _live_name(job, win_no, win_total))
    if os.path.abspath(new_path) == os.path.abspath(path):
        return path
    try:
        os.replace(path, new_path)
        return new_path
    except OSError as e:
        logger.warning("[wgp] Couldn't live-rename %s (%s); keeping the "
                       "original name.", os.path.basename(path), e)
        return path


# ---------------------------------------------------------------------------
# Wan2GP adapter (ported: manifest, conda wrap, stdout parsing, retries)
# ---------------------------------------------------------------------------
def _mmgp_profile_flags(cfg: Dict) -> List[str]:
    """Build --profile / --perc-reserved-mem-max from the named config
    knobs. mmgp logs, on every run, the exact MB needed for full pinning vs
    what's reservable at the current perc -- use that to compute the perc
    that actually fits (see wgp_perc_reserved_mem_max's comment in
    config.py) rather than guessing upward. Warns once per call if perc is
    pushed high enough to risk starving the OS/other processes."""
    flags = []
    profile = cfg.get("wgp_profile")
    if profile:
        flags += ["--profile", str(profile)]
    perc = cfg.get("wgp_perc_reserved_mem_max")
    if perc is not None:
        try:
            if float(perc) >= 0.85:
                logger.warning(
                    "[wgp] wgp_perc_reserved_mem_max=%s reserves a very "
                    "large share of system RAM for model pinning -- if "
                    "this doesn't leave ~15-20%% headroom for the OS/other "
                    "processes, prefer the pruned checkpoint or a lower "
                    "wgp_profile instead of pushing this further.",
                    perc)
        except (TypeError, ValueError):
            pass
        flags += ["--perc-reserved-mem-max", str(perc)]
    return flags


def _wgp_task(job: Dict, cfg: Dict) -> Dict:
    # wgp.py runs with cwd=wgp_repo_dir: every path in the task MUST be
    # absolute or it resolves inside the Wan2GP repo and fails.
    task = {
        "model_type": cfg["wgp_model_type"],
        "prompt": job["prompt"],
        "resolution": f'{cfg["h3_width"]}x{cfg["h3_height"]}',
        "video_length": job["num_frames"],
        "seed": job["seed"],
        "num_inference_steps": cfg["wgp_num_inference_steps"],
        "image_prompt_type": "",
        "video_prompt_type": cfg.get("wgp_video_prompt_type", "KI"),
    }
    refs = []
    if job.get("image_ref"):
        refs.append(os.path.abspath(job["image_ref"]))
    for extra in (job.get("image_refs_extra") or []):
        p = os.path.abspath(extra)
        if os.path.exists(p) and p not in refs:
            refs.append(p)
    if refs:
        # Visual references: the scene still (identity memory the fork
        # carries across every sliding window) and/or banked speaker
        # frames pairing each reference voice with a face. Field names
        # are fork-dependent -- verify once with wgp_dry_run=True.
        task["image_refs"] = refs
        if (job.get("image_ref") and cfg.get("scene_start_image", True)
                and job.get("_start_image_ok", True)):
            # Only the scene still may pin frame 1 -- never a speaker
            # frame.
            task[str(cfg.get("wgp_scene_start_image_field",
                             "image_start"))] = refs[0]
            task["video_prompt_type"] = (
                cfg.get("wgp_video_prompt_type", "KI")
                + str(cfg.get("wgp_scene_start_image_prompt_type", "S")))
    else:
        # Pure text conditioning (stills-free, no audio refs attached).
        task["video_prompt_type"] = cfg.get(
            "wgp_video_prompt_type_textonly", "")
    if job.get("audio_ref"):
        task["audio_prompt_type"] = "A"
        task["audio_guide"] = os.path.abspath(job["audio_ref"])
    extra = dict(job.get("_extra_task_fields") or {})
    for k, v in list(extra.items()):
        if isinstance(v, list):
            extra[k] = [os.path.abspath(x) if isinstance(x, str)
                        and x.endswith((".png", ".jpg", ".jpeg", ".wav",
                                        ".mp4")) else x for x in v]
        elif isinstance(v, str) and v.endswith((".png", ".jpg", ".jpeg",
                                                ".wav", ".mp4")):
            extra[k] = os.path.abspath(v)
    task.update(extra)
    if cfg.get("wgp_use_turbo_lora"):
        task["activated_loras"] = [cfg["wgp_turbo_lora_filename"]]
        task["loras_multipliers"] = cfg.get("wgp_lora_multiplier", "1.0")
    return task


def ensure_turbo_lora(cfg: Dict) -> bool:
    if not cfg.get("wgp_use_turbo_lora"):
        return False
    filename = cfg["wgp_turbo_lora_filename"]
    target_dir = os.path.join(cfg["wgp_repo_dir"], cfg["wgp_lora_subdir"])
    target = os.path.join(target_dir, filename)
    if os.path.exists(target):
        return True
    search = [target_dir] + [os.path.join(cfg["wgp_repo_dir"], d)
                             for d in (cfg.get("wgp_lora_search_dirs") or [])]
    found = next((os.path.join(d, filename) for d in search
                  if os.path.exists(os.path.join(d, filename))), None)
    os.makedirs(target_dir, exist_ok=True)
    if found:
        try:
            shutil.copy(found, target)
            return True
        except Exception as e:
            logger.warning("[wgp] Couldn't copy Turbo LoRA (%s); disabling.", e)
            cfg["wgp_use_turbo_lora"] = False
            return False
    try:
        from huggingface_hub import hf_hub_download
        logger.info("[wgp] Downloading Turbo LoRA...")
        dl = hf_hub_download(repo_id=cfg["wgp_turbo_lora_repo"],
                             filename=filename,
                             token=cfg.get("hf_token") or None)
        shutil.copy(dl, target)
        return True
    except Exception as e:
        logger.warning("[wgp] Turbo LoRA download failed (%s); continuing "
                       "without it -- consider wgp_num_inference_steps=20.", e)
        cfg["wgp_use_turbo_lora"] = False
        return False


def _conda_wrapped(cmd: List[str], cfg: Dict) -> str:
    quoted = " ".join(shlex.quote(p) for p in cmd)
    env, base = cfg["wgp_conda_env"], cfg["wgp_conda_base_env"]
    return (
        f'CONDA_BASE="$(conda info --base 2>/dev/null)"; '
        f'if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then '
        f'source "$CONDA_BASE/etc/profile.d/conda.sh"; fi; '
        f'conda activate {shlex.quote(env)}; {quoted}; status=$?; '
        f'conda activate {shlex.quote(base)} 2>/dev/null || '
        f'conda deactivate 2>/dev/null || true; exit $status')


def _render_batch(jobs: List[Dict], render_dir: str, cfg: Dict,
                  label: str = "") -> None:
    """One wgp.py --process call for a wave. Mutates jobs (_rendered)."""
    ensure_turbo_lora(cfg)
    to_render = [j for j in jobs if not j.get("_rendered")]
    for j in to_render:
        j.pop("_startup_crash", None)
    if not to_render:
        return
    os.makedirs(render_dir, exist_ok=True)
    mname = f"batch_manifest{('_' + label) if label else ''}.json"
    manifest = os.path.abspath(os.path.join(render_dir, mname))
    with open(manifest, "w") as f:
        json.dump([_wgp_task(j, cfg) for j in to_render], f, indent=2)
    raw_dir = os.path.abspath(os.path.join(render_dir, "raw"))
    os.makedirs(raw_dir, exist_ok=True)

    cmd = ([cfg["wgp_python_executable"], cfg["wgp_script"],
           "--process", manifest, "--output-dir", raw_dir]
          + _mmgp_profile_flags(cfg)
          + list(cfg.get("wgp_extra_cli_args") or []))
    logger.info("$ %s  (cwd=%s, conda env: %s)",
                " ".join(cmd), cfg["wgp_repo_dir"], cfg["wgp_conda_env"])
    if cfg.get("wgp_dry_run"):
        logger.info("[dry-run] Not executing wgp.py; manifest at %s", manifest)
        return

    task_start_re = re.compile(r"\[Task (\d+)/(\d+)\]")
    window_re = re.compile(r"Sliding Window (\d+)/(\d+)")
    saved_re = re.compile(r"New video saved to Path:\s*(.+\.mp4)\s*$")
    completed_re = re.compile(r"Task (\d+) completed")
    skipped_re = re.compile(r"\[SKIP\] Task (\d+) failed")

    current = None
    current_window: Dict[int, tuple] = {}
    task_files: Dict[int, str] = {}
    all_seen = set()
    completed, skipped = set(), set()
    returncode = None
    try:
        env = os.environ.copy()
        env.update(cfg.get("wgp_env_overrides") or {})
        for k in (cfg.get("wgp_env_unset") or []):
            env.pop(k, None)
        proc = subprocess.Popen(["bash", "-c", _conda_wrapped(cmd, cfg)],
                                cwd=cfg["wgp_repo_dir"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        for line in proc.stdout:
            print(line, end="")
            m = task_start_re.search(line)
            if m:
                current = int(m.group(1))
                if 1 <= current <= len(to_render):
                    j = to_render[current - 1]
                    logger.info("[wgp] >>> Task %d/%s started: shot %s "
                                "(%d frames, %.1fs).",
                                current, m.group(2), j["shot_id"],
                                j["num_frames"],
                                float(j.get("target_seconds",
                                            j.get("duration", 0))))
            m = window_re.search(line)
            if m and current is not None:
                current_window[current] = (int(m.group(1)), int(m.group(2)))
            m = saved_re.search(line)
            if m and current is not None:
                # Rename IMMEDIATELY so wgp_render/raw stays readable while
                # the batch is still running. The LAST save seen inside a
                # task's block wins (final sliding window = complete clip).
                path = m.group(1).strip()
                win_no, win_total = current_window.get(current, (1, 1))
                if 1 <= current <= len(to_render):
                    path = _rename_live(path, to_render[current - 1],
                                        win_no, win_total)
                task_files[current] = path
                all_seen.add(path)
            m = completed_re.search(line)
            if m:
                completed.add(int(m.group(1)))
            m = skipped_re.search(line)
            if m:
                skipped.add(int(m.group(1)))
        proc.wait(timeout=cfg.get("wgp_call_timeout_seconds", 14400))
        returncode = proc.returncode
    except Exception as e:
        logger.error("[wgp] Error running wgp.py: %s", e)
        try:
            if proc is not None and proc.poll() is None:
                logger.error("[wgp] Monitor failed mid-run -- killing the "
                             "wgp process so retries never stack a second "
                             "copy on top of it.")
                proc.kill()
                proc.wait(timeout=30)
        except Exception:
            pass

    if returncode not in (0, None) and not completed:
        logger.error("[wgp] wgp.py exited %s before any task completed -- "
                     "startup crash; fix the environment and rerun.",
                     returncode)
        for j in to_render:
            j["_rendered"] = False
            j["_startup_crash"] = True
        return

    fps = float(cfg.get("h3_fps", 24.0))
    for idx, j in enumerate(to_render):
        task_no = idx + 1
        produced = task_files.get(task_no)
        if not produced or not os.path.exists(produced):
            reason = ("validation failed" if task_no in skipped
                      else "no output parsed")
            logger.warning("  Shot %s (task %d): %s.", j["shot_id"],
                           task_no, reason)
            j["_rendered"] = False
            continue
        os.makedirs(os.path.dirname(j["output_path"]), exist_ok=True)
        shutil.move(produced, j["output_path"])
        conform_clip(j["output_path"],
                     float(j.get("target_seconds", j.get("duration", 0))),
                     fps)
        j["_rendered"] = True
        logger.info("  Shot %s (task %d) -> %s", j["shot_id"], task_no,
                    os.path.basename(j["output_path"]))

    if cfg.get("wgp_cleanup_intermediate_window_files", True):
        selected = set(task_files.values())
        for f in all_seen - selected:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError:
                pass


def _render_with_retries(jobs: List[Dict], render_dir: str, cfg: Dict,
                         label: str = "") -> None:
    max_attempts = max(1, int(cfg.get("wgp_max_render_attempts", 3)))
    sleep_s = float(cfg.get("wgp_retry_sleep_seconds", 20))
    for attempt in range(1, max_attempts + 1):
        pending = [j for j in jobs if not j.get("_rendered")]
        if not pending:
            break
        if attempt > 1:
            logger.info("[render] Retry pass %d/%d: %d shot(s) pending.",
                        attempt, max_attempts, len(pending))
            time.sleep(sleep_s)
        _render_batch(jobs, render_dir, cfg, label=label)
        if any(j.get("_startup_crash") for j in jobs if not j.get("_rendered")):
            logger.error("[render] Startup crash -- aborting retries.")
            break
    missing = [j["shot_id"] for j in jobs if not j.get("_rendered")]
    if missing:
        logger.warning("[render] %d shot(s) failed: %s. Rerun produce (or "
                       "regenerate_shots) to backfill; finished shots are "
                       "reused.", len(missing), ", ".join(missing))


def render_all(jobs: List[Dict], dirs: Dict, cfg: Dict,
               scenes: Optional[List[Dict]] = None) -> None:
    """Scene-job scheduler: two voice waves. Wave 1 renders silent scenes
    and each character's establishing scene (their first solo-voiced
    scene); voices are harvested from those clips into the bank; wave 2
    renders everything else with banked voice references attached, so
    voices stay consistent ACROSS scenes (within a scene, one continuous
    generation keeps them consistent by construction)."""
    voice_dir = dirs["voices"]
    os.makedirs(voice_dir, exist_ok=True)
    bank: Dict[str, str] = {}
    # Reload any previously harvested voices (resume support).
    if os.path.isdir(voice_dir):
        for f in os.listdir(voice_dir):
            if f.endswith(".wav") and not f.startswith("combo_"):
                bank[os.path.splitext(f)[0]] = os.path.join(voice_dir, f)
    _mark_establishing(jobs)
    # Backfill companion speaker frames for voices banked by earlier runs
    # (wav present, png missing): grab a frame from any already-rendered
    # clip featuring that speaker so resumed projects keep voice refs.
    for sp, wav in list(bank.items()):
        png = os.path.splitext(wav)[0] + ".png"
        if os.path.exists(png):
            continue
        for j in jobs:
            if sp in j.get("speakers", []) and j.get("_rendered") \
                    and os.path.exists(j["output_path"]):
                if extract_speaker_frame(j["output_path"], png):
                    logger.info("[render] Backfilled speaker frame for "
                                "%s from %s.", sp,
                                os.path.basename(j["output_path"]))
                break
    waves = _plan_waves(jobs, cfg)
    for w_i, wave in enumerate(waves, 1):
        if cfg.get("voice_chaining", True) and w_i > 1:
            _attach_voice_refs(wave, bank, voice_dir, cfg)
        todo = [j for j in wave if not j.get("_rendered")]
        logger.info("[render] === Wave %d/%d: %d scene(s), %d to render "
                    "===", w_i, len(waves), len(wave), len(todo))
        if todo:
            _render_with_retries(todo, dirs["render"], cfg,
                                 label=f"wave{w_i}")
        if cfg.get("voice_chaining", True):
            _post_round_voice_harvest(wave, voice_dir, cfg, bank)
        _write_render_status(jobs, dirs)
    _write_render_status(jobs, dirs)


def _post_round_voice_harvest(round_jobs, voice_dir, cfg, bank):
    _harvest_voices(round_jobs, voice_dir, cfg, bank)
    for j in round_jobs:
        sp = j.get("_establishing_for")
        if sp:
            safe = re.sub(r"[^A-Za-z0-9_-]", "_", sp)
            if safe in bank and sp not in bank:
                bank[sp] = bank[safe]


def _write_render_status(jobs: List[Dict], dirs: Dict) -> None:
    """Live per-shot status file (wgp_render/render_status.json), refreshed
    after every wave so a long batch can be monitored from another terminal:
        watch -n 30 'python -c "import json;d=json.load(open(...));..."'
    or just opened in an editor. Also human-scannable as plain text."""
    import datetime
    status = {
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        "total": len(jobs),
        "rendered": sum(1 for j in jobs if j.get("_rendered")),
        "scenes": {j["shot_id"]: {
            "rendered": bool(j.get("_rendered")),
            "windows": j.get("n_windows", 1),
            "seconds": j.get("duration"),
            "source": ("Z-Image still" if j.get("image_ref")
                       else "text-only identity lock"),
            "clip": j.get("output_path") if j.get("_rendered") else None,
        } for j in jobs},
    }
    try:
        with open(os.path.join(dirs["render"], "render_status.json"),
                  "w") as f:
            json.dump(status, f, indent=1)
    except OSError:
        pass
