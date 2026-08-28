"""
film_maker.storyboard
=====================
The "whole film at a glance" context, ported from the music pipeline's
storyboard-summary fix for batching blindness. One compact line per scene
(and, once authored, a rolling digest of written action + key dialogue) that
gets threaded into EVERY sequential or batched call:

  * scene authoring       — sees the full film map (what came before, what's
                            coming) plus everything already written, so
                            setups land, callbacks connect, and motifs build
                            instead of each scene being written in a tunnel.
  * shot-composition plan — each batch sees the film map + all assignments
                            made so far, so variety rules span batches.
  * image prompts         — each scene's call sees the film map, so imagery
                            can echo and escalate deliberately.
  * motion + soundscapes  — each scene's call sees the film map plus the
                            most recent soundscapes, so the sound design
                            develops across scenes instead of resetting.

Kept to one line per scene so it stays cheap to include in full on every
call regardless of film length; the authored digest is token-clamped.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .llm import clamp_text


def scene_storyboard(scenes: List[Dict], mark_scene_id: Optional[int] = None,
                     header: str = "") -> str:
    """One compact line per scene for the whole film. mark_scene_id gets a
    '>>' marker so a per-scene call knows exactly where it sits."""
    lines = [header] if header else []
    for s in scenes:
        tag = ">>" if s.get("scene_id") == mark_scene_id else "  "
        loc = (s.get("location") or s.get("location_hint") or "")[:34]
        extras = []
        if s.get("long_take"):
            extras.append("ONER")
        if s.get("connection_note"):
            extras.append("connection")
        if s.get("eureka_note"):
            extras.append("eureka")
        extra = f" [{'/'.join(extras)}]" if extras else ""
        lines.append(
            f"{tag}SC{int(s.get('scene_id', 0)):02d} act{s.get('act','?')} "
            f"[{s.get('function','')}] T={float(s.get('tension',0)):.2f} "
            f"@ {loc} :: {(s.get('summary') or '')[:95]} "
            f"(emotion: {s.get('emotion','')}){extra}")
    return "\n".join(lines)


def authored_digest(scenes: List[Dict], upto_scene_id: Optional[int] = None,
                    max_tokens: int = 3200) -> str:
    """Rolling digest of scenes ALREADY WRITTEN: their staged actions and up
    to two notable dialogue lines each -- the material later scenes can
    echo, call back to, or deliberately contrast. Token-clamped so a long
    film can't balloon the context."""
    lines = []
    for s in scenes:
        if upto_scene_id is not None and \
                int(s.get("scene_id", 0)) >= int(upto_scene_id):
            break
        shots = s.get("shots") or []
        if not shots:
            continue
        acts = "; ".join((sh.get("action") or "")[:60]
                         for sh in shots[:3] if sh.get("action"))
        lines.append(f"SC{int(s.get('scene_id', 0)):02d}: {acts}")
        dlg = [f'{ln.get("speaker","")}: "{ln.get("text","")}"'
               for sh in shots for ln in sh.get("lines", [])
               if ln.get("text")][:2]
        for d in dlg:
            lines.append(f"    {d}")
    if not lines:
        return ""
    return clamp_text("\n".join(lines), max_tokens, where="authored_digest")


def recent_soundscapes(scenes: List[Dict], upto_scene_id: int,
                       n: int = 4) -> str:
    """The last n soundscapes already designed, so the next scene's sound
    develops from them instead of resetting or repeating."""
    prior = []
    for s in scenes:
        if int(s.get("scene_id", 0)) >= int(upto_scene_id):
            break
        for sh in s.get("shots") or []:
            if sh.get("soundscape"):
                prior.append(f"SC{int(s.get('scene_id',0)):02d}/"
                             f"{sh.get('shot_id','')}: {sh['soundscape'][:90]}")
    return "\n".join(prior[-n:])
