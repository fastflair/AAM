"""
film_maker.cinematography
=========================
The visual + sonic design layer. Ports the music pipeline's strongest ideas
(locked style bible, entity locks, globally-planned shot composition,
tension-band camera grammar) and the comic pipeline's period lock, then adds
the film-specific pieces:

  * STYLE BIBLE       — one locked medium/palette/lighting/film-grammar for
                        every frame, generated per film from story+registers.
  * LOCATION DESIGN   — scene-graph-lite: each beat's location becomes a
                        designed set with a signature detail and a LOCKED
                        visual description; returning to a location reuses
                        its lock verbatim (spatial continuity for free).
  * WARDROBE PLAN     — per-character costume states across the film (act
                        changes, damage, weather) layered over the base
                        visual_lock so appearance evolves with the story
                        instead of resetting every scene.
  * SHOT PLAN         — global scale/angle/time-of-day/weather per shot with
                        the same anti-repetition rules that fixed the music
                        pipeline's batching blindness.
  * IMAGE PROMPTS     — structured Subject/Attire/Action/Environment/
                        Lighting/Style stills prompts with entity+location
                        locks reproduced verbatim.
  * SOUNDSCAPE DESIGN — per-shot overall_soundscape written under the
                        registers' sound-design methods, with per-location
                        signature ambiences reused for sonic continuity and
                        threat/wonder motifs planted across scenes.
  * PERIOD SCRUB      — era lock + LLM anachronism rewrite of any prompt
                        text that leaks modern objects into a period world.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from .llm import get_llm, safe_json_dict, safe_json_list, clean_text, logger
from .registers import (camera_grammar_block, sound_method_block,
                        visual_bias_block, content_rating_block)

SHOT_SCALE_POOL = ["extreme close-up", "close-up", "medium close-up",
                   "medium shot", "medium-wide shot", "wide shot",
                   "extreme wide shot"]
CAMERA_ANGLE_POOL = ["eye-level", "low-angle", "high-angle", "dutch tilt",
                     "over-the-shoulder", "point-of-view"]


# ---------------------------------------------------------------------------
# Style bible
# ---------------------------------------------------------------------------
DEFAULT_STYLE_BIBLE = {
    "medium": "cinematic 3D animation with painterly light, hyperreal detail",
    "palette": "deep teal shadows, warm amber key light, desaturated midtones",
    "lighting": "motivated single-source lighting, volumetric atmosphere, "
                "high-contrast chiaroscuro",
    "film_grammar": "anamorphic lens character, shallow depth of field, "
                    "subtle film grain, one clear focal subject per frame",
}


def _as_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x).strip() for x in v if x)
    return str(v).strip()


def generate_style_bible(story: Dict, cfg: Dict) -> Dict:
    logger.info("[cine] Generating the locked style bible...")
    registers = story.get("registers") or ["drama"]
    from .styles import visual_style_block
    vstyle = visual_style_block(cfg)
    prompt = f"""
You are the production designer + DP. Lock ONE visual language for this
entire animated film -- every frame renders in it; cohesion comes from the
LOOK while places and subjects vary.

Film: {story.get('title','')} -- {story.get('logline','')}
World: {story.get('world','')}   Era: {story.get('era','')}
Mood: {story.get('mood','')}   Themes: {', '.join(story.get('themes', []))}
{vstyle}
{visual_bias_block(registers)}

Return ONLY raw JSON, values short and concrete:
{{"medium": "<ONE rendering medium for every shot, committed>",
  "palette": "<3-4 signature colors + the grade>",
  "lighting": "<one recurring lighting philosophy>",
  "film_grammar": "<shared lens/camera texture language>",
  "opening_image": "<what the film's very first frame should feel like>",
  "closing_image": "<what the last frame should feel like, rhyming with or
                    inverting the opening>"}}
"""
    bible = safe_json_dict(get_llm(prompt, temperature=0.6))
    for k, v in DEFAULT_STYLE_BIBLE.items():
        bible[k] = _as_str(bible.get(k)) or v
    # Author-locked visual style is a hard guarantee: if the LLM's medium
    # drifted or the call failed, anchor the medium on the style itself.
    from .styles import _resolve, VISUAL_STYLES
    locked = _resolve(cfg.get("visual_style", ""), VISUAL_STYLES)
    if locked:
        med = bible.get("medium", "").lower()
        probe = " ".join(locked.split()[:3]).lower()
        if probe.split()[0] not in med:
            bible["medium"] = locked
    for k in ("opening_image", "closing_image"):
        bible[k] = _as_str(bible.get(k))
    logger.info("[cine] Style: %s", bible["medium"])
    return bible


def style_block(style_bible: Dict) -> str:
    lines = ["LOCKED VISUAL LANGUAGE (render every frame in exactly this; "
             "diversity comes from PLACE and SUBJECT, never the look):"]
    for key, label in (("medium", "Medium"), ("palette", "Palette"),
                       ("lighting", "Lighting"), ("film_grammar", "Cinematography")):
        v = _as_str(style_bible.get(key))
        if v:
            lines.append(f"- {label}: {v}")
    return "\n".join(lines)


def style_image_prefix(style_bible: Dict) -> str:
    parts = [_as_str(style_bible.get(k)) for k in ("medium", "palette", "lighting")]
    return (", ".join(p for p in parts if p) +
            ", a single clear focal composition suitable as a video reference "
            "image (no split panels, no collage, no text)")


# ---------------------------------------------------------------------------
# Location design (scene-graph-lite with locked set descriptions)
# ---------------------------------------------------------------------------
def design_locations(story: Dict, scenes: List[Dict], cfg: Dict) -> Dict[str, Dict]:
    """Cluster the beats' location hints into designed SETS. Each set gets a
    canonical name, a locked 40-70 word visual description reproduced
    verbatim on every visit, a signature detail, and a signature ambience
    (its sonic identity). Returns {canonical_name: set_dict} and stamps each
    scene with scene["location"] = canonical name."""
    logger.info("[cine] Designing locations...")
    hints = [{"scene_id": s["scene_id"], "hint": s.get("location_hint", ""),
              "summary": s.get("summary", "")[:120]} for s in scenes]
    prompt = f"""
You are the location designer. Cluster these scenes into a small set of
DESIGNED locations (reuse a location for scenes that plausibly share it --
returning to a place is powerful; a new set per scene is a slideshow).
Aim for roughly {max(4, len(scenes) // 3)} distinct sets. Consecutive scenes
in DIFFERENT sets should contrast in at least two of: interior/exterior,
scale, light quality, palette accent.

Film: {story.get('logline','')}  World: {story.get('world','')}
Era: {story.get('era','')} (everything must belong to this era)

Scenes:
{json.dumps(hints, indent=1)}

Return ONLY raw JSON:
{{"locations": [
   {{"name": "<short canonical set name>",
     "lock": "<40-70 words of purely visual set description: architecture or
              terrain, materials, signature light behavior, and ONE
              unmistakable signature detail that appears in every shot
              here>",
     "ambience": "<the set's sonic identity: 8-20 words of the ambient bed
                  heard whenever we are here>",
     "scene_ids": [<ints>]}}
 ]}}
"""
    data = safe_json_dict(get_llm(prompt, temperature=0.75, large=True))
    sets: Dict[str, Dict] = {}
    assigned = {}
    for loc in (data.get("locations") or []):
        name = clean_text(str(loc.get("name", ""))).strip()
        if not name:
            continue
        sets[name] = {"name": name,
                      "lock": clean_text(str(loc.get("lock", ""))),
                      "ambience": clean_text(str(loc.get("ambience", ""))),
                      "scene_ids": []}
        for sid in (loc.get("scene_ids") or []):
            try:
                sid = int(sid)
                assigned[sid] = name
                sets[name]["scene_ids"].append(sid)
            except (TypeError, ValueError):
                pass
    for s in scenes:
        name = assigned.get(s["scene_id"])
        if not name:                      # fallback: hint becomes its own set
            name = (s.get("location_hint") or f"Set {s['scene_id']}")[:40]
            sets.setdefault(name, {"name": name,
                                   "lock": s.get("location_hint", ""),
                                   "ambience": ""})
        s["location"] = name
    logger.info("[cine] %d designed sets across %d scenes.",
                len(sets), len(scenes))
    return sets


# ---------------------------------------------------------------------------
# Wardrobe plan (appearance continuity at scene granularity)
# ---------------------------------------------------------------------------
def plan_wardrobe(story: Dict, scenes: List[Dict], cfg: Dict) -> Dict:
    """Per-character costume STATE changes across the film (act shifts,
    damage, weather, undress/redress) so appearance evolves with the story.
    Returns {character: [{from_scene, state}]}; resolve with wardrobe_for()."""
    logger.info("[cine] Planning wardrobe continuity...")
    chars = story.get("characters") or []
    scene_digest = "\n".join(
        f"{s['scene_id']}. act{s.get('act')} @ {s.get('location','')}: "
        f"{s.get('summary','')[:110]}" for s in scenes)
    char_digest = "\n".join(f"- {c['name']}: base look = {c['visual_lock'][:150]}"
                            for c in chars)
    prompt = f"""
Plan costume continuity for the film. Each character starts in their base
look. Introduce a SMALL number of justified state changes (2-4 per principal
across the whole film): a coat gained for weather, damage after an ordeal,
formal wear for an occasion, dishevelment at the low point. Every change
must be motivated by an on-screen event and persist until changed again.

Characters:
{char_digest}

Scenes:
{scene_digest}

Return ONLY raw JSON:
{{"wardrobe": [
   {{"character": "<name>", "from_scene": <scene_id>,
     "state": "<15-30 words describing the changed elements only, layered
              over the base look>",
     "reason": "<the on-screen motivation>"}}
 ]}}
"""
    data = safe_json_dict(get_llm(prompt, temperature=0.6))
    plan: Dict[str, List[Dict]] = {}
    for w in (data.get("wardrobe") or []):
        name = clean_text(str(w.get("character", "")))
        try:
            frm = int(w.get("from_scene", 1))
        except (TypeError, ValueError):
            continue
        state = clean_text(str(w.get("state", "")))
        if name and state:
            plan.setdefault(name, []).append({"from_scene": frm, "state": state})
    for v in plan.values():
        v.sort(key=lambda x: x["from_scene"])
    return plan


def wardrobe_for(character: str, scene_id: int, wardrobe: Dict) -> str:
    state = ""
    for change in wardrobe.get(character, []):
        if change["from_scene"] <= scene_id:
            state = change["state"]
    return state


# ---------------------------------------------------------------------------
# Global shot composition plan
# ---------------------------------------------------------------------------
def plan_shot_compositions(story: Dict, scenes: List[Dict], cfg: Dict) -> None:
    """Assign scale/angle/time-of-day/weather per shot with film-wide
    anti-repetition. BATCHED (a 20-minute film has 130+ shots; one response
    carrying 130+ objects reliably truncates): batches run in order, and
    each batch receives a recap of the assignments made so far -- the tail
    of recent picks plus (scale, angle) pair usage counts -- so the global
    variety rules survive batching instead of each batch going blind."""
    from .llm import batched_json_call
    flat = []
    for s in scenes:
        for sh in s.get("shots", []):
            flat.append({"i": len(flat), "scene": s["scene_id"],
                         "loc": s.get("location", "")[:40],
                         "action": sh.get("action", "")[:100],
                         "band": sh.get("tension_band", "mid"),
                         "framing_hint": sh.get("framing_hint", "")[:60],
                         "has_dialogue": bool(sh.get("lines"))})
    if not flat:
        return
    logger.info("[cine] Planning %d shot compositions (batched)...", len(flat))
    from .styles import camera_style_block as _csb
    _cam_style = _csb(cfg)
    from .storyboard import scene_storyboard
    film_map = scene_storyboard(
        scenes, header="FULL FILM STORYBOARD (the arc your coverage serves; "
                       "coverage should tighten and intensify with the "
                       "tension curve across the WHOLE film):")

    def _context(results: Dict) -> str:
        if not results:
            return (film_map + "\n\n(no shots assigned yet -- this is the "
                    "film's opening)")
        done = sorted((k, v) for k, v in results.items() if isinstance(k, int))
        tail = done[-10:]
        tail_lines = "\n".join(
            f"  shot {i}: {v.get('shot_scale','')} / {v.get('camera_angle','')}"
            f" / {v.get('time_of_day','')} / {v.get('weather','')}"
            for i, v in tail)
        pair_counts: Dict[str, int] = {}
        for _, v in done:
            key = f"{v.get('shot_scale','')} + {v.get('camera_angle','')}"
            pair_counts[key] = pair_counts.get(key, 0) + 1
        heavy = sorted(pair_counts.items(), key=lambda kv: -kv[1])[:8]
        heavy_lines = "\n".join(f"  {k}: used {n}x" for k, n in heavy)
        return (f"{film_map}\n\n"
                f"ASSIGNMENTS ALREADY MADE ({len(done)} shots). The LAST 10 "
                f"(your first shot must not repeat the final one's scale+"
                f"angle):\n{tail_lines}\n"
                f"(scale + angle) pairs already used most (avoid adding to "
                f"the heaviest):\n{heavy_lines}")

    def _prompt(batch, ctx):
        return f"""
You are the DP planning coverage for a film, working through it in
sequential batches so the finished edit reads as continuously varied
camerawork ACROSS THE WHOLE FILM, not just within this batch.

{ctx}

Assign every shot below:
- "shot_scale": one of {', '.join(SHOT_SCALE_POOL)}
- "camera_angle": one of {', '.join(CAMERA_ANGLE_POOL)}
- "time_of_day": dawn|morning|midday|golden hour|dusk|blue hour|night
- "weather": short atmosphere phrase ("still and clear", "light rain",
  "drifting fog", "heat shimmer", ...)

HARD RULES (they span batch boundaries -- use the recap above):
{_cam_style}
- No two CONSECUTIVE shots share both shot_scale AND camera_angle.
- No (scale, angle) pairing repeats more than once per 6 shots; steer away
  from the film's most-used pairs listed above.
- Shots WITH dialogue need the speaker's face readable: close-up through
  medium-wide only, mouth unobstructed, angle no more extreme than a gentle
  low/high -- lip-sync must read.
- time_of_day and weather stay CONSTANT within a scene (same scene number)
  unless the story motivates a change; they may shift between scenes -- and
  must stay consistent with what the recap already assigned to this scene.
- Higher tension -> generally tighter/more dynamic; lower -> wider/calmer;
  variety wins ties. Honor each shot's framing_hint where present.
- Coverage grammar within a scene: establish wide early, tighten as tension
  rises, save the tightest frame for the scene's key emotional beat.

Shots:
{json.dumps(batch, indent=0)}

Return ONLY a raw JSON array, one object per shot in order:
{{"i": <copied int>, "shot_scale": "...", "camera_angle": "...",
  "time_of_day": "...", "weather": "..."}}
"""

    plan = batched_json_call(
        flat, _prompt, id_key="i",
        batch_size=int(cfg.get("shot_plan_batch_size", 36)),
        temperature=0.8, large=True, context_fn=_context, label="shots")
    k = 0
    for s in scenes:
        for sh in s.get("shots", []):
            it = plan.get(k, {})
            dialog = bool(sh.get("lines"))
            fallback_scales = (["close-up", "medium close-up", "medium shot",
                                "medium-wide shot"] if dialog else SHOT_SCALE_POOL)
            sh["shot_scale"] = clean_text(str(it.get("shot_scale", ""))) or \
                fallback_scales[k % len(fallback_scales)]
            sh["camera_angle"] = clean_text(str(it.get("camera_angle", ""))) or \
                CAMERA_ANGLE_POOL[(k * 3) % len(CAMERA_ANGLE_POOL)]
            sh["time_of_day"] = clean_text(str(it.get("time_of_day", ""))) or "day"
            sh["weather"] = clean_text(str(it.get("weather", ""))) or "still and clear"
            k += 1


# ---------------------------------------------------------------------------
# Entity block (visual locks + wardrobe overlay)
# ---------------------------------------------------------------------------
def entity_bible_for(story: Dict) -> Dict[str, Dict]:
    return {c["name"]: {"description": c.get("visual_lock", ""),
                        "type": "character"}
            for c in (story.get("characters") or [])}


def entity_block(shot: Dict, scene: Dict, story: Dict, wardrobe: Dict) -> str:
    present = _chars_in_shot(shot, scene, story)
    if not present:
        return ""
    lines = ["LOCKED CHARACTERS (reproduce each description VERBATIM; never "
             "invent a different look):"]
    for c in present:
        base = c.get("visual_lock", "")
        state = wardrobe_for(c["name"], scene["scene_id"], wardrobe)
        overlay = f" CURRENT STATE (overrides base costume where they touch): {state}" if state else ""
        lines.append(f"- {c['name']}: {base}{overlay}")
    return "\n".join(lines)


def _chars_in_shot(shot: Dict, scene: Dict, story: Dict) -> List[Dict]:
    by_name = {c["name"].lower(): c for c in (story.get("characters") or [])}
    named = set()
    for ln in shot.get("lines", []):
        s = (ln.get("speaker") or "").lower()
        if s in by_name:
            named.add(s)
    text = (shot.get("action", "") + " " + shot.get("framing_hint", "")).lower()
    for nm in by_name:
        if nm in text:
            named.add(nm)
    if not named:
        for nm in (scene.get("characters") or []):
            if nm.lower() in by_name:
                named.add(nm.lower())
    return [by_name[n] for n in named]


# ---------------------------------------------------------------------------
# Image prompts (structured stills, verbatim locks, period-safe)
# ---------------------------------------------------------------------------
def write_image_prompts(scenes: List[Dict], story: Dict, style_bible: Dict,
                        locations: Dict, wardrobe: Dict, cfg: Dict,
                        graph=None) -> None:
    """One LLM call per scene batch writing every shot's still prompt in the
    structured format, with character + location locks reproduced verbatim
    and the global composition honored. Also consumes: the color script
    (scene grade), cut-design transition notes on boundary shots, per-
    character performance physicality from the EI graph, and the style
    bible's opening/closing image rhyme on the film's first/last shots."""
    registers = story.get("registers") or ["drama"]
    era = story.get("era", "present day")
    sblock = style_block(style_bible)

    def _physicality(chars: List[Dict]) -> str:
        if graph is None:
            return ""
        lines = []
        for c in chars:
            node = graph.get_node(c["name"])
            if not node:
                continue
            bits = []
            if getattr(node, "emotion_style", None):
                styles = "; ".join(f"{k} shows as {v}" for k, v in
                                   list(node.emotion_style.items())[:6])
                bits.append(styles)
            else:
                if node.fear_response:
                    bits.append(f"fear shows as {node.fear_response}")
                if node.joy_response:
                    bits.append(f"joy shows as {node.joy_response}")
            if getattr(node, "tell", ""):
                bits.append(f"when their wound is touched, the involuntary "
                            f"tell: {node.tell}")
            if getattr(node, "relic", ""):
                bits.append(f"personal relic sometimes on them or in frame: "
                            f"{node.relic}")
            if getattr(node, "idle_behavior", ""):
                bits.append(f"signature idle behavior their hands/body "
                            f"default to when unoccupied (stage it "
                            f"mid-motion where the shot allows): "
                            f"{node.idle_behavior}")
            if getattr(node, "skill_display", ""):
                bits.append(f"their competence shows in small confident "
                            f"actions: {node.skill_display}")
            se = float(getattr(node, "social_energy", 0) or 0)
            if se >= 2.0:
                bits.append("high social energy: framed toward people, "
                            "often the room's gravitational center, other "
                            "figures' eye-lines drawn to them")
            elif se <= -2.0:
                bits.append("low social energy: framed at edges, angled "
                            "away, attention on objects rather than faces")
            if node.defense_mechanism:
                bits.append(f"under pressure they {node.defense_mechanism}")
            if bits:
                lines.append(f"- {c['name']}: {'; '.join(bits)}")
        if not lines:
            return ""
        block = ("PERFORMANCE PHYSICALITY (emotion lives on THESE bodies in "
                 "THESE personality-specific ways -- use the style matching "
                 "each shot's key_emotion, never generic sadness/joy; a "
                 "suppressor's grief is stillness and busy hands, not "
                 "tears):\n" + "\n".join(lines))
        chem = graph.chemistry_block([c["name"] for c in chars]) \
            if len(chars) > 1 else ""
        if chem:
            block += ("\n" + chem +
                      "\nWhere the shot's moment touches this chemistry, "
                      "make it VISIBLE in the still: a blush caught "
                      "mid-bloom, a smile being fought down, eye contact "
                      "held across the frame, bodies unconsciously angled "
                      "toward each other.")
        return block

    all_shots = [sh for s in scenes for sh in s.get("shots", [])]
    first_id = all_shots[0]["shot_id"] if all_shots else None
    last_id = all_shots[-1]["shot_id"] if all_shots else None
    from .storyboard import scene_storyboard as _sb

    for s_i, scene in enumerate(scenes):
        loc = locations.get(scene.get("location", ""), {})
        shots = scene.get("shots", [])
        if not shots:
            continue
        specs = []
        for sh in shots:
            spec = {
                "shot_id": sh["shot_id"],
                "action": sh.get("action", ""),
                "framing_hint": sh.get("framing_hint", ""),
                "shot_scale": sh.get("shot_scale", ""),
                "camera_angle": sh.get("camera_angle", ""),
                "time_of_day": sh.get("time_of_day", ""),
                "weather": sh.get("weather", ""),
                "key_emotion": sh.get("key_emotion", ""),
                "characters_present": [c["name"] for c in
                                       _chars_in_shot(sh, scene, story)],
            }
            if sh.get("carry_state"):
                spec["continuity_state"] = sh["carry_state"]
            if sh.get("transition_note"):
                spec["transition_note"] = sh["transition_note"]
            if sh["shot_id"] == first_id and style_bible.get("opening_image"):
                spec["film_opening_image"] = style_bible["opening_image"]
            if sh["shot_id"] == last_id and style_bible.get("closing_image"):
                spec["film_closing_image"] = style_bible["closing_image"]
            specs.append(spec)
        eblocks = "\n\n".join(
            dict.fromkeys(entity_block(sh, scene, story, wardrobe)
                          for sh in shots if
                          entity_block(sh, scene, story, wardrobe)))
        phys = _physicality(
            [c for sh in shots for c in _chars_in_shot(sh, scene, story)])
        grade_line = (f"SCENE GRADE (this scene's position in the film's "
                      f"color script; realize it in the light and grade of "
                      f"every shot here): {scene.get('grade','')}"
                      if scene.get("grade") else "")
        prompt = f"""
Write the reference-still image prompt for every shot of this scene. Each
still seeds one animated take, so it must show the shot's action MID-BEAT
(caught in motion, not posed), composed exactly to its assigned scale/angle,
with faces of any speaking characters clearly visible and mouths
unobstructed.

{_sb(scenes, mark_scene_id=scene['scene_id'],
     header='FULL FILM STORYBOARD (this scene is marked >>; let imagery '
            'echo and escalate deliberately across the film -- planted '
            'images should recur transformed, never duplicated by '
            'accident):')}

{sblock}
{grade_line}

{eblocks or 'No locked characters in this scene.'}

{phys}

LOCKED LOCATION -- "{loc.get('name', scene.get('location',''))}" (build the
Environment from this, verbatim details intact; its signature detail
appears in every shot here):
{loc.get('lock','')}

ERA LOCK: {era}. Only era-appropriate technology, materials, dress, and
props may appear anywhere in the prompt.
{visual_bias_block(registers)}
{content_rating_block(cfg.get('content_rating', 'teen'))}

Scene: {scene.get('summary','')[:250]}

For each shot return a single dense prompt in this exact labelled structure:
"Subject: ... Attire: ... Action: ... Environment: ... Lighting: ...
Style Details: ..."
- Subject: the locked character descriptions VERBATIM for everyone present
  (or the environment itself for empty frames), with the shot's emotion
  visible in posture and face -- expressed through that character's own
  performance physicality where given.
- Action: the shot's action frozen mid-beat.
- Environment: the locked location, its signature detail, time_of_day and
  weather realized in light and atmosphere, graded per the scene grade.
- Lighting/Style: the locked style bible language, inflected by the grade.
- Shots carrying a "continuity_state" list PHYSICAL FACTS that must be
  visibly true in the frame (a torn right sleeve, mud to the knees, the
  chair still overturned, the letter clutched in her left hand). Render
  them EXACTLY -- they override any conflicting detail in the base
  character lock, and omitting them breaks the film's continuity.
- Shots carrying a "transition_note" are boundary shots: compose them to
  honor the note (a match-cut echo, a contrast collision, an intentional
  final/opening frame).
- A shot carrying "film_opening_image" or "film_closing_image" is the
  film's very first or very last frame: realize that intention exactly --
  these two frames should rhyme across the whole film.
- No text, captions, or watermarks anywhere.

Shots:
{json.dumps(specs, indent=1)}

Return ONLY a raw JSON array, one object per shot in order:
{{"shot_id": "<copied>", "image_prompt": "<the structured prompt>"}}
"""
        items = safe_json_list(get_llm(prompt, temperature=0.8, large=True))
        by_id = {it.get("shot_id"): it for it in items if isinstance(it, dict)}
        ent_bible = entity_bible_for(story)
        for sh in shots:
            it = by_id.get(sh["shot_id"], {})
            p = clean_text(str(it.get("image_prompt", "")), ent_bible)
            if not p or len(p) < 80:
                p = _fallback_image_prompt(sh, scene, story, style_bible,
                                           loc, wardrobe)
            sh["image_prompt"] = p
        logger.info("[cine] Image prompts written for scene %d (%d shots).",
                    scene["scene_id"], len(shots))
    scrub_anachronisms(scenes, story, cfg)


def _fallback_image_prompt(sh, scene, story, style_bible, loc, wardrobe) -> str:
    chars = _chars_in_shot(sh, scene, story)
    subject = "; ".join(
        f"{c['name']}: {c.get('visual_lock','')}"
        f"{(' ' + wardrobe_for(c['name'], scene['scene_id'], wardrobe)) if wardrobe_for(c['name'], scene['scene_id'], wardrobe) else ''}"
        for c in chars) or f"the environment of {loc.get('name','the scene')}"
    return (f"Subject: {subject}. Action: {sh.get('action','')} "
            f"Environment: {loc.get('lock','')} at {sh.get('time_of_day','day')}, "
            f"{sh.get('weather','')}. Lighting: {_as_str(style_bible.get('lighting'))}. "
            f"Style Details: {_as_str(style_bible.get('medium'))}, "
            f"{_as_str(style_bible.get('film_grammar'))}. "
            f"{sh.get('shot_scale','medium shot')}, {sh.get('camera_angle','eye-level')}. "
            f"No text, no watermark.")


def scrub_anachronisms(scenes: List[Dict], story: Dict,
                       cfg: Dict = None) -> None:
    """Period lock: for non-modern eras, rewrite any leaked modern object in
    the image prompts to a period equivalent. BATCHED -- a 130-shot film's
    prompt list is far too large for one call. No repair passes needed: an
    item the model doesn't return is an item with nothing to fix."""
    era = (story.get("era") or "present day").lower()
    if any(w in era for w in ("present", "modern", "contemporary", "today",
                              "future", "202", "21st")):
        return
    logger.info("[cine] Period scrub for era '%s' (batched)...",
                story.get("era"))
    flat = [(sh, sh.get("image_prompt", "")) for s in scenes
            for sh in s.get("shots", [])]
    batch_size = int((cfg or {}).get("scrub_batch_size", 30))
    for start in range(0, len(flat), batch_size):
        chunk = flat[start:start + batch_size]
        listing = "\n".join(f"[{start + j}] {p[:400]}"
                            for j, (sh, p) in enumerate(chunk))
        prompt = f"""
Era lock: "{story.get('era')}". Scan these image prompts for ANY anachronism
(objects, materials, garments, technology, lighting sources that do not
belong to the era) and return rewrites ONLY for prompts that contain one,
replacing the anachronism with a period-true equivalent and leaving
everything else in the prompt untouched.

{listing}

Return ONLY a raw JSON array (empty if nothing to fix):
{{"i": <int>, "image_prompt": "<full corrected prompt>"}}
"""
        for fx in safe_json_list(get_llm(prompt, temperature=0.3, large=True)):
            i = fx.get("i")
            new = clean_text(str(fx.get("image_prompt", "")))
            if isinstance(i, int) and 0 <= i < len(flat) and len(new) > 80:
                flat[i][0]["image_prompt"] = new


# ---------------------------------------------------------------------------
# Physical-state ledger (the anti-slop continuity pass)
# ---------------------------------------------------------------------------
def track_physical_state(scenes: List[Dict], story: Dict, cfg: Dict) -> None:
    """Walk the authored film in order and maintain a cumulative PHYSICAL
    STATE ledger: costume damage (a torn sleeve STAYS torn), dirt, wounds,
    wetness, held/dropped props, moved furniture, weather effects. Each shot
    is stamped with sh['carry_state'] -- the state that must be TRUE at that
    moment -- which is then written into both the still prompts and the H3
    prompts, so the text and the chained frames agree. State persists until
    an on-screen event changes it (a sleeve mends only if we see it mended;
    a coat returns only if we see it put on)."""
    logger.info("[cine] Tracking physical continuity state across the film...")
    rolling = "(film opens; every character in their base look; no damage, "\
              "nothing held, sets undisturbed)"
    for scene in scenes:
        shots = scene.get("shots", [])
        if not shots:
            continue
        specs = [{"shot_id": sh["shot_id"],
                  "action": sh.get("action", "")[:220]} for sh in shots]
        prompt = f"""
You are the on-set continuity supervisor. Maintain the cumulative PHYSICAL
STATE of characters and set as this scene's shots play out in order.

STATE CARRIED IN from everything before this scene:
{rolling}

Scene {scene.get('scene_id')} @ {scene.get('location','')}:
{scene.get('summary','')[:200]}

Shots in order:
{json.dumps(specs, indent=1)}

RULES:
- State PERSISTS until an on-screen action changes it: a ripped sleeve stays
  ripped for the rest of the film unless we see it repaired or the garment
  changed; mud, blood, rain-soak, a bandage, a dropped or picked-up object,
  an overturned chair -- all carry forward.
- For EACH shot, state the physical facts that must be visibly true AT THAT
  MOMENT (after the previous shots' actions, before/while this shot's
  action plays). Keep each under 40 words, concrete and visual, only facts
  that differ from a clean base look ('' when nothing differs).
- Then state the scene's END state to carry into the next scene, under 60
  words.

Return ONLY raw JSON:
{{"shots": [{{"shot_id": "<copied>", "carry_state": "<facts or ''>"}}],
  "end_state": "<state carried out of this scene>"}}
"""
        data = safe_json_dict(get_llm(prompt, temperature=0.3))
        by_id = {it.get("shot_id"): clean_text(str(it.get("carry_state", "")))
                 for it in (data.get("shots") or []) if isinstance(it, dict)}
        for sh in shots:
            sh["carry_state"] = by_id.get(sh["shot_id"], "")
        end = clean_text(str(data.get("end_state", "")))
        if end:
            rolling = end
        scene["end_state"] = end or rolling


# ---------------------------------------------------------------------------
# Long-take phase authoring (the anti-restart fix)
# ---------------------------------------------------------------------------
def estimate_chain_segment_count(duration_seconds: float, cfg: Dict) -> int:
    """How many <=1-window segments a shot of this duration will render as
    in chain mode. Kept in sync with render.ceil_to_grid's single-window cap
    so planning-time phase counts match render-time segment counts."""
    fps = float(cfg.get("h3_fps", 24.0))
    total_frames = max(1, int(round(duration_seconds * fps)))
    single_cap = int(cfg.get("wgp_video_length_max_frames",
                             cfg.get("wgp_frames_maximum", 737)))
    if total_frames <= single_cap:
        return 1
    return -(-total_frames // single_cap)  # ceil


def estimate_phase_count(duration_seconds: float, cfg: Dict) -> int:
    """How many progressive PHASES a long take needs, matching how the
    scene generation will actually split it at render time:
      * "windowed" / "sliding_window" -- one phase per SLIDING WINDOW
        (window count depends on window size AND overlap, so it differs
        from the chain segment count for the same duration).
      * "chain" -- one phase per explicit segment.
    Keeping planning-time phase counts equal to render-time window/segment
    counts is what lets every window carry its own progressive prompt with
    a clean 1:1 mapping (no proportional spreading fallback)."""
    from .render import estimate_window_count, ceil_to_grid
    fps = float(cfg.get("h3_fps", 24.0))
    total_frames = max(1, int(round(duration_seconds * fps)))
    single_cap = int(cfg.get("wgp_windowed_window_frames") or
                     cfg.get("wgp_video_length_max_frames",
                             cfg.get("wgp_frames_maximum", 737)))
    if total_frames <= single_cap:
        return 1
    render_frames = ceil_to_grid(
        total_frames, cfg,
        max_frames=total_frames + int(cfg.get("wgp_frames_steps", 17)))
    return estimate_window_count(render_frames, cfg)


def plan_long_take_phases(scenes: List[Dict], story: Dict, cfg: Dict) -> None:
    """THE FIX for the 'action restarts every window' failure: any shot
    long enough to need multiple render windows gets a distinct PROGRESSIVE
    beat written for each window instead of one static description repeated
    across all of them. A door-exit shot doesn't get 'walks out the door'
    three times -- it gets 'reaches for the handle and pulls it open' /
    'steps through the threshold onto the porch' / 'the door swings shut
    behind her as she crosses the yard', each its own phase.

    Stamps shot['chain_phases'] = [{"action", "motion", "line_indices"}]
    aligned to estimate_chain_segment_count(). Dialogue lines are assigned
    to the single phase where they're actually spoken (by index into the
    shot's existing "lines"), so a line is never repeated across segments.
    No-op for shots that fit in one window."""
    max_sec = float(cfg.get("max_shot_seconds", 30.0))
    targets = []
    for scene in scenes:
        for sh in scene.get("shots", []):
            dur = float(sh.get("duration", 0))
            if dur > max_sec + 0.5:
                n = estimate_phase_count(dur, cfg)
                if n > 1:
                    targets.append((scene, sh, n))
    if not targets:
        return
    logger.info("[cine] Authoring progressive phases for %d long-take "
               "shot(s)...", len(targets))
    for scene, sh, n in targets:
        lines = sh.get("lines") or []
        line_listing = "\n".join(
            f"  [{i}] {ln.get('speaker','')}: \"{ln.get('text','')}\""
            for i, ln in enumerate(lines))
        prompt = f"""
This shot is long enough that it will render as {n} SEQUENTIAL segments,
each continuing from the exact last frame of the one before it (the model
is shown that frame and told nothing resets). Your job is to break the
shot's ONE continuous action into {n} PROGRESSIVE beats -- what physically
happens in the FIRST third/quarter of the take, then the NEXT, then the
LAST -- so segment 2 picks up exactly where segment 1's motion stopped and
carries it forward, never re-describing or repeating what segment 1 already
showed. Think of it as the same continuous take described as consecutive
sentences of a single unbroken sports commentary, not {n} separate replays
of the same moment.

Full shot action (the complete arc to break into {n} progressive parts):
{sh.get('action', '')}
Full shot motion/camera (already established -- vary WITHIN this spirit,
one dominant move per phase, phases may use different moves as the action
develops): {sh.get('motion', '')}
Scene: {scene.get('summary', '')[:150]}
{f"Dialogue lines that occur somewhere in this shot (assign each to the ONE phase where it is actually spoken; a line must appear in exactly one phase, never repeated):{chr(10)}{line_listing}" if lines else "No dialogue in this shot."}

Return ONLY raw JSON:
{{"phases": [
   {{"phase": 1,
     "action": "<concrete continuous behavior for JUST this part of the
                take, written as what happens next, never repeating an
                earlier phase>",
     "motion": "<one dominant camera move for this phase, <=40 words>",
     "line_indices": [<indices from the list above spoken during this
                      phase, or empty>]}}
   ... exactly {n} objects ...
 ]}}
"""
        data = safe_json_dict(get_llm(prompt, temperature=0.7))
        phases = data.get("phases") or []
        clean_phases = []
        seen_lines = set()
        for i in range(n):
            p = phases[i] if i < len(phases) else {}
            idxs = [int(x) for x in (p.get("line_indices") or [])
                    if isinstance(x, (int, float))
                    and 0 <= int(x) < len(lines) and int(x) not in seen_lines]
            seen_lines.update(idxs)
            clean_phases.append({
                "action": clean_text(str(p.get("action", "")))
                          or sh.get("action", ""),
                "motion": clean_text(str(p.get("motion", "")))
                          or sh.get("motion", ""),
                "line_indices": idxs,
            })
        # Any line the model failed to place goes to phase 0 rather than
        # silently vanishing.
        missing = [i for i in range(len(lines)) if i not in seen_lines]
        if missing and clean_phases:
            clean_phases[0]["line_indices"].extend(missing)
        sh["chain_phases"] = clean_phases
        logger.info("[cine]   %s: %d phase(s) authored.", sh["shot_id"], n)


# ---------------------------------------------------------------------------
# Color script (palette progression across the film, Pixar-style)
# ---------------------------------------------------------------------------
def generate_color_script(story: Dict, scenes: List[Dict],
                          style_bible: Dict) -> None:
    """Per-scene GRADE within the locked palette family: the palette doesn't
    change, but its balance does -- warmth drains toward the low point,
    saturation peaks with joy, the climax gets the film's most extreme
    grade, the resolution resolves it. Stamped as scene['grade'] and
    injected into every image prompt."""
    logger.info("[cine] Writing the color script (palette progression)...")
    digest = "\n".join(
        f"{s['scene_id']}. act{s.get('act')} [{s.get('function','')}] "
        f"T={s.get('tension',0):.2f} emotion={s.get('emotion','')} :: "
        f"{s.get('summary','')[:90]}" for s in scenes)
    prompt = f"""
You are the colorist writing the film's COLOR SCRIPT. The palette is locked:
"{_as_str(style_bible.get('palette'))}". Chart how its BALANCE progresses
scene by scene -- which of the locked colors leads, how warm/cool the grade
sits, where saturation swells or drains -- so the film's color tells the
emotional story underneath the images. Classic moves: warmth bleeds out as
the lie fails, the low point is the most desaturated/coldest frame of the
film, the climax gets the most extreme grade, the resolution earns back a
transformed warmth. Adjacent scenes shift gradually except at deliberate
act turns.

Scenes:
{digest}

Return ONLY a raw JSON array, one object per scene in order:
{{"scene_id": <copied>, "grade": "<10-20 words: which locked color leads,
  warmth, saturation, contrast for THIS scene>"}}
"""
    by_id = {}
    for it in safe_json_list(get_llm(prompt, temperature=0.6)):
        sid = it.get("scene_id")
        if isinstance(sid, int):
            by_id[sid] = clean_text(str(it.get("grade", "")))
    for s in scenes:
        s["grade"] = by_id.get(s["scene_id"], "")


# ---------------------------------------------------------------------------
# Cut design (scene-boundary transitions: contrast, match cuts, sound bridges)
# ---------------------------------------------------------------------------
def design_cuts(scenes: List[Dict], story: Dict, locations: Dict) -> None:
    """Author every SCENE BOUNDARY as an editorial choice instead of an
    accident. Three tools:
      hard_contrast -- the outgoing and incoming frames collide (scale,
                      light, palette accent) for energy;
      match_cut     -- the incoming frame visually echoes the outgoing one
                      (shape, motion, gesture) so meaning carries across;
      sound_bridge  -- a faint pre-echo of the NEXT scene's ambience enters
                      under the outgoing shot's soundscape (a J-cut built
                      inside H3's own audio).
    Notes land on the boundary shots as sh['transition_note'] (visual) and
    sh['sound_bridge'] (audio), consumed by the image-prompt and soundscape
    passes."""
    if len(scenes) < 2:
        return
    logger.info("[cine] Designing %d scene-boundary cuts...", len(scenes) - 1)
    pairs = []
    for i in range(len(scenes) - 1):
        a, b = scenes[i], scenes[i + 1]
        pairs.append({
            "boundary": i,
            "out_scene": {"id": a["scene_id"], "loc": a.get("location", ""),
                          "summary": a.get("summary", "")[:100],
                          "emotion": a.get("emotion", ""),
                          "tension": a.get("tension", 0)},
            "in_scene": {"id": b["scene_id"], "loc": b.get("location", ""),
                         "summary": b.get("summary", "")[:100],
                         "emotion": b.get("emotion", ""),
                         "tension": b.get("tension", 0),
                         "ambience": locations.get(b.get("location", ""),
                                                   {}).get("ambience", "")},
        })
    prompt = f"""
You are the editor designing every scene-to-scene cut in this film. For each
boundary choose ONE transition craft and give concrete notes:
- "hard_contrast": the frames collide -- name the axis (scale, light,
  palette accent, motion direction) the collision rides on.
- "match_cut": the incoming frame ECHOES the outgoing one -- name the shared
  shape, gesture, or motion (a closing hand becomes a closing door; a
  falling ember becomes a rising star). Use sparingly, at meaning-bearing
  turns; 2-4 in the whole film, never twice in a row.
- "sound_bridge": the next scene's ambience faintly pre-enters under the
  outgoing shot's final seconds -- describe the pre-echo in 6-12 words.
Rising tension favors contrast; thematic turns favor match cuts; gentle
passages favor sound bridges. Vary across the film.

Film: {story.get('logline','')}
Through-line: {story.get('through_line','')}

Boundaries:
{json.dumps(pairs, indent=1)}

Return ONLY a raw JSON array, one object per boundary in order:
{{"boundary": <copied int>, "type": "hard_contrast|match_cut|sound_bridge",
  "note_out": "<how the OUTGOING scene's final shot should end (visual), or
               '' if unchanged>",
  "note_in": "<how the INCOMING scene's first shot should open (visual), or
              '' if unchanged>",
  "sound_pre_echo": "<only for sound_bridge: the faint incoming sound>"}}
"""
    for it in safe_json_list(get_llm(prompt, temperature=0.75)):
        bi = it.get("boundary")
        if not isinstance(bi, int) or not (0 <= bi < len(scenes) - 1):
            continue
        a, b = scenes[bi], scenes[bi + 1]
        a_shots, b_shots = a.get("shots", []), b.get("shots", [])
        ttype = str(it.get("type", "")).strip()
        note_out = clean_text(str(it.get("note_out", "")))
        note_in = clean_text(str(it.get("note_in", "")))
        if a_shots and note_out:
            a_shots[-1]["transition_note"] = f"[{ttype} out] {note_out}"
        if b_shots and note_in:
            b_shots[0]["transition_note"] = f"[{ttype} in] {note_in}"
        if ttype == "sound_bridge" and a_shots:
            echo = clean_text(str(it.get("sound_pre_echo", "")))
            if echo:
                a_shots[-1]["sound_bridge"] = echo


# ---------------------------------------------------------------------------
# Dynamics map (the film's loudness/density arc; the hush before the climax)
# ---------------------------------------------------------------------------
def compute_dynamics(scenes: List[Dict]) -> None:
    """Deterministic sound-density arc from the tension curve, stamped as
    scene['sound_density'] and consumed by the soundscape pass. The scene
    BEFORE the tension peak is deliberately hushed (the held breath), the
    peak is the densest, and the final scene settles."""
    if not scenes:
        return
    peak = max(range(len(scenes)), key=lambda i: scenes[i].get("tension", 0))
    for i, s in enumerate(scenes):
        band = s.get("tension_band", "mid")
        if i == peak - 1 and peak > 0:
            d = ("HUSHED -- the held breath before the peak: strip the bed "
                 "to near-silence, one small close sound carrying the frame")
        elif i == peak:
            d = ("PEAK -- the film's densest, most foreground-forward sound; "
                 "every layer earned and specific")
        elif i == len(scenes) - 1:
            d = "SETTLING -- warm, receding, space opening up as the film lets go"
        elif band == "low":
            d = "sparse -- air and room, few sounds, each one placed"
        elif band == "high":
            d = "dense -- layered and driving, foreground events frequent"
        else:
            d = "moderate -- a living bed with occasional foreground events"
        s["sound_density"] = d


# ---------------------------------------------------------------------------
# Camera motion + soundscape (per shot, register-crafted)
# ---------------------------------------------------------------------------
def write_motion_and_sound(scenes: List[Dict], story: Dict,
                           locations: Dict, cfg: Dict,
                           graph=None) -> None:
    """Per scene: author each shot's camera/motion clause (one dominant move,
    tension-band grammar) and its overall_soundscape (register sound-design
    methods + the location's signature ambience + planted sound motifs).
    When the CharacterGraph is provided, the cast's SONIC SIGNATURES and
    musical idle behaviors are offered as leitmotif candidates -- a
    character's whistled tune becoming a film-wide sound motif (planted,
    transformed at the low point, whistled back by someone else at the
    resolution) is the highest-empathy use of the motif machinery."""
    registers = story.get("registers") or ["drama"]
    smethod = sound_method_block(registers)
    from .styles import camera_style_block
    camera_style_line = camera_style_block(cfg)

    # Character sounds as leitmotif candidates.
    char_sounds = ""
    if graph is not None:
        cand = []
        for n in graph.nodes.values():
            bits = []
            if getattr(n, "sonic_signature", ""):
                bits.append(n.sonic_signature)
            idle = getattr(n, "idle_behavior", "") or ""
            if any(w in idle.lower() for w in
                   ("whistl", "hum", "tap", "drum", "sing", "beat",
                    "rhythm", "tune", "melody")):
                bits.append(idle)
            if bits:
                cand.append(f"- {n.name}: {'; '.join(bits)}")
        if cand:
            char_sounds = (
                "CHARACTER SOUNDS (strong leitmotif candidates -- adopt at "
                "least ONE as a film motif if it fits: a character's "
                "whistled or tapped tune planted casually early, "
                "transformed by their emotional state at each return "
                "[slower in grief, broken off mid-phrase at the low "
                "point], and paid off when someone ELSE echoes it back "
                "near the resolution; a sonic signature can also announce "
                "its owner off-screen, and its ABSENCE can mark that "
                "something is wrong):\n" + "\n".join(cand))

    # Ask once for the film's recurring SOUND MOTIFS so the sound design is
    # composed across the film, not improvised per shot.
    logger.info("[cine] Designing the film's sound motifs...")
    prompt = f"""
Design 2-3 recurring diegetic SOUND MOTIFS for this film -- specific,
nameable sounds attached to a character, threat, object, or idea, planted
early and transformed at key turns (the sound-design equivalent of the
through-line object "{story.get('through_line','')}").

Film: {story.get('logline','')}
{smethod}

{char_sounds}

Return ONLY raw JSON:
{{"motifs": [{{"name": "<short name>",
             "sound": "<8-18 words describing the exact sound>",
             "attached_to": "<what it belongs to>",
             "arc": "<how it transforms across the film>"}}]}}
"""
    motifs = (safe_json_dict(get_llm(prompt, temperature=0.7)).get("motifs")
              or [])[:3]
    story["sound_motifs"] = motifs
    motif_block = "\n".join(
        f"- {m.get('name','')}: {m.get('sound','')} (attached to "
        f"{m.get('attached_to','')}; arc: {m.get('arc','')})" for m in motifs)

    for scene in scenes:
        shots = scene.get("shots", [])
        if not shots:
            continue
        loc = locations.get(scene.get("location", ""), {})
        from .storyboard import scene_storyboard as _sbm, recent_soundscapes
        film_map = _sbm(scenes, mark_scene_id=scene["scene_id"],
                        header="FULL FILM STORYBOARD (this scene marked >>; "
                               "camera language and sound design develop "
                               "ACROSS this arc, not per-scene):")
        prior_sound = recent_soundscapes(scenes, scene["scene_id"], n=4)
        prior_sound_block = (
            f"SOUND ALREADY DESIGNED (the most recent soundscapes -- develop "
            f"from them: continue arcs, transform motifs, never repeat a "
            f"soundscape verbatim):\n{prior_sound}" if prior_sound else "")
        specs = [{"shot_id": sh["shot_id"], "duration": sh["duration"],
                  "action": sh.get("action", "")[:200],
                  "band": sh.get("tension_band", "mid"),
                  "key_emotion": sh.get("key_emotion", ""),
                  "silence": bool(sh.get("silence")),
                  "has_dialogue": bool(sh.get("lines")),
                  **({"sound_bridge": sh["sound_bridge"]}
                     if sh.get("sound_bridge") else {})} for sh in shots]
        grammar = camera_grammar_block(scene.get("tension_band", "mid"),
                                       registers)
        density_line = (f"SCENE DYNAMICS (this scene's place in the film's "
                        f"loudness arc; every soundscape here obeys it): "
                        f"{scene.get('sound_density','')}"
                        if scene.get("sound_density") else "")
        if scene.get("voice_design") == "atmospheric":
            density_line += (
                "\nDESIGNED SILENCE: no one speaks in this scene -- the "
                "SOUNDSCAPE IS ITS VOICE. Compose it like a held piece of "
                "music: richer and more specific than a voiced scene's "
                "(the grain of individual waves, one gull far off, breath, "
                "cloth, heartbeat-adjacent low sound), with a clear "
                "emotional arc across the scene's shots that lands the "
                "feeling words would have carried.")
        prompt = f"""
For each shot of this scene author (a) its CAMERA/MOTION clause and (b) its
diegetic SOUNDSCAPE. Each shot is one continuous take animated from a still.

{film_map}

{prior_sound_block}

{grammar}

{camera_style_line}
MOTION RULES:
- 1-2 sentences, under 60 words. The subject ENACTS the shot's action as one
  coherent movement; exactly ONE dominant camera move written as a natural
  clause. During spoken lines the camera stays steady enough that the
  speaking face reads clearly.
- Vary the dominant move shot to shot; never the same move twice in a row.
- LONG TAKES: any shot over 30 seconds is one continuous unbroken take.
  Write its motion as 2-3 PHASES in strict temporal sequence ("The take
  opens as... midway, ... by the end, ..."), each phase with its own single
  dominant camera move, and keep the camera CONTINUOUSLY moving through the
  phase boundaries (a drifting arc or slow track, never a static hold at a
  transition) so the take reads as one unbroken breath. Up to 90 words.

SOUNDSCAPE RULES (this text becomes the shot's audio, verbatim):
{smethod}
{density_line}
- LOCATION AMBIENCE for this scene (the sonic bed under every shot here):
  {loc.get('ambience','a natural ambient bed true to the setting')}
- FILM SOUND MOTIFS (plant/transform where the arc says; never all at once):
{motif_block or '  (none)'}
- A shot carrying "sound_bridge" is the last shot before a scene change:
  in its final seconds, let that described sound faintly PRE-ENTER under
  the current ambience (distant, low, arriving) -- the next scene's world
  bleeding in early.
- 12-35 words: the ambient bed + 1-3 specific physical sounds tied to the
  visible action, with clear foreground/background layering. Concrete and
  recordable ("boots grinding frost, a rope creaking under load, wind
  thinning to silence"), never abstract ("tense atmosphere").
- Shots marked silence=true: design the QUIET -- what tiny sound remains.
- Never mention music (music is handled separately) and never describe
  voices or dialogue (spoken lines are provided elsewhere).

Scene: {scene.get('summary','')[:200]}  Tension: {scene.get('tension',0):.2f}

Shots:
{json.dumps(specs, indent=1)}

Return ONLY a raw JSON array, one object per shot in order:
{{"shot_id": "<copied>",
  "motion": "<the motion clause>",
  "soundscape": "<the diegetic sound design>"}}
"""
        items = safe_json_list(get_llm(prompt, temperature=0.8))
        by_id = {it.get("shot_id"): it for it in items if isinstance(it, dict)}
        prev_move = ""
        for sh in shots:
            it = by_id.get(sh["shot_id"], {})
            motion = clean_text(str(it.get("motion", "")))
            if not motion:
                act = (sh.get("action") or "moves through the moment")[:120]
                motion = (f"The subject {act} as the camera pushes in slowly "
                          f"with small amplitude.")
            sh["motion"] = motion
            sound = clean_text(str(it.get("soundscape", "")))
            if not sound:
                sound = loc.get("ambience") or \
                    "a quiet natural ambient bed true to the setting"
            sh["soundscape"] = sound
