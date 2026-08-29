"""
film_maker.screenplay
=====================
Beats → scenes → shots → dialogue.

  * PACING ENGINE — each beat becomes one scene; each scene is broken into
    shots whose durations follow the beat's TENSION through the register's
    pacing curve (comedy cuts fast and holds a beat after punchlines; horror
    holds long then snaps). Durations stay inside H3's renderable window;
    the renderer later snaps them to the 5+17k frame grid.

  * EI DIALOGUE — whole-scene rewrites using the CharacterGraph (wounds,
    longings, relationship edges, theory-of-mind), so exchanges carry
    subtext instead of turn-taking. Register dialogue methods (comic
    timing, dread under-writing, romantic undertow) shape every line.

  * DELIVERY + BUDGET — every line gets a delivery cue (whispers, snaps,
    deadpan...) that H3 renders vocally; spoken words are budgeted to the
    shot's duration (~2.2 words/sec ceiling) so lip-sync never rushes.

  * REPETITION GUARD — deterministic difflib pass (ported from the comic
    pipeline) that removes near-verbatim repeats within a sliding window
    while protecting deliberate long-distance callbacks.

Shot/Scene are plain dicts throughout so plan.json stays hand-editable.
"""
from __future__ import annotations

import difflib
import math
import json
import re
from typing import Dict, List, Optional

from .llm import get_llm, safe_json_list, safe_json_dict, clean_text, logger
from .registers import (dialogue_method_block, content_rating_block,
                        pacing_multiplier, post_peak_hold)
from .graph import CharacterGraph


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------
def _band(tension: float) -> str:
    if tension < 0.34:
        return "low"
    if tension < 0.67:
        return "mid"
    return "high"


def _target_shot_seconds(tension: float, cfg: Dict, registers: List[str]) -> float:
    band = _band(tension)
    base = {"low": float(cfg.get("cut_target_low_seconds", 18.0)),
            "mid": float(cfg.get("cut_target_mid_seconds", 11.0)),
            "high": float(cfg.get("cut_target_high_seconds", 6.5))}[band]
    t = base * pacing_multiplier(registers, band)
    return max(float(cfg.get("min_shot_seconds", 5.0)),
               min(float(cfg.get("max_shot_seconds", 30.0)), t))


def plan_scenes_and_shots(story: Dict, cfg: Dict) -> List[Dict]:
    """Deterministic skeleton: one scene per beat; shot count and durations
    from the tension/pacing engine; scene screen time scaled so the film
    totals target_minutes."""
    registers = story.get("registers") or ["drama"]
    beats = story.get("beats") or []
    total_target = float(cfg.get("target_minutes", 16.0)) * 60.0

    # Weight each beat's screen time by (0.7 + tension): peaks get more room.
    weights = [0.7 + float(b.get("tension", 0.5)) for b in beats]
    wsum = sum(weights) or 1.0
    scenes: List[Dict] = []
    for i, beat in enumerate(beats):
        scene_seconds = total_target * (weights[i] / wsum)
        t = float(beat.get("tension", 0.5))
        tgt = _target_shot_seconds(t, cfg, registers)
        n_shots = max(1, int(round(scene_seconds / tgt)))
        # distribute the scene's seconds across shots with mild variation so
        # cutting doesn't feel metronomic; keep every shot inside the window.
        lo = float(cfg.get("min_shot_seconds", 5.0))
        hi = float(cfg.get("max_shot_seconds", 30.0))
        durs = []
        for k in range(n_shots):
            wobble = 1.0 + 0.22 * math.sin((i * 3 + k) * 1.7)
            durs.append(max(lo, min(hi, tgt * wobble)))
        scale = scene_seconds / max(1e-6, sum(durs))
        durs = [max(lo, min(hi, d * scale)) for d in durs]
        scenes.append({
            "scene_id": i + 1,
            "beat_id": beat.get("beat_id", i + 1),
            "act": beat.get("act", 1),
            "function": beat.get("function", ""),
            "summary": beat.get("summary", ""),
            "location_hint": beat.get("location_hint", ""),
            "characters": list(beat.get("characters") or []),
            "tension": t,
            "tension_band": _band(t),
            "emotion": beat.get("emotion", ""),
            "shot_durations": [round(d, 2) for d in durs],
            "shots": [],
        })
    n_total = sum(len(s["shot_durations"]) for s in scenes)
    logger.info("[screenplay] %d scenes, %d shots, ~%.1f min planned.",
                len(scenes), n_total,
                sum(sum(s["shot_durations"]) for s in scenes) / 60.0)
    _promote_long_takes(scenes, cfg, registers)
    return scenes


def _promote_long_takes(scenes: List[Dict], cfg: Dict,
                        registers: List[str]) -> None:
    """Promote up to long_take_budget LOW-tension scenes into a single
    continuous long take (a oner), where the register's craft favors a held
    shot (drama/horror/romance/tragedy hold; comedy/adventure rarely should).
    A scene qualifies when its total planned time lands between ~32s and the
    long-take cap. Renderer handles durations past one H3 window via
    sliding-window stitching (or explicit chaining)."""
    if not cfg.get("enable_long_takes", True):
        return
    budget = int(cfg.get("long_take_budget", 3))
    cap = float(cfg.get("long_take_max_seconds", 60.0))
    if budget <= 0 or cap <= float(cfg.get("max_shot_seconds", 30.0)):
        return
    if pacing_multiplier(registers, "low") < 1.05:   # fast-cutting registers
        budget = min(budget, 1)
    candidates = []
    for s in scenes:
        total = sum(s["shot_durations"])
        if (s.get("tension_band") == "low" and len(s["shot_durations"]) >= 2
                and 32.0 <= total <= cap):
            candidates.append((total, s))
    # Spread picks across the film rather than clustering.
    candidates.sort(key=lambda ts: ts[1]["scene_id"])
    step = max(1, len(candidates) // max(1, budget))
    picked = [s for _, s in candidates[::step]][:budget]
    for s in picked:
        total = round(min(cap, sum(s["shot_durations"])), 2)
        s["shot_durations"] = [total]
        s["long_take"] = True
        logger.info("[screenplay] Scene %d promoted to a %ds continuous "
                    "long take.", s["scene_id"], int(total))


# ---------------------------------------------------------------------------
# Shot authoring (visual action per shot, then dialogue per scene)
# ---------------------------------------------------------------------------
def _dialogue_word_budget(duration: float, cfg: Dict) -> int:
    per10 = float(cfg.get("max_dialogue_words_per_10s", 22))
    return max(0, int(duration / 10.0 * per10))


def author_scene_shots(scene: Dict, prev_scene: Optional[Dict],
                       next_scene: Optional[Dict], story: Dict,
                       graph: CharacterGraph, cfg: Dict,
                       introduces: Optional[List[str]] = None,
                       film_storyboard: str = "",
                       written_digest: str = "") -> None:
    """One LLM call per scene: break the beat into its planned shots with
    concrete blocking, action, and dialogue written in the EI voices.
    Mutates scene["shots"] in place. `introduces` lists characters making
    their FIRST on-screen appearance here -- their entrance is designed as
    a character-defining image, a classic craft beat. `film_storyboard`
    (the whole film, this scene marked) and `written_digest` (everything
    authored so far, with key lines) give the writer the full map, so
    setups land, callbacks connect, and nothing is planned in a tunnel."""
    registers = story.get("registers") or ["drama"]
    durations = scene["shot_durations"]
    speakers = scene.get("characters") or []
    subtext = graph.scene_subtext_block(speakers) if speakers else ""
    chemistry = graph.chemistry_block(speakers) if len(speakers) > 1 else ""
    behavior = graph.behavior_block(speakers) if speakers else ""
    from .behavior_grammar import behavior_menu_block
    behavior_menu = behavior_menu_block(
        scene.get("emotion", ""),
        extra_emotions=[scene.get("summary", "")[:160]]) \
        if cfg.get("behavior_grammar", True) else ""
    dial_block = dialogue_method_block(registers)
    from .styles import dialogue_style_block
    dstyle_block = dialogue_style_block(cfg)
    rating = content_rating_block(cfg.get("content_rating", "teen"))
    hold = post_peak_hold(registers)
    intro_rule = ""
    if introduces:
        intro_rule = (
            f"- CHARACTER INTRODUCTION: {', '.join(introduces)} appear(s) on "
            f"screen for the FIRST time in this scene. Design each entrance "
            f"as a character-defining image: we meet them mid-behavior that "
            f"reveals who they are before they speak (what their hands are "
            f"doing, how they occupy the space, what they notice first). "
            f"Never introduce a principal standing idle.\n")
    if scene.get("voice_design") == "atmospheric":
        note = scene.get("voice_note") or \
            "the emotion is carried without words"
        voice_rule = (
            "- DESIGNED SILENCE: this scene is an ATMOSPHERIC beat ("
            + note + "). Write NO dialogue and NO narration -- every "
            "shot has an empty lines list and silence true. The "
            "emotion lives entirely in bodies, faces, and the world: "
            "give the ACTION field the expressive physical performance "
            "(the shoulders finally dropping, tears without sound, the "
            "run flat-out), and trust the soundscape -- waves, wind, "
            "breath, the roar of the moment -- to be the scene's "
            "voice. Nonverbal vocal sound (a sob, a gasped breath, "
            "laughter) belongs in the action and soundscape "
            "descriptions, never in dialogue lines.")
    else:
        voice_rule = (
            "- VOICE IS THE DEFAULT: every scene needs a voice -- "
            "dialogue where characters share the frame, NARRATOR "
            "interiority where a character is alone or the moment is "
            "internal. Total silence is reserved for scenes the voice "
            "plan designates as atmospheric; within this voiced scene, "
            "at most one fully silent shot.")
    connection_block = ""
    narrator = (story.get("narrator") or {}
                if str(cfg.get("narration", "auto")).lower() != "off" else {})
    narrator_block = ""
    if narrator:
        narrator_block = (
            f"- NARRATOR (this film HAS a designed narrator -- use them): "
            f"persona: {narrator.get('persona','')}. Voice: "
            f"{narrator.get('voice_texture','')}. Use speaker \"NARRATOR\" "
            f"for interiority (the quiet part said out loud -- what a "
            f"character thinks or feels but would never say), for scenes "
            f"where a character has no scene partner, and for orientation "
            f"at act turns. Narrator lines are off-screen voice, mouth "
            f"never shown speaking, and their delivery cue mirrors the "
            f"moment's emotional temperature: an intimate near-whisper for "
            f"private/sensual thoughts, hot and driving for anger, slow "
            f"and weighted for grief, wry lightness for comedy. Narration "
            f"counts against the shot's word budget.\n")
    if scene.get("connection_note"):
        connection_block = (
            f"AUDIENCE CONNECTION DIRECTION (mandatory -- realize this "
            f"inside the shots below as concrete staged behavior, faces, "
            f"and objects; these beats are why the audience will care):\n"
            f"{scene['connection_note']}\n")
    if scene.get("eureka_note"):
        connection_block += (
            f"EUREKA LADDER DIRECTION (mandatory -- this scene carries a "
            f"rung of the film's realization; stage it exactly, and let the "
            f"audience see the piece a beat before any character reacts to "
            f"it):\n{scene['eureka_note']}\n")

    prev_ctx = ""
    if prev_scene and prev_scene.get("shots"):
        last = prev_scene["shots"][-1]
        prev_ctx = (f"PREVIOUS SCENE ended on: {last.get('action','')[:180]} "
                    f"(emotion: {prev_scene.get('emotion','')}). Cut from it "
                    f"with intent: contrast or continuation, never accident.")
    next_ctx = ""
    if next_scene:
        next_ctx = (f"NEXT SCENE will be: {next_scene.get('summary','')[:150]} "
                    f"-- let this scene's final shot hand off to it.")
    storyboard_block = ""
    if film_storyboard:
        storyboard_block = (
            "FULL FILM STORYBOARD (your scene is marked with >>; know what "
            "came before and what is coming, so this scene sets up what it "
            "must and pays off what it can -- and never accidentally "
            "duplicates another scene's event or staging):\n"
            + film_storyboard)
    if written_digest:
        storyboard_block += (
            "\n\nALREADY WRITTEN (staged actions and key lines from every "
            "scene authored so far -- you may echo a line, call back a "
            "gesture, or deliberately contrast, but never repeat action or "
            "dialogue verbatim):\n" + written_digest)

    shot_specs = "\n".join(
        f"  shot {k+1}: ~{d:.0f}s (spoken-word budget: "
        f"{_dialogue_word_budget(d, cfg)} words max across ALL lines)"
        for k, d in enumerate(durations))

    hold_rule = ("- COMEDY TIMING: after any shot ending on a punchline or "
                 "visual gag, the NEXT shot opens on a silent reaction beat "
                 "before anything else happens. Punchlines land on the last "
                 "word of the line.\n") if hold else ""

    prompt = f"""
Direct this scene shot by shot. Every shot becomes ONE ~{durations[0]:.0f}-30s
continuously animated take from a single reference image, with a video model
that speaks the dialogue and performs the action -- so each shot must be one
coherent, stageable piece of continuous action in one framing.

FILM: {story.get('title','')} -- {story.get('logline','')}
ERA: {story.get('era','')}   WORLD: {story.get('world','')[:220]}
THROUGH-LINE OBJECT: {story.get('through_line','')}
SCENE {scene['scene_id']} (act {scene['act']}, {scene.get('function','')}),
tension {scene['tension']:.2f} ({scene['tension_band']}):
{scene['summary']}
Location: {scene.get('location_hint','')}
Emotion to leave in the audience: {scene.get('emotion','')}
{prev_ctx}
{next_ctx}
{storyboard_block}
{connection_block}
CHARACTERS IN SCENE and their psychology (write dialogue FROM these; the
relationship edges and theory-of-mind beliefs are where subtext lives --
what a character wrongly believes the other feels should bend how they
speak; each character's emotion styles, backstory, relic, and tell are
performance tools -- when the present rhymes with a backstory, the tell
fires BEFORE any line acknowledges it):
{subtext or '(no speaking characters -- a visual scene)'}

{chemistry}

{behavior}

{behavior_menu}

{dial_block}
{dstyle_block}
{rating}
{hold_rule}
SHOT PLAN (honor each shot's duration and word budget exactly):
{shot_specs}

RULES:
- NARRATION STAGING: for any shot where ONLY the NARRATOR speaks, stage
  the visuals AWAY from talking-ready faces -- favor hands, objects,
  environment, silhouettes, backs, wide B-roll of the setting -- or, if
  a face is framed, give the mouth a visible reason to be closed and
  busy (drinking, jaw set, biting a pen, wind-squint). Never put a
  character in a face-forward close-up during pure narration; the
  narrator's voice floats over a world that is not talking.
- SCENE LANDING (critical): the LAST shot of the scene must NOT end on
  its final spoken line -- author a wordless AFTERMATH into that shot's
  action: 2-4 seconds of what happens AFTER the words that dramatizes
  their impact (a hand finishing or abandoning its gesture; a cut of
  attention to the object whose meaning just changed; the environment
  answering -- rain hardening, a counter still climbing, a light going
  steady). The words land, then the image proves them.
- ACTION: one concrete continuous piece of staging per shot -- movement,
  business with objects, physical behavior that dramatizes the beat. Never
  "stands and talks" unless stillness IS the drama; give hands and bodies
  something true to do.
- BEHAVIORAL IDIOM: when a character's hands are free, reach for THEIR
  signature idle behavior (from the psychology above) before any generic
  gesture -- and make it react to the moment: it quickens under stress,
  softens in comfort, stops dead at a shock. A passion should leak into
  physical life (the music-lover taps a beat on the can he's holding,
  whistles a phrase while working -- and the tune he whistles is his,
  reusable across the film). Draw other physical behavior from the
  BEHAVIOR PALETTE matched to each character's emotion style; never
  default to clenched jaws, gripped railings, or a single tear.
{intro_rule}- DIALOGUE: short lines (a spoken line over ~18 words strains lip-sync).
  Respect each shot's total word budget. Characters may speak in at most a
  natural back-and-forth (1-4 lines per shot). Write all numbers, dates,
  and abbreviations as SPOKEN WORDS ("Doctor", "nineteen forty", "three"),
  never digits or shorthand -- these lines are literally voiced.
{voice_rule}
- EMOTIONAL DELIVERY CUES (mandatory on EVERY line, dialogue and
  narration): the "delivery" field must name the emotional color AND the
  physical vocal quality -- "through clenched teeth, barely controlled",
  "an intimate murmur, almost to herself", "bright and too fast, papering
  over panic", "hushed, reverent", "hot, rising". Never leave delivery
  empty and never use flat cues like "normally" or "says". The voice must
  carry the emotion even with eyes closed.
- SPOKEN VOICE (critical): write every line exactly as this character would
  SAY it out loud -- their dialect, idiom, contractions, and signature
  speech markers from the psychology above ("y'all", "mate", "mija"), with
  light eye-dialect only, never dense phonetic spelling. A stranger reading
  the lines aloud should be able to tell every character apart with eyes
  closed. Code-switch where the character's background supports it: brief
  embedded words in another language ("mija", "ay, Dios") stay inside a
  normal line; a line spoken FULLY in another language sets that line's
  "language" field (e.g. "Spanish") so the voice speaks it natively.
- SOCIAL REFLEXES: characters react like humans in the room -- a compliment
  from someone they're drawn to earns a visible blush/deflection before any
  reply; praise from someone respected straightens the spine; teasing
  between friends reads as affection; embarrassment shows on the body
  before it's spoken. Honor the CHEMISTRY direction above in both the
  action staging and the lines.
- DELIVERY: every line gets a delivery cue the voice can act: whispers /
  snaps / deadpan / through tears / bubbling with excitement / cold and
  measured / slurring / breathless...
{narrator_block}- SUBTEXT: at least one line per dialogue scene should mean more than it
  says. People deflect, test, and answer the question they wish was asked.
- CAMERA-FACING: characters never address the camera unless the register is
  comedy and it is a deliberate device used sparingly.

Return ONLY a raw JSON array, one object per shot, in order:
{{"shot": <1-based int>,
  "action": "<2-3 sentences of concrete visible staging for this take>",
  "framing_hint": "<what the frame should favor: whose face, what object,
                   what space>",
  "lines": [{{"speaker": "<exact character name or NARRATOR>",
             "text": "<the spoken words only>",
             "delivery": "<vocal delivery cue>",
             "language": "<only when the WHOLE line is in a non-default
                          language, e.g. 'Spanish'; omit otherwise>"}}],
  "silence": true|false,
  "key_emotion": "<the shot's emotional note>"}}
"""
    items = safe_json_list(get_llm(prompt, temperature=0.85, large=True))
    shots = []
    for k, d in enumerate(durations):
        item = items[k] if k < len(items) else {}
        lines = []
        for ln in (item.get("lines") or []):
            if not isinstance(ln, dict):
                continue
            text = clean_text(str(ln.get("text", "")).strip().strip('"'))
            if not text:
                continue
            entry = {"speaker": clean_text(str(ln.get("speaker", ""))),
                     "text": text,
                     "delivery": clean_text(str(ln.get("delivery", "")))}
            lang = clean_text(str(ln.get("language", "") or "")).strip()
            if lang:
                entry["language"] = lang
            lines.append(entry)
        budget = _dialogue_word_budget(d, cfg)
        lines = _trim_lines_to_budget(lines, budget)
        shots.append({
            "shot_id": f"{scene['scene_id']:02d}_{k+1:02d}",
            "scene_id": scene["scene_id"],
            "index_in_scene": k + 1,
            "duration": round(d, 2),
            "action": clean_text(str(item.get("action", "")) or
                                 scene["summary"][:200]),
            "framing_hint": clean_text(str(item.get("framing_hint", ""))),
            "lines": lines,
            "silence": bool(item.get("silence", not lines)),
            "key_emotion": clean_text(str(item.get("key_emotion", "")) or
                                      scene.get("emotion", "")),
            "tension": scene["tension"],
            "tension_band": scene["tension_band"],
        })
    scene["shots"] = shots


def _trim_lines_to_budget(lines: List[Dict], budget: int) -> List[Dict]:
    if not lines:
        return lines
    if budget <= 0:
        return []
    out, used = [], 0
    for ln in lines:
        w = len(ln["text"].split())
        if used + w > budget:
            room = budget - used
            if room >= 4 and not out:
                ln = dict(ln)
                ln["text"] = " ".join(ln["text"].split()[:room])
                out.append(ln)
            break
        out.append(ln)
        used += w
    return out


# ---------------------------------------------------------------------------
# Repetition guard (ported, deterministic)
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def repetition_guard(scenes: List[Dict], cfg: Dict) -> int:
    """Remove near-verbatim repeats within a sliding window of shots.
    Distance gate protects deliberate long-range callbacks."""
    window = int(cfg.get("dialogue_repetition_window", 6))
    threshold = float(cfg.get("dialogue_repetition_threshold", 0.72))
    flat: List[Dict] = []
    for sc in scenes:
        for sh in sc.get("shots", []):
            for ln in sh.get("lines", []):
                flat.append({"shot": sh, "line": ln})
    removed = 0
    for i, cur in enumerate(flat):
        a = _norm(cur["line"]["text"])
        if not a or len(a.split()) < 3:
            continue
        for j in range(max(0, i - window), i):
            b = _norm(flat[j]["line"]["text"])
            if not b:
                continue
            if difflib.SequenceMatcher(None, a, b).ratio() >= threshold:
                cur["shot"]["lines"] = [l for l in cur["shot"]["lines"]
                                        if l is not cur["line"]]
                if not cur["shot"]["lines"]:
                    cur["shot"]["silence"] = True
                cur["line"]["text"] = ""
                removed += 1
                break
    if removed:
        logger.info("[screenplay] Repetition guard removed %d duplicate "
                    "line(s).", removed)
    return removed


# ---------------------------------------------------------------------------
# Scene-level dialogue polish (EI second pass over weak exchanges)
# ---------------------------------------------------------------------------
def polish_dialogue(scenes: List[Dict], story: Dict, graph: CharacterGraph,
                    cfg: Dict) -> None:
    """A second EI pass over the scenes with the most dialogue: sharpen
    subtext, distinct voices, and register timing without changing staging.
    Cheap: only the top-N talkiest scenes get the pass."""
    registers = story.get("registers") or ["drama"]
    from .styles import dialogue_style_block as _dsb
    _dstyle = _dsb(cfg)
    talky = sorted(
        [s for s in scenes if sum(len(sh.get("lines", []))
                                  for sh in s.get("shots", [])) >= 3],
        key=lambda s: -sum(len(sh.get("lines", [])) for sh in s.get("shots", []))
    )[:8]
    if not talky:
        return
    logger.info("[screenplay] Polishing dialogue in %d scene(s)...", len(talky))
    for sc in talky:
        speakers = sorted({ln["speaker"] for sh in sc["shots"]
                           for ln in sh.get("lines", []) if ln.get("speaker")})
        addressable = []
        for sh in sc["shots"]:
            for li, ln in enumerate(sh.get("lines", [])):
                addressable.append((sh, li, ln))
        listing = "\n".join(
            f"[{k}] shot {sh['shot_id']} {ln['speaker']} "
            f"({ln.get('delivery','')}): {ln['text']}"
            for k, (sh, li, ln) in enumerate(addressable))
        prompt = f"""
Punch up this scene's dialogue. Keep meaning, staging, speakers, and line
COUNT identical; improve only the words and delivery cues. Make each voice
unmistakable, load subtext from the psychology below, and honor the timing
craft of the registers. Keep every line at or under its current word count.
EVERY line (dialogue and NARRATOR alike) must carry an emotional delivery
cue naming the emotional color AND vocal quality ("an intimate murmur,
almost to herself", "hot, through clenched teeth", "slow, weighted with
grief") -- rewrite any flat, empty, or purely mechanical cue.
Every line must sound like THIS character speaking aloud -- their dialect,
idiom, and signature markers, with light eye-dialect only. Where the
chemistry direction applies, let attraction color the wording: teasing with
precision, compliments deflected a beat too fast, warmth fought down.

{dialogue_method_block(registers)}
{_dstyle}

For any character with a COMIC TOOLKIT below: their timing lives in the
delivery cue -- write it as playable direction ("flat, then hitting the
word 'fine' way too hard", "a beat of silence first, then deadpan"); put
word emphasis in the CUE, never as capital letters in the spoken text;
punchlines land on the last word; sarcasm aims along the theory-of-mind
edges (sharpest with whoever they wrongly believe disrespects them).

Scene: {sc.get('summary','')[:250]}
Psychology:
{graph.scene_subtext_block(speakers)}

{graph.chemistry_block(speakers)}

{graph.behavior_block(speakers)}

Lines (revise only the ones you can genuinely improve):
{listing}

Return ONLY a raw JSON array of revisions:
{{"k": <int index from the listing>, "text": "<better line>",
  "delivery": "<delivery cue>"}}
"""
        for rev in safe_json_list(get_llm(prompt, temperature=0.85)):
            k = rev.get("k")
            if not isinstance(k, int) or not (0 <= k < len(addressable)):
                continue
            sh, li, ln = addressable[k]
            new = clean_text(str(rev.get("text", "")).strip().strip('"'))
            if new and len(new.split()) <= max(4, int(len(ln["text"].split()) * 1.3)):
                ln["text"] = new
                if rev.get("delivery"):
                    ln["delivery"] = clean_text(str(rev["delivery"]))


def design_voice_plan(scenes: List[Dict], story: Dict, cfg: Dict) -> None:
    """Decide, deliberately, which scenes are VOICED and which are DESIGNED
    SILENCE -- atmospheric beats where image and soundscape carry the
    emotion and any voice would cheapen it: the relief of waves after a
    storm of a scene, a wordless action surge, a held closeup of a face in
    grief. Silence chosen on purpose is craft; silence by accident is a
    gap. Stamps scene['voice_design'] in {'voiced','atmospheric'} plus
    scene['voice_note'] with the reasoning, consumed by scene authoring,
    the narration weave, the soundscape pass, and the coverage audit."""
    if not scenes:
        return
    logger.info("[screenplay] Designing the film's voice plan...")
    budget = max(1, round(len(scenes) / 5.0))
    digest = "\n".join(
        f"{s['scene_id']}. act{s.get('act')} [{s.get('function','')}] "
        f"T={s.get('tension',0):.2f} emotion={s.get('emotion','')} :: "
        f"{s.get('summary','')[:100]}" for s in scenes)
    prompt = f"""
Design this film's VOICE PLAN: for each scene decide whether it is VOICED
(dialogue and/or narration carry it) or ATMOSPHERIC (designed silence --
no dialogue, no narration; the image, the actors' bodies and faces, and a
rich soundscape carry the emotion entirely).

ATMOSPHERIC scenes are powerful precisely because they're rationed: choose
about {budget} of them (roughly one per five scenes), placed where the
emotional journey needs BREATH -- the exhale after a peak (relief on open
water, wind through grass), a wordless surge of pure action, a held
closeup of grief or awe where any word would be smaller than the face.
Never make the first or last scene atmospheric-only (the film's opening
orientation and closing bookend narration play there). A scene whose beat
REQUIRES an exchange or a revelation spoken aloud must stay voiced.

Scenes:
{digest}

Return ONLY a raw JSON array, one object per scene in order:
{{"scene_id": <copied>, "voice_design": "voiced"|"atmospheric",
  "voice_note": "<one short clause of reasoning, e.g. 'the exhale after
   the confrontation; waves and breath carry it'>"}}
"""
    by_id = {}
    for it in safe_json_list(get_llm(prompt, temperature=0.6)):
        if isinstance(it, dict) and isinstance(it.get("scene_id"), int):
            by_id[it["scene_id"]] = it
    n_atmo = 0
    for i, s in enumerate(scenes):
        it = by_id.get(s["scene_id"], {})
        design = str(it.get("voice_design", "voiced")).strip().lower()
        # First and last scenes always stay voiced (opening/closing
        # narration mandates play there).
        if i in (0, len(scenes) - 1):
            design = "voiced"
        s["voice_design"] = ("atmospheric" if design == "atmospheric"
                             else "voiced")
        s["voice_note"] = clean_text(str(it.get("voice_note", "")))
        if s["voice_design"] == "atmospheric":
            n_atmo += 1
            logger.info("[screenplay]   Scene %d: DESIGNED SILENCE -- %s",
                        s["scene_id"], s["voice_note"][:80])
    logger.info("[screenplay] Voice plan: %d voiced, %d atmospheric.",
                len(scenes) - n_atmo, n_atmo)


def design_narrator(story, cfg):
    """Design the film's narrator PERSONA before any scene is written: who
    the voice is (first-person retrospective protagonist, omniscient
    storyteller, wry observer...), their voice texture, and their
    relationship to the story. Honors cfg['narration_style'] free text.
    Stamps story['narrator']; no-op when narration is 'off'."""
    if str(cfg.get("narration", "auto")).lower() == "off":
        story.pop("narrator", None)
        return
    style = (cfg.get("narration_style") or "").strip()
    logger.info("[screenplay] Designing the narrator persona...")
    prompt = f"""
Design the NARRATOR for this film -- the off-screen voice that opens it,
speaks its interiority, and closes it. The strongest narrators have a
specific relationship to the story (a character remembering it years
later, an omniscient teller with a point of view, a wry witness) and a
voice texture as distinct as any character's.

Film: {story.get('title','')} -- {story.get('logline','')}
Registers: {', '.join(story.get('registers', []))}
Themes: {', '.join(story.get('themes', []))}
Protagonist: {(story.get('characters') or [{}])[0].get('name','')}
{f'AUTHOR-LOCKED NARRATION STYLE (honor exactly): {style}' if style else ''}

Return ONLY raw JSON:
{{"persona": "<who the narrator is and their relationship to the story, 1-2 sentences>",
  "voice_texture": "<how the voice sounds: age, grain, tempo, warmth -- the description a voice director would give, <=25 words>",
  "diction": "<their word-choice character: plain/lyrical/wry/formal..., <=15 words>",
  "knows_the_ending": true|false}}
"""
    data = safe_json_dict(get_llm(prompt, temperature=0.7))
    story["narrator"] = {
        "persona": clean_text(str(data.get("persona", "a measured, unseen storyteller"))),
        "voice_texture": clean_text(str(data.get("voice_texture", ""))),
        "diction": clean_text(str(data.get("diction", ""))),
        "knows_the_ending": bool(data.get("knows_the_ending", True)),
    }
    logger.info("[screenplay] Narrator: %s", story["narrator"]["persona"][:90])


def _shot_words_used(sh):
    return sum(len((ln.get("text") or "").split())
               for ln in sh.get("lines", []))


def weave_narration(scenes, story, graph, cfg):
    """Post-pass after scene authoring: walk the film and weave NARRATOR
    lines where they earn their place, within each shot's remaining word
    budget. Guarantees: (1) the film OPENS with orientation narration --
    who/where/what's at stake -- across scene 1's first shots; (2)
    interiority beats say the quiet part out loud at key emotional moments,
    in a delivery matched to the emotion (intimate near-whisper for private
    thoughts, hot for anger, slow for grief); (3) no scene is voiceless
    unless its silence is the point; (4) the film CLOSES with a bookend
    reflection. Narration is added only into budget slack, so lip-synced
    dialogue timing is never crowded."""
    narrator = story.get("narrator")
    if not narrator or str(cfg.get("narration", "auto")).lower() == "off":
        return
    rich = str(cfg.get("narration", "auto")).lower() == "rich"
    logger.info("[screenplay] Weaving narration (%s)...",
                "rich" if rich else "auto")
    recent = []
    protagonist = (story.get("characters") or [{}])[0].get("name", "")
    ration = ("Weave generously: most shots can carry a line." if rich
              else "Ration it: 1-3 beats in this scene at most, at the "
                   "moments that need a voice.")
    for s_i, scene in enumerate(scenes):
        shots = scene.get("shots", [])
        if not shots:
            continue
        if scene.get("voice_design") == "atmospheric":
            logger.info("[screenplay]   Scene %d: designed silence -- no "
                        "narration woven (%s).", scene.get("scene_id"),
                        (scene.get("voice_note") or "")[:60])
            continue
        slack_specs = []
        for sh in shots:
            budget = _dialogue_word_budget(float(sh.get("duration", 5)), cfg)
            remaining = max(0, budget - _shot_words_used(sh))
            slack_specs.append({
                "shot_id": sh["shot_id"],
                "action": (sh.get("action") or "")[:160],
                "key_emotion": sh.get("key_emotion", ""),
                "existing_lines": [f'{ln.get("speaker","")}: "{ln.get("text","")}"'
                                   for ln in sh.get("lines", [])],
                "narration_words_available": remaining,
            })
        scene_voiced = any(sh.get("lines") for sh in shots)
        is_opening = (s_i == 0)
        is_closing = (s_i == len(scenes) - 1)
        mandate = ""
        if is_opening:
            mandate = ("MANDATORY: this is the film's OPENING. Its first "
                       "shot(s) MUST carry orientation narration -- who "
                       "this is, where we are, what is at stake -- in the "
                       "narrator's own voice, planting the film's central "
                       "question. ")
        if is_closing:
            mandate += ("MANDATORY: this is the film's CLOSE. Its final "
                        "shot MUST carry a bookend reflection that answers "
                        "or transforms the opening narration. ")
        if not scene_voiced and not mandate:
            mandate = ("This scene currently has NO voice at all: unless "
                       "pure silence is unmistakably this scene's meaning, "
                       "give it at least one narration beat. ")
        node = graph.get_node(protagonist) if protagonist else None
        inner = ""
        if node:
            inner = (f"{protagonist}'s inner landscape (the material for "
                     f"interiority): wound={node.core_wound[:80]}; "
                     f"longing={node.core_longing[:80]}")
        recent_block = "\n".join("  " + r for r in recent[-4:]) or "  (none yet)"
        prompt = f"""
You are weaving the narrator's voice through one scene of this film.

NARRATOR: {narrator.get('persona','')}
Voice: {narrator.get('voice_texture','')} Diction: {narrator.get('diction','')}
{inner}

{mandate}THE CRAFT:
- Interiority is the narrator's best tool: say the QUIET PART out loud --
  what a character is thinking, feeling, or refusing to admit, which the
  frame alone cannot show. Never describe what we can already see.
- DELIVERY MATCHES THE EMOTION: private or sensual thoughts arrive as an
  intimate near-whisper, close to the ear; anger runs hot and driving;
  grief slow and weighted; wonder hushed; comedy wry and light. Name the
  emotional color AND vocal quality in every delivery cue.
- Narration never talks over dialogue's meaning -- it frames, deepens, or
  contrasts it. {ration}
- STRICT BUDGET: each addition must fit its shot's
  narration_words_available. A shot with 0 available gets nothing.
- Avoid echoing recent narration phrasing:
{recent_block}

Scene {scene.get('scene_id')} (emotion: {scene.get('emotion','')},
tension {scene.get('tension',0):.2f}): {scene.get('summary','')[:200]}

Shots:
{json.dumps(slack_specs, indent=1)}

Return ONLY a raw JSON array (empty if this scene truly needs none):
[{{"shot_id": "<copied>",
   "position": "before"|"after",
   "text": "<the narration, spoken words only, within budget>",
   "delivery": "<emotional color + vocal quality>"}}]
"""
        additions = safe_json_list(get_llm(prompt, temperature=0.75))
        by_shot = {sh["shot_id"]: sh for sh in shots}
        n_added = 0
        for add in additions:
            if not isinstance(add, dict):
                continue
            sh = by_shot.get(add.get("shot_id"))
            text = clean_text(str(add.get("text", "")))
            if sh is None or not text:
                continue
            budget = _dialogue_word_budget(float(sh.get("duration", 5)), cfg)
            remaining = max(0, budget - _shot_words_used(sh))
            words = text.split()
            if remaining < 4:
                continue
            if len(words) > remaining:
                text = " ".join(words[:remaining]).rstrip(",;: ") + "."
            line = {"speaker": "NARRATOR", "text": text,
                    "delivery": clean_text(str(add.get("delivery", "")))
                    or "quiet and close"}
            sh.setdefault("lines", [])
            if str(add.get("position", "before")).lower() == "after":
                sh["lines"].append(line)
            else:
                sh["lines"].insert(0, line)
            sh["silence"] = False
            recent.append(text[:60])
            n_added += 1
        if n_added:
            logger.info("[screenplay]   Scene %d: %d narration beat(s) "
                        "woven.", scene.get("scene_id", s_i + 1), n_added)


def validate_voice_coverage(scenes, cfg):
    """Deterministic audit logged after the screenplay is written: voice
    coverage per scene, voiceless scenes, lines missing emotional delivery
    cues, and whether the film opens/closes with narration."""
    total_shots = voiced_shots = 0
    voiceless_scenes = []
    designed_silent = []
    flat_cues = 0
    weak = {"", "says", "normally", "plainly", "speaks"}
    for sc in scenes:
        shots = sc.get("shots", [])
        any_voice = False
        for sh in shots:
            total_shots += 1
            if sh.get("lines"):
                voiced_shots += 1
                any_voice = True
                for ln in sh["lines"]:
                    if (ln.get("delivery") or "").strip().lower() in weak:
                        flat_cues += 1
        if shots and not any_voice:
            if sc.get("voice_design") == "atmospheric":
                designed_silent.append(sc.get("scene_id"))
            else:
                voiceless_scenes.append(sc.get("scene_id"))
    first = (scenes[0].get("shots") or [{}])[0] if scenes else {}
    last = (scenes[-1].get("shots") or [{}])[-1] if scenes else {}
    opens_narrated = any((ln.get("speaker") or "").upper() == "NARRATOR"
                         for ln in first.get("lines", []))
    closes_narrated = any((ln.get("speaker") or "").upper() == "NARRATOR"
                          for ln in last.get("lines", []))
    stats = {"total_shots": total_shots, "voiced_shots": voiced_shots,
             "voiceless_scenes": voiceless_scenes,
             "designed_silent_scenes": designed_silent,
             "lines_missing_emotional_delivery": flat_cues,
             "opens_with_narration": opens_narrated,
             "closes_with_narration": closes_narrated}
    logger.info("[screenplay] Voice coverage: %d/%d shots voiced (%.0f%%). "
                "Designed-silent (atmospheric) scenes: %s. UNINTENDED "
                "voiceless scenes: %s. Flat/missing delivery cues: %d. "
                "Opens narrated: %s | Closes narrated: %s.",
                voiced_shots, max(1, total_shots),
                100.0 * voiced_shots / max(1, total_shots),
                designed_silent or "none",
                voiceless_scenes or "none", flat_cues,
                opens_narrated, closes_narrated)
    if voiceless_scenes:
        logger.warning("[screenplay] %d scene(s) are voiceless WITHOUT a "
                       "designed-silence designation: %s -- either mark "
                       "them atmospheric in plan.json (voice_design) or "
                       "give them a voice.",
                       len(voiceless_scenes), voiceless_scenes)
    return stats


def write_screenplay(story: Dict, graph: CharacterGraph, cfg: Dict) -> List[Dict]:
    from .connection import design_connection_plan, design_eureka
    design_narrator(story, cfg)
    scenes = plan_scenes_and_shots(story, cfg)
    # Empathy architecture runs on the skeleton so scene authoring can
    # realize the beats as staged behavior.
    try:
        design_connection_plan(story, scenes, graph, cfg)
        design_eureka(story, scenes, cfg)
    except Exception as e:
        logger.warning("[screenplay] Connection design failed (%s); "
                       "continuing without it.", e)
    design_voice_plan(scenes, story, cfg)
    seen_chars = set()
    principals = {c["name"] for c in (story.get("characters") or [])
                  if c.get("role", "").lower() in
                  ("protagonist", "antagonist", "ally", "mentor")} or \
        {c["name"] for c in (story.get("characters") or [])}
    from .storyboard import scene_storyboard, authored_digest
    for i, sc in enumerate(scenes):
        introduces = [n for n in (sc.get("characters") or [])
                      if n in principals and n not in seen_chars]
        seen_chars.update(sc.get("characters") or [])
        logger.info("[screenplay] Scene %d/%d: %s",
                    i + 1, len(scenes), sc["summary"][:70])
        author_scene_shots(
            sc, scenes[i - 1] if i else None,
            scenes[i + 1] if i + 1 < len(scenes) else None,
            story, graph, cfg, introduces=introduces,
            film_storyboard=scene_storyboard(
                scenes, mark_scene_id=sc["scene_id"]),
            written_digest=authored_digest(
                scenes, upto_scene_id=sc["scene_id"]))
    weave_narration(scenes, story, graph, cfg)
    polish_dialogue(scenes, story, graph, cfg)
    repetition_guard(scenes, cfg)
    if cfg.get("behavior_grammar", True):
        from .behavior_grammar import behavior_repetition_report
        report = behavior_repetition_report(scenes, cfg)
        story["behavior_report"] = report
        if not report:
            logger.info("[behavior] Gesture audit clean: staged actions "
                        "stay varied across the film.")
    if cfg.get("hook_design", True):
        design_hooks(scenes, story, cfg)
    validate_voice_coverage(scenes, cfg)
    return scenes


def design_hooks(scenes, story, cfg):
    """Film-wide HOOK ARCHITECTURE (one LLM call): every scene gets an
    ENTRY HOOK (arrive late -- open mid-motion on something already
    wrong, wanted, or in progress), an EXIT HOOK (the concrete final
    beat: a held glance, an interrupted action, a sound sting, a small
    reveal, a question landing) and the OPEN QUESTION it leaves hanging.
    Hooks must chain: each exit's question is picked up within 1-2
    scenes, stakes escalate act by act, act-break scenes end on the
    hardest cliffs, and the finale PAYS OFF the planted questions rather
    than opening new ones. Render consumes these (entry hook shapes the
    scene's first window; exit hook replaces the stock emotion button)."""
    sc_lines = "\n".join(
        f'{s["scene_id"]} (act {s.get("act","?")}, {s.get("emotion","")}): '
        f'{str(s.get("summary",""))[:140]}' for s in scenes)
    prompt = f"""Design the hook architecture for this film -- the chain of
open questions that keeps a viewer unable to look away.
Film: {story.get('logline','')}
Scenes:
{sc_lines}

For EVERY scene return: "entry" -- how the scene opens ALREADY IN MOTION
(<=18 words, concrete and stageable, no throat-clearing); "exit" -- the
final held beat (<=20 words: a glance, gesture, sound, interruption, or
micro-reveal that lands a specific emotion); "question" -- what the
viewer is now dying to know (<=12 words). Chain them: questions planted
early deepen mid-film and PAY OFF at the end; act breaks get the
sharpest cliffs. Return ONLY raw JSON:
{{"hooks": [{{"scene_id": 1, "entry": "...", "exit": "...",
"question": "..."}}]}}"""
    from .llm import safe_json_dict
    data = safe_json_dict(get_llm(prompt, temperature=0.7, large=True))
    got = 0
    for h in (data or {}).get("hooks", []):
        for s in scenes:
            if s["scene_id"] == h.get("scene_id"):
                s["entry_hook"] = clean_text(h.get("entry", ""))
                s["exit_hook"] = clean_text(h.get("exit", ""))
                s["open_question"] = clean_text(h.get("question", ""))
                got += 1
    logger.info("[screenplay] Hook architecture: %d/%d scenes hooked.",
                got, len(scenes))
