"""
film_maker.connection
=====================
The empathy layer: the craft of making an audience CARE, then rewarding the
care. Runs on the scene skeleton (after pacing, before shots are authored),
stamping per-scene direction that the scene author must realize.

Three passes:

  CARE HOOKS (design_connection_plan)
    The proven levers by which audiences bond to a character, assigned to
    specific early scenes so the bond is EARNED on screen, not assumed:
      - undeserved misfortune  (something unfair happens to them)
      - kindness unobserved    (they do something good when no one's watching
                               -- the "save the cat" beat)
      - competence             (they're genuinely good at something, shown)
      - vulnerability admitted (a crack in the armor the audience sees first)
      - humor                  (they make us laugh, or laugh at themselves)
      - longing shown not told (we watch them want something small and human)
    Plus BACKSTORY ECHOES: each character's grounding memory (the universal
    experience in their graph node) surfaces obliquely across the film --
    the relic glimpsed, a half-sentence that stops, the tell firing when
    something rhymes with the memory -- and pays off ONCE in full near the
    character's hardest choice. Never a flashback dump; the audience
    assembles the memory from fragments, which is what makes it theirs.
    Plus REACTION BEATS: empathy is transmitted through faces reacting.
    After each high-tension event, the plan names WHOSE face we hold on.
    Plus MIRROR MOMENTS: one or two beats staged so the audience recognizes
    their own life in the frame (waiting rooms, school corridors, the walk
    to a front door with bad news).

  EUREKA LADDER (design_eureka)
    Realizations land hardest when the audience solves them ONE BEAT before
    the character. The ladder: 3 clues planted in separate scenes (each
    innocent in context, each visible), a false floor (the near-miss where
    character and audience almost see it), the CLICK (a visual reframe --
    two planted images collide into new meaning; no explaining dialogue),
    then RELEASE (the emotion of understanding: joy, grief, awe -- expressed
    in that character's own emotion style). Works identically for a story's
    truth-arc realization and an educational film's concept click.

All output is plain strings on scenes (connection_note, eureka_note) so
plan.json stays hand-editable and the scene author consumes them verbatim.
"""
from __future__ import annotations

import json
from typing import Dict, List

from .llm import get_llm, safe_json_list, safe_json_dict, clean_text, logger
from .graph import CharacterGraph

CARE_HOOKS = ["undeserved_misfortune", "kindness_unobserved", "competence",
              "vulnerability_admitted", "humor", "longing_shown_not_told"]


def design_connection_plan(story: Dict, scenes: List[Dict],
                           graph: CharacterGraph, cfg: Dict) -> None:
    """Stamp scene['connection_note'] with concrete empathy direction."""
    logger.info("[connection] Designing care hooks, backstory echoes, and "
                "reaction beats...")
    chars = story.get("characters") or []
    char_block = []
    for c in chars:
        node = graph.get_node(c["name"])
        if not node:
            continue
        char_block.append(
            f"- {c['name']} ({c.get('role','')}): wound={node.core_wound[:90]}"
            f" | backstory={node.backstory_moment[:160]}"
            f" | relic={node.relic or '-'} | tell={node.tell or '-'}")
    scene_digest = "\n".join(
        f"{s['scene_id']}. act{s.get('act')} [{s.get('function','')}] "
        f"T={s.get('tension',0):.2f} chars={','.join(s.get('characters') or [])} "
        f":: {s.get('summary','')[:110]}" for s in scenes)

    prompt = f"""
You are designing the AUDIENCE CONNECTION architecture for this film -- the
specific on-screen beats that make viewers care about these people and feel
their journey as their own.

Cast psychology (each backstory is a universal human experience the audience
has lived some version of -- that recognition is the empathy engine):
{chr(10).join(char_block)}

Scenes:
{scene_digest}

ASSIGN, with restraint (empathy beats work because they're rationed):

1. CARE HOOKS -- in the FIRST QUARTER of the film, give the protagonist 2-3
   distinct hooks from: {', '.join(CARE_HOOKS)}. Give the antagonist or a
   key secondary character ONE (understood is not the same as excused).
   Each hook is a concrete stageable micro-beat inside an existing scene's
   event, never a new scene.

2. BACKSTORY ECHOES -- for each principal, place 2-3 OBLIQUE surfacings of
   their backstory across the film (the relic glimpsed in their hands or
   pocket, the tell firing when the present rhymes with the memory, a
   sentence that starts toward it and stops), and ONE full payoff scene near
   their hardest choice where the memory is finally spoken or shown plainly
   -- at the moment it costs the most. Fragments first, meaning assembled by
   the audience.

3. REACTION BEATS -- for each scene with tension >= 0.6, name WHOSE face we
   hold on immediately after the scene's biggest event, and what their
   personality-specific expression looks like (use their emotion styles;
   a suppressor's devastation is stillness, not tears).

4. MIRROR MOMENTS -- choose 1-2 scenes to stage inside instantly-recognized
   life textures (a waiting room, a school corridor at day's end, the long
   walk to a front door carrying news) so the audience sees their own life
   in the frame.

Return ONLY a raw JSON array, one object per scene THAT RECEIVES direction
(scenes with nothing assigned are omitted):
{{"scene_id": <int>,
  "connection_note": "<2-5 sentences of concrete, stageable direction for
   this scene: which hook/echo/reaction/mirror, who, and exactly what we
   see -- written to a director, not an essayist>"}}
"""
    by_id = {}
    for it in safe_json_list(get_llm(prompt, temperature=0.75, large=True)):
        sid = it.get("scene_id")
        note = clean_text(str(it.get("connection_note", "")))
        if isinstance(sid, int) and note:
            by_id[sid] = note
    for s in scenes:
        if s["scene_id"] in by_id:
            s["connection_note"] = by_id[s["scene_id"]]
    logger.info("[connection] %d scene(s) carry connection direction.",
                len(by_id))


def design_eureka(story: Dict, scenes: List[Dict], cfg: Dict) -> None:
    """Build the film's realization ladder and stamp eureka_note on the
    scenes that carry it. For story mode the realization is the truth arc's
    truth; for educational mode it is the learning objective's click."""
    edu = story.get("educational") or {}
    if edu:
        realization = (f"the concept clicking: {edu.get('learning_objective','')}"
                       f" -- via the core analogy: {edu.get('core_analogy','')}")
    else:
        realization = story.get("truth_arc", {}).get("truth", "")
    if not realization:
        return
    logger.info("[connection] Engineering the eureka ladder...")
    scene_digest = "\n".join(
        f"{s['scene_id']}. act{s.get('act')} [{s.get('function','')}] "
        f"T={s.get('tension',0):.2f} :: {s.get('summary','')[:110]}"
        for s in scenes)
    prompt = f"""
Engineer this film's EUREKA -- the realization the whole film builds to:
"{realization}"

The craft: a realization lands hardest when the AUDIENCE assembles it one
beat before the character does. Build the ladder:

1. THREE CLUES planted in three separate scenes, each a concrete VISIBLE
   image or event that is innocent in its own context but unmistakable in
   hindsight. Spread across acts one and two. Name the exact image.
2. A FALSE FLOOR -- one scene where character and audience almost see it and
   the moment slips away (interrupted, misread, dismissed). This is what
   makes the audience lean in.
3. THE CLICK -- the scene where two of the planted images physically collide
   or align into new meaning IN THE FRAME (a visual reframe, not a line of
   dialogue explaining it). The character may say almost nothing; their
   body does the understanding.
4. THE RELEASE -- the shot(s) right after: the emotion of understanding
   (awe, grief, joy, laughter through tears) expressed in the realizing
   character's own emotion style, and one immediate ACTION the realization
   unlocks.

Scenes:
{scene_digest}

Return ONLY raw JSON:
{{"assignments": [
   {{"scene_id": <int>,
     "eureka_note": "<1-3 sentences of concrete direction: which rung of
      the ladder this scene carries (clue one/two/three, false floor,
      click, release) and exactly what we see>"}}
 ]}}
"""
    data = safe_json_dict(get_llm(prompt, temperature=0.7))
    n = 0
    for it in (data.get("assignments") or []):
        sid = it.get("scene_id")
        note = clean_text(str(it.get("eureka_note", "")))
        target = next((s for s in scenes if s.get("scene_id") == sid), None)
        if target is not None and note:
            target["eureka_note"] = note
            n += 1
    if n:
        story["eureka_ladder_scenes"] = n
        logger.info("[connection] Eureka ladder spans %d scene(s).", n)
