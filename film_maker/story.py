"""
film_maker.story
================
Idea → story. Ports the comic/novel brain's strongest structural machinery,
self-contained:

  develop_story(idea, cfg) →
    story dict:
      premise / logline / genre / themes / mood / world / era
      characters[]            — name, role, summary, LOCKED visual description
      truth_arc               — want vs. need, the lie believed, the truth
      beats[]                 — act-structured beat sheet, register-shaped
      payoff_ledger[]         — planted setups with owed payoffs
      through_line            — one recurring concrete object/image
    plus the EI CharacterGraph (built in graph.py, attached by the pipeline).

Craft passes ported: act-structured beat generation, the CRITIC pass
(critique_and_amplify: attack the outline's weakest beats and rewrite them
stronger), truth-arc construction with climax verification, and the
setup/payoff ledger so nothing planted goes unpaid.
"""
from __future__ import annotations

import math
from typing import Dict, List

from .llm import get_llm, safe_json_dict, safe_json_list, clean_text, logger
from .registers import (story_method_block, content_rating_block,
                        resolve_registers)


# ---------------------------------------------------------------------------
def _cast_targets(cast_size: str) -> str:
    if cast_size == "medium":
        return ("3-4 principal characters, 2-3 supporting. Keep the named "
                "cast under 7 total.")
    return ("2 principal characters, 1-3 supporting. Keep the named cast "
            "under 5 total (a small cast keeps the animation model's "
            "character identity coherent).")


def _beat_count(target_minutes: float) -> int:
    # ~45-60s of screen time per beat gives scenes room to breathe.
    return max(12, min(28, int(round(target_minutes * 60 / 50.0))))


def develop_story(idea: str, cfg: Dict) -> Dict:
    registers = resolve_registers(cfg.get("registers"),
                                  cfg.get("content_rating", "teen"))
    film_mode = str(cfg.get("film_mode", "story")).lower()
    if film_mode == "educational" and "explainer" not in registers:
        registers = ["explainer"] + registers
    reg_block = story_method_block(registers)
    rating_block = content_rating_block(cfg.get("content_rating", "teen"))
    tone = (cfg.get("tone_notes") or "").strip()
    tone_block = f"Additional tonal guidance from the author: {tone}" if tone else ""

    # ------------------------------------------- 0. educational concept spec
    edu = {}
    if film_mode == "educational":
        logger.info("[story] Educational mode: building the concept spec...")
        prompt = f"""
You are a master teacher designing a short film that gives its audience a
genuine EUREKA -- the felt click of understanding, not a recitation of
facts. The topic:

TOPIC: {idea}

Design the pedagogy before the story:

Return ONLY raw JSON:
{{"learning_objective": "<one sentence: what the viewer will genuinely
   UNDERSTAND (be able to re-explain) after watching>",
  "hook_question": "<the concrete curiosity-gap question that opens the
   film -- specific, surprising, answerable only by the concept>",
  "misconception": "<the intuitive WRONG model most people hold, stated the
   way a person would actually say it>",
  "why_wrong_shows": "<the concrete observable failure that exposes the
   wrong model when acted on>",
  "core_analogy": "<ONE physical, filmable analogy vehicle that carries the
   whole concept consistently (choose for extendability -- it must survive
   every key idea without breaking)>",
  "key_ideas": ["<3-5 ideas in strict dependency order, each one scene's
   worth, each phrased as something SEEN not stated>"],
  "payoff_application": "<the concrete problem the understanding solves at
   the end -- the win the character needed it for>",
  "audience_age": "<the youngest audience that should fully follow it>"}}
"""
        edu = safe_json_dict(get_llm(prompt, temperature=0.6))
        edu = {k: (clean_text(str(v)) if not isinstance(v, list) else
                   [clean_text(str(x)) for x in v])
               for k, v in edu.items()}
        if not edu.get("key_ideas"):
            edu["key_ideas"] = ["the core mechanism, seen once, simply"]

    # ---------------------------------------------------------- 1. concept
    logger.info("[story] Expanding the idea into a film concept...")
    edu_block = ""
    if edu:
        edu_block = f"""
EDUCATIONAL VEHICLE (this film must TEACH while it moves us):
- Learning objective: {edu.get('learning_objective','')}
- Hook question that opens the film: {edu.get('hook_question','')}
- The misconception the story dramatizes and breaks: {edu.get('misconception','')}
- The ONE core analogy carried throughout: {edu.get('core_analogy','')}
- Payoff: the understanding must solve {edu.get('payoff_application','')}
Design the protagonist as someone with real personal stakes in this problem
(the audience surrogate whose questions are our questions), and optionally
a guide figure whose own wonder is contagious. The concept IS the plot's
engine, never a lecture inserted into one.
"""
    prompt = f"""
Expand this rough idea into a complete film concept worthy of awards
consideration. The idea is a seed, not a cage: keep its heart, and give it
the strongest possible dramatic form.

IDEA: {idea}
{edu_block}
{reg_block}
{rating_block}
{tone_block}

Return ONLY raw JSON:
{{"title": "<evocative film title>",
  "logline": "<one sentence: protagonist + goal + obstacle + stakes>",
  "premise": "<3-5 sentences expanding the idea into a filmable story>",
  "genre": "<primary genre phrase>",
  "themes": ["<2-4 themes>"],
  "mood": "<the film's dominant emotional weather>",
  "world": "<2-3 sentences: the concrete world/setting rules of this story>",
  "era": "<time period / era, e.g. 'present day', '1890s frontier',
          'far-future orbital colony'>",
  "central_question": "<the dramatic question the whole film asks>",
  "through_line": "<ONE concrete recurring object or image that physically
                   threads the story (a locket, a lit lantern, a cracked
                   watch, a paper boat). It must be able to appear on
                   screen, transformed, at the climax.>"}}
"""
    story = safe_json_dict(get_llm(prompt, temperature=0.8))
    for k, default in (("title", "Untitled Film"), ("logline", idea[:200]),
                       ("premise", idea), ("genre", "Drama"),
                       ("mood", "cinematic"), ("world", ""),
                       ("era", "present day"), ("central_question", ""),
                       ("through_line", "a small keepsake")):
        story[k] = clean_text(str(story.get(k) or default))
    if not isinstance(story.get("themes"), list) or not story["themes"]:
        story["themes"] = ["change"]
    story["themes"] = [clean_text(str(t)) for t in story["themes"]][:4]
    story["registers"] = registers
    if edu:
        story["educational"] = edu
    logger.info("[story] %s -- %s", story["title"], story["logline"])

    # ------------------------------------------------------------- 2. cast
    logger.info("[story] Casting...")
    prompt = f"""
Cast this film. {_cast_targets(cfg.get('cast_size', 'small'))}

Story: {story['logline']}
Premise: {story['premise']}
World: {story['world']}   Era: {story['era']}
{reg_block}
{rating_block}

For each character, the "visual_lock" is CRITICAL: it is reproduced VERBATIM
in every image prompt featuring them, so it must be 60-100 words of purely
VISUAL, concrete, distinctive description: age range, build and height,
skin tone and ethnicity (fit the world), face shape and eyes, exact hair
color/length/style, their DEFAULT costume in full detail (era-appropriate),
and one unmistakable identifying detail (a scar, a pin, glasses, a specific
coat). No backstory, no personality words, no held props. Make every
character instantly distinguishable from every other at a glance.

Return ONLY a raw JSON array:
{{"name": "<first name or short name>",
  "role": "protagonist | antagonist | ally | mentor | ...",
  "summary": "<2-3 sentences: who they are and what they carry into the story>",
  "visual_lock": "<60-100 words as specified>",
  "screen_tag": "<a 4-7 word instantly-recognizable visual identifier used
                 to bind their voice to their on-screen body, e.g. 'the
                 gray-bearded keeper in the oilskin coat'>",
  "wants": "<what they pursue>",
  "voice_texture": "<1 sentence: how their voice should SOUND when the
                    animation model speaks for them: pitch, pace, grain,
                    accent flavor -- e.g. 'low, unhurried, gravel at the
                    edges, faint coastal drawl'>"}}
"""
    characters = []
    for item in safe_json_list(get_llm(prompt, temperature=0.8)):
        name = clean_text(item.get("name", ""))
        if not name:
            continue
        characters.append({
            "name": name,
            "role": clean_text(item.get("role", "")),
            "summary": clean_text(item.get("summary", "")),
            "visual_lock": clean_text(item.get("visual_lock", "")),
            "screen_tag": clean_text(item.get("screen_tag", "")),
            "wants": clean_text(item.get("wants", "")),
            "voice_texture": clean_text(item.get("voice_texture", "")),
        })
    if not characters:
        characters = [{"name": "The Protagonist", "role": "protagonist",
                       "summary": story["premise"][:200],
                       "visual_lock": "an adult of medium height with dark "
                                      "hair and a weathered coat",
                       "wants": "", "voice_texture": ""}]
    story["characters"] = characters
    logger.info("[story] Cast: %s", ", ".join(c["name"] for c in characters))

    # -------------------------------------------------------- 3. truth arc
    logger.info("[story] Building the character truth arc...")
    prot = next((c for c in characters if "protag" in c["role"].lower()),
                characters[0])
    prompt = f"""
Build the protagonist's truth arc -- the spine every scene will press on.

Protagonist: {prot['name']} -- {prot['summary']}
Story: {story['logline']}
Themes: {', '.join(story['themes'])}
{reg_block}

Return ONLY raw JSON:
{{"protagonist": "{prot['name']}",
  "want": "<the external goal they pursue all film>",
  "need": "<the internal truth they must accept>",
  "lie_believed": "<the false belief that blocks the need>",
  "truth": "<the truth, stated plainly, that the climax must dramatize>",
  "climax_requirement": "<the concrete ACTION at the climax that proves the
                         truth accepted (or tragically refused) -- something
                         visible on screen, not a realization>"}}
"""
    story["truth_arc"] = {k: clean_text(str(v)) for k, v in
                          safe_json_dict(get_llm(prompt, temperature=0.6)).items()}

    # ------------------------------------------------------------ 4. beats
    n_beats = _beat_count(float(cfg.get("target_minutes", 16.0)))
    logger.info("[story] Generating %d act-structured beats...", n_beats)
    edu_beat_rules = ""
    if edu:
        idea_ladder = "\n".join(f"   {i+1}. {k}" for i, k in
                                enumerate(edu.get("key_ideas", [])))
        edu_beat_rules = f"""
EDUCATIONAL STRUCTURE (mandatory, woven INTO the drama, never beside it):
- Beat 1 poses the hook question concretely: {edu.get('hook_question','')}
- Early Act 1: the protagonist ACTS on the misconception
  ("{edu.get('misconception','')}") and it visibly fails:
  {edu.get('why_wrong_shows','')}
- Act 2 walks the key-idea ladder IN ORDER, one idea per beat, each idea
  DISCOVERED through story events inside the core analogy
  ("{edu.get('core_analogy','')}"), never explained from outside:
{idea_ladder}
- The climax is the eureka: the ideas assemble and the understanding is
  USED to achieve: {edu.get('payoff_application','')}
- The resolution lets the protagonist re-explain the idea in their own
  words to someone else, briefly and imperfectly, and it lands -- teaching
  it is the proof they own it.
"""
    prompt = f"""
Write the beat sheet: exactly {n_beats} beats in a rigorous three-act
structure for a {cfg.get('target_minutes', 16)}-minute film. Each beat is
one filmable dramatic unit (roughly one scene).

Story: {story['logline']}
Premise: {story['premise']}
World: {story['world']}   Era: {story['era']}
Truth arc: {story['truth_arc']}
Through-line object (must appear, transformed, across the film and at the
climax): {story['through_line']}
Cast: {', '.join(c['name'] + ' (' + c['role'] + ')' for c in characters)}
{reg_block}
{rating_block}
{tone_block}
{edu_beat_rules}

STRUCTURAL REQUIREMENTS:
- Act 1 (~25% of beats): world + want established through ACTION, an opening
  hook in beat 1 that lands inside 60 seconds of screen time, an inciting
  incident, a threshold decision.
- Act 2 (~50%): escalating attempts and costs, a midpoint that changes what
  the goal MEANS, a low point where the lie_believed fails them completely.
- Act 3 (~25%): the climax beat must dramatize the climax_requirement above,
  then a resolution that shows the changed world in images, not speech.
- TENSION (0.0-1.0) must chart a real curve: valleys after peaks, an overall
  rise, the maximum at the climax. Never three consecutive beats within 0.1
  of each other.
- Every beat is a concrete EVENT someone could film, never an abstraction.

Return ONLY a raw JSON array, one object per beat, in order:
{{"beat_id": <int, 1-based>,
  "act": 1|2|3,
  "function": "hook|setup|inciting|threshold|escalation|midpoint|low_point|
               climax|resolution|...",
  "summary": "<2-3 sentences: the concrete filmable event>",
  "location_hint": "<where this wants to happen>",
  "characters": ["<names present>"],
  "tension": <0.0-1.0>,
  "emotion": "<the dominant feeling this beat should leave in the audience>"}}
"""
    beats = safe_json_list(get_llm(prompt, temperature=0.8, large=True))
    beats = [b for b in beats if isinstance(b, dict) and b.get("summary")]
    for i, b in enumerate(beats):
        b["beat_id"] = i + 1
        b["summary"] = clean_text(str(b.get("summary", "")))
        b["location_hint"] = clean_text(str(b.get("location_hint", "")))
        b["emotion"] = clean_text(str(b.get("emotion", "")))
        try:
            b["tension"] = max(0.0, min(1.0, float(b.get("tension", 0.5))))
        except (TypeError, ValueError):
            b["tension"] = 0.5
        if not isinstance(b.get("characters"), list):
            b["characters"] = []
    story["beats"] = beats

    # ------------------------------------------------- 5. critic pass
    story["beats"] = critique_and_amplify(story, cfg, reg_block)

    # ------------------------------------------- 6. setup/payoff ledger
    logger.info("[story] Building the setup/payoff ledger...")
    beat_digest = "\n".join(f"{b['beat_id']}. [{b.get('function','')}] "
                            f"{b['summary'][:160]}" for b in story["beats"])
    prompt = f"""
Audit this beat sheet as a setup/payoff ledger. Identify every planted
element (object, promise, skill, image, line) and where it must pay off.
Include the through-line object "{story['through_line']}". Add up to 3 NEW
plants worth inserting early to make late beats land harder.

Beats:
{beat_digest}

Return ONLY a raw JSON array:
{{"element": "<the planted thing>",
  "setup_beat": <beat_id where planted>,
  "payoff_beat": <beat_id where it must pay off>,
  "transformation": "<how its meaning changes between setup and payoff>"}}
Keep every string value under 25 words; at most 10 objects total.
"""
    ledger = safe_json_list(get_llm(prompt, temperature=0.5))
    story["payoff_ledger"] = [
        {"element": clean_text(str(x.get("element", ""))),
         "setup_beat": int(x.get("setup_beat", 1) or 1),
         "payoff_beat": int(x.get("payoff_beat", len(story["beats"])) or 1),
         "transformation": clean_text(str(x.get("transformation", "")))}
        for x in ledger if x.get("element")][:10]

    return story


def critique_and_amplify(story: Dict, cfg: Dict, reg_block: str) -> List[Dict]:
    """Port of the comic pipeline's beat critic: attack the weakest beats,
    rewrite only those, keep everything else untouched."""
    beats = story.get("beats") or []
    if len(beats) < 6:
        return beats
    logger.info("[story] Critic pass: attacking the weakest beats...")
    edu = story.get("educational") or {}
    edu_critique = ""
    if edu:
        edu_critique = (
            f"\nADDITIONALLY, as a learning-science reviewer: verify the "
            f"key-idea ladder appears in order and each idea is SHOWN inside "
            f"the core analogy ('{edu.get('core_analogy','')}'), not stated; "
            f"flag any beat that lectures, breaks the analogy, or introduces "
            f"more than one new idea; verify the misconception visibly fails "
            f"before the true model begins. A beat that teaches nothing and "
            f"moves nothing is the weakest kind -- rewrite it to do both.\n")
    digest = "\n".join(f"{b['beat_id']}. act{b.get('act','?')} "
                       f"[{b.get('function','')}] T={b.get('tension',0):.2f} "
                       f"{b['summary'][:170]}" for b in beats)
    prompt = f"""
You are a ruthless story editor. Find the 3-5 WEAKEST beats in this sheet --
generic events, unearned turns, low-stakes filler, beats that state emotion
instead of dramatizing it, or missed opportunities the registers demand
(a comedy beat with no comic engine, a horror beat with no dread mechanism).
Rewrite ONLY those beats to be specific, surprising, and register-true,
preserving each beat's structural function, act, characters, and its place
in the tension curve.
{edu_critique}
{reg_block}
Truth arc: {story.get('truth_arc', {})}
Through-line: {story.get('through_line', '')}

Beat sheet:
{digest}

Return ONLY a raw JSON array of the REWRITTEN beats (3-5 objects), each:
{{"beat_id": <int, copied>, "summary": "<the stronger version, 2-3 sentences>",
  "why": "<one line: what was weak>"}}
"""
    fixes = safe_json_list(get_llm(prompt, temperature=0.85))
    by_id = {b["beat_id"]: b for b in beats}
    n = 0
    for fx in fixes:
        bid = fx.get("beat_id")
        new = clean_text(str(fx.get("summary", "")))
        if bid in by_id and new and len(new) > 40:
            by_id[bid]["summary"] = new
            n += 1
    logger.info("[story] Critic strengthened %d beat(s).", n)
    return beats
