"""
film_maker.pipeline
===================
Orchestration. Two-phase, like story_to_animation's proven workflow:

  PHASE 1 — plan_film(idea, cfg)
    idea → story (beats, truth arc, payoff ledger, critic pass)
         → EI CharacterGraph
         → screenplay (scenes/shots/dialogue, repetition-guarded)
         → cinematography (style bible, locations, wardrobe, global shot
           plan, image prompts, period scrub, motion + soundscape design,
           sound motifs)
    Everything lands in an EDITABLE plan.json (+ readable plan.md). No image
    or video model loads in this phase — it's pure authoring, cheap to
    iterate.

  ...open plan.json, change any line / action / prompt / duration...

  PHASE 2 — produce_film(plan_path, cfg)
    reads the (edited) plan → stills (N variants + vision pick) → H3 render
    in voice-chaining waves → assemble. Idempotent: re-run to resume;
    finished stills/clips are reused.

  regenerate_scenes(plan_path, cfg, [ids]) / regenerate_shots(...)
    delete the targeted clips (and optionally stills) and re-run produce for
    just those, then reassemble.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional

from . import llm
from .llm import logger, clean_text
from .story import develop_story
from .graph import build_character_graph, CharacterGraph
from .registers import story_method_block, resolve_registers
from .screenplay import write_screenplay
from .cinematography import (generate_style_bible, design_locations,
                             plan_wardrobe, plan_shot_compositions,
                             write_image_prompts, write_motion_and_sound,
                             generate_color_script, design_cuts,
                             compute_dynamics, track_physical_state,
                             plan_long_take_phases)
from .images import generate_stills
from .render import build_scene_jobs, render_all
from .assembly import assemble_film


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s or "film").strip("_")[:48] or "film"


def _dirs(cfg: Dict, slug: str) -> Dict[str, str]:
    # ONE absolute project root; wgp.py runs with cwd set to the Wan2GP
    # repo, so every path must be absolute. Legacy split-layout projects
    # (film_work/<slug> + film_output/<slug>) are auto-migrated in on first
    # touch so old runs aren't stranded.
    root = os.path.abspath(
        os.path.join(cfg.get("base_dir", "./films"), slug))
    _migrate_legacy_layout(root, cfg, slug)
    d = {"root": root, "work": root, "output": root,
         "images": os.path.join(root, "stills"),
         "voices": os.path.join(root, "voice_bank"),
         "clips": os.path.join(root, "clips"),
         "render": os.path.join(root, "render")}
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


def _migrate_legacy_layout(root: str, cfg: Dict, slug: str) -> None:
    """Move a pre-v3.5 project (film_work/<slug> + film_output/<slug>) into
    the unified <base_dir>/<slug>/ layout. Runs once: skipped when the
    unified root already holds a plan or when no legacy dirs exist."""
    if os.path.exists(os.path.join(root, "plan.json")):
        return
    legacy_work = os.path.abspath(
        os.path.join(cfg.get("base_working_dir", "./film_work"), slug))
    legacy_out = os.path.abspath(
        os.path.join(cfg.get("base_output_dir", "./film_output"), slug))
    if not (os.path.isdir(legacy_work) or os.path.isdir(legacy_out)):
        return
    if root in (legacy_work, legacy_out):
        return
    logger.info("[project] Migrating legacy layout for '%s' into %s ...",
                slug, root)
    os.makedirs(root, exist_ok=True)
    moves = []
    if os.path.isdir(legacy_work):
        for name in os.listdir(legacy_work):
            moves.append((os.path.join(legacy_work, name),
                          os.path.join(root, name)))
    if os.path.isdir(legacy_out):
        rename_map = {"shot_clips": "clips", "wgp_render": "render"}
        for name in os.listdir(legacy_out):
            moves.append((os.path.join(legacy_out, name),
                          os.path.join(root, rename_map.get(name, name))))
    moved = 0
    for src, dst in moves:
        try:
            if os.path.exists(dst):
                logger.warning("[project]   skip %s (already exists at "
                               "destination)", os.path.basename(src))
                continue
            shutil.move(src, dst)
            moved += 1
        except OSError as e:
            logger.warning("[project]   couldn't move %s: %s", src, e)
    for legacy in (legacy_work, legacy_out):
        try:
            if os.path.isdir(legacy) and not os.listdir(legacy):
                os.rmdir(legacy)
        except OSError:
            pass
    logger.info("[project] Migration done: %d item(s) moved. Old clip paths "
                "inside plan.json are recomputed on the next produce.", moved)


def find_plan(cfg: Dict) -> str:
    """The current project's plan.json WITHOUT re-planning: derived from
    cfg as base_dir/<slug(project_name)>/plan.json, so the notebook flow
    after a restart is simply:

        final = produce_film(cfg=FILM_CONFIG)          # auto-finds plan
        # or explicitly:
        final = produce_film(find_plan(FILM_CONFIG), FILM_CONFIG)

    With no project_name set, falls back to the most recently modified
    plan.json under base_dir. Raises FileNotFoundError (with the path it
    looked at) if no plan exists yet -- run plan_film() once first."""
    import glob
    base = cfg.get("base_dir", "./films")
    name = (cfg.get("project_name") or "").strip()
    if name:
        p = os.path.abspath(os.path.join(base, _slug(name), "plan.json"))
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"No plan at {p} -- run plan_film() first, or check "
                f"project_name/base_dir in your config.")
        return p
    candidates = glob.glob(os.path.join(base, "*", "plan.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No plan.json found under {os.path.abspath(base)} -- run "
            f"plan_film() first.")
    return os.path.abspath(max(candidates, key=os.path.getmtime))


def resolve_plan_path(plan_path: str, cfg: Dict) -> str:
    """Accept old plan paths gracefully: if the given path doesn't exist,
    look for the plan in the unified layout under the same project slug
    (covers plans migrated by _migrate_legacy_layout)."""
    if os.path.exists(plan_path):
        return plan_path
    slug = os.path.basename(os.path.dirname(os.path.abspath(plan_path)))
    candidate = os.path.abspath(os.path.join(
        cfg.get("base_dir", "./films"), slug, "plan.json"))
    if os.path.exists(candidate):
        logger.info("[project] Plan found at migrated location: %s", candidate)
        return candidate
    return plan_path


# ---------------------------------------------------------------------------
# Planning stage cache — a 15-minute film's plan is ~40+ LLM calls; a crash
# or a rate-limit at stage 6 shouldn't cost stages 1-5. With a project_name
# set, the cache lives INSIDE the project folder (planning_cache.json), so a
# project is fully self-contained and multiple films coexist cleanly. The
# cache stores a fingerprint of (idea, registers, rating, length, tone,
# mode); if any of those change, the cache is invalidated automatically so
# stale stages from a different creative brief are never mixed in.
# ---------------------------------------------------------------------------
def _fingerprint(idea: str, cfg: Dict) -> str:
    import hashlib
    return hashlib.md5(
        f"{idea}|{cfg.get('registers')}|{cfg.get('content_rating')}|"
        f"{cfg.get('target_minutes')}|{cfg.get('tone_notes','')}|"
        f"{cfg.get('film_mode','story')}".encode()).hexdigest()[:12]


def _cache_path(idea: str, cfg: Dict) -> str:
    proj = (cfg.get("project_name") or "").strip()
    if proj:
        d = os.path.abspath(os.path.join(cfg.get("base_dir", "./films"),
                                         _slug(proj)))
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "planning_cache.json")
    d = os.path.abspath(os.path.join(cfg.get("base_dir", "./films"),
                                     "_planning_cache"))
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{_fingerprint(idea, cfg)}.json")


class _StageCache:
    def __init__(self, path: str, resume: bool, fingerprint: str):
        self.path = path
        self.fingerprint = fingerprint
        self.data: Dict = {}
        if resume and os.path.exists(path):
            try:
                with open(path) as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
            if self.data and self.data.get("_fp") != fingerprint:
                logger.warning(
                    "[plan] Cached planning stages were built from a "
                    "DIFFERENT idea/config (fingerprint mismatch); starting "
                    "fresh so briefs never mix.")
                self.data = {}
            elif self.data:
                logger.info("[plan] Resuming: %d cached stage(s) found.",
                            len([k for k in self.data if k != "_fp"]))
        self.data["_fp"] = fingerprint

    def get_or_run(self, stage: str, fn):
        if stage in self.data:
            logger.info("[plan] Stage '%s' loaded from checkpoint.", stage)
            return self.data[stage]
        result = fn()
        self.data[stage] = result
        with open(self.path, "w") as f:
            json.dump(self.data, f)
        return result


# ---------------------------------------------------------------------------
# PHASE 1
# ---------------------------------------------------------------------------
def plan_film(idea: str, cfg: Dict, resume: bool = True) -> str:
    llm.bind_config(cfg)
    cfg["registers"] = resolve_registers(cfg.get("registers"),
                                         cfg.get("content_rating", "teen"))
    logger.info("=" * 70)
    logger.info("FILM MAKER -- planning phase")
    logger.info("Registers: %s | Rating: %s | Target: %.1f min",
                ", ".join(cfg["registers"]), cfg.get("content_rating"),
                float(cfg.get("target_minutes", 16)))
    logger.info("=" * 70)

    # Plan-aware fast path: with a project_name set and a finished
    # plan.json on disk, plan_film is idempotent -- if the creative brief
    # (idea/registers/rating/length/tone/mode fingerprint) is unchanged,
    # it returns the existing plan immediately with zero LLM calls; if
    # the brief HAS changed, it refuses to silently re-plan over a
    # finished film (a reworded idea string would otherwise invalidate
    # the cache and rebuild everything). Re-plan deliberately with
    # resume=False, or change project_name for a new film.
    if resume and (cfg.get("project_name") or "").strip():
        try:
            existing = find_plan(cfg)
        except FileNotFoundError:
            existing = None
        if existing:
            cpath = _cache_path(idea, cfg)
            cached_fp = None
            if os.path.exists(cpath):
                try:
                    with open(cpath) as f:
                        cached_fp = json.load(f).get("fingerprint")
                except Exception:
                    pass
            if cached_fp == _fingerprint(idea, cfg) or cached_fp is None:
                logger.info("[plan] Plan already exists for this project"
                            " -- returning %s (produce_film(cfg=...) "
                            "finds it automatically; pass resume=False "
                            "to re-plan from scratch).", existing)
                return existing
            raise ValueError(
                "plan_film: a finished plan exists for project "
                f"'{cfg.get('project_name')}' but the idea/config "
                "fingerprint changed -- refusing to silently re-plan "
                "over it. Either restore the original idea string, pass "
                "resume=False to deliberately rebuild, or set a new "
                "project_name.")

    cache = _StageCache(_cache_path(idea, cfg), resume,
                        _fingerprint(idea, cfg))

    story = cache.get_or_run("story", lambda: develop_story(idea, cfg))
    slug = _slug(cfg.get("project_name") or story.get("title", "film"))
    dirs = _dirs(cfg, slug)

    reg_block = story_method_block(story["registers"])
    graph_dict = cache.get_or_run(
        "graph", lambda: build_character_graph(
            story["characters"], story, reg_block).to_dict())
    graph = CharacterGraph.from_dict(graph_dict)

    scenes = cache.get_or_run(
        "screenplay", lambda: write_screenplay(story, graph, cfg))

    style_bible = cache.get_or_run(
        "style_bible", lambda: generate_style_bible(story, cfg))
    locations = cache.get_or_run(
        "locations", lambda: design_locations(story, scenes, cfg))
    # design_locations stamps scene["location"]; when the scenes stage was
    # itself cached, restamp from the cached location assignments.
    _restamp_locations(scenes, locations)
    wardrobe = cache.get_or_run(
        "wardrobe", lambda: plan_wardrobe(story, scenes, cfg))

    def _stage_shots():
        plan_shot_compositions(story, scenes, cfg)
        track_physical_state(scenes, story, cfg)
        generate_color_script(story, scenes, style_bible)
        design_cuts(scenes, story, locations)
        compute_dynamics(scenes)
        write_image_prompts(scenes, story, style_bible, locations,
                            wardrobe, cfg, graph=graph)
        write_motion_and_sound(scenes, story, locations, cfg, graph=graph)
        plan_long_take_phases(scenes, story, cfg)
        return scenes
    scenes = cache.get_or_run("shot_design", _stage_shots)

    plan = {
        "version": 2,
        "created": datetime.now().isoformat(timespec="seconds"),
        "idea": idea,
        "slug": slug,
        "story": story,
        "graph": graph.to_dict(),
        "style_bible": style_bible,
        "locations": locations,
        "wardrobe": wardrobe,
        "scenes": scenes,
    }
    plan_path = os.path.join(dirs["work"], "plan.json")
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    _write_plan_md(plan, os.path.join(dirs["work"], "plan.md"))
    _table_read(plan, os.path.join(dirs["work"], "table_read.md"))
    n_shots = sum(len(s.get("shots", [])) for s in scenes)
    total = sum(sh.get("duration", 0) for s in scenes
                for sh in s.get("shots", []))
    logger.info("=" * 70)
    logger.info("Plan written: %s", plan_path)
    logger.info("%d scenes, %d shots, ~%.1f minutes of film.",
                len(scenes), n_shots, total / 60.0)
    logger.info("Review table_read.md for the story editor's notes, edit "
                "anything in plan.json, then run produce_film(plan_path).")
    logger.info("=" * 70)
    return plan_path


def _restamp_locations(scenes: List[Dict], locations: Dict) -> None:
    """After a cache resume, scenes may lack scene['location'] stamps."""
    if all(s.get("location") for s in scenes):
        return
    by_scene = {}
    for name, loc in locations.items():
        for sid in loc.get("scene_ids", []) if isinstance(loc, dict) else []:
            by_scene[sid] = name
    for s in scenes:
        if not s.get("location"):
            s["location"] = by_scene.get(s["scene_id"],
                                         next(iter(locations), ""))


def _table_read(plan: Dict, path: str) -> None:
    """Non-destructive story-editor pass over the finished plan: a table
    read that flags the weakest scenes, flat dialogue, pacing lulls, and
    unpaid setups -- written to table_read.md for the human review loop
    (nothing is auto-changed; you decide what to edit in plan.json)."""
    logger.info("[plan] Running the table read...")
    story = plan["story"]
    digest_lines = []
    for sc in plan["scenes"]:
        digest_lines.append(
            f"Scene {sc['scene_id']} (act {sc.get('act')}, "
            f"T={sc.get('tension',0):.2f}) @ {sc.get('location','')}: "
            f"{sc.get('summary','')[:130]}")
        for sh in sc.get("shots", []):
            for ln in sh.get("lines", []):
                digest_lines.append(f"    {ln.get('speaker','')}: "
                                    f"\"{ln.get('text','')}\"")
    ledger = "\n".join(
        f"- {p.get('element','')} (setup beat {p.get('setup_beat')}, pays "
        f"off beat {p.get('payoff_beat')})"
        for p in story.get("payoff_ledger", []))
    edu = story.get("educational") or {}
    edu_check = ""
    if edu:
        edu_check = (
            f"8. COMPREHENSION CHECK (this is an educational film): after "
            f"watching, could a smart viewer of age "
            f"{edu.get('audience_age','12')} re-explain "
            f"\"{edu.get('learning_objective','the concept')}\" in their own "
            f"words? Walk the key-idea ladder scene by scene: is each idea "
            f"SHOWN inside the core analogy, in order, one per scene? Name "
            f"any idea that is told instead of shown, any break in the "
            f"analogy, and whether the eureka click is visual or explained.\n")
    prompt = f"""
You are a veteran story editor at a table read of this screenplay. Write
frank, USEFUL notes -- the kind that make the next draft better, not
compliments. Cover, with specific scene/line references:
1. EMPATHY AUDIT: by which scene do we genuinely care about the
   protagonist, and which specific beat earns it? If the answer is later
   than the first quarter, name where a care hook should move. Does each
   character express emotion in their OWN way, or do they blur together?
2. The 3 weakest scenes and exactly why (generic event, unearned turn,
   stated emotion, register promise unkept).
3. Dialogue: which exchanges are flat, on-the-nose, or interchangeable
   between characters; quote the worst offenders.
4. Pacing: where the film sags or rushes against its tension curve.
5. The setup/payoff ledger below: anything planted that never visibly pays
   off in a scene, or paying off without a plant. Include each character's
   backstory echoes -- fragments must assemble; a backstory that never
   surfaces is a broken promise.
6. VOICE & NARRATION AUDIT: does the film open with orientation
   narration that plants the central question, and close with a bookend
   that answers it? Where does narration describe what the frame already
   shows (cut it) vs voice true interiority (keep it)? Which scenes are
   voiceless -- earned silence or a gap? Do delivery cues carry real
   emotional color, and does the narrator's delivery shift with the
   moment's temperature (intimate for private thoughts, hot for anger,
   slow for grief)?
7. The 3 strongest moments (so edits don't break them).
{edu_check}Keep it under 900 words, in markdown.

Truth arc: {story.get('truth_arc', {})}
Payoff ledger:
{ledger}

Screenplay digest:
{chr(10).join(digest_lines)}
"""
    from .llm import get_llm as _g
    notes = _g(prompt, temperature=0.6, large=True)
    behavior_report = story.get("behavior_report") or ""
    with open(path, "w") as f:
        f.write(f"# Table read -- {story.get('title','')}\n\n"
                f"{notes or '(table read unavailable)'}\n")
        if behavior_report:
            f.write(f"\n---\n\n## {behavior_report}\n")


def _write_plan_md(plan: Dict, path: str) -> None:
    """Human-readable screenplay view of the plan for quick review."""
    s = plan["story"]
    lines = [f"# {s.get('title','Untitled')}",
             f"*{s.get('logline','')}*", "",
             f"Registers: {', '.join(s.get('registers', []))} | "
             f"Era: {s.get('era','')} | Through-line: {s.get('through_line','')}",
             "", "## Cast"]
    for c in s.get("characters", []):
        lines.append(f"- **{c['name']}** ({c.get('role','')}) — "
                     f"{c.get('summary','')}")
        node = (plan.get("graph", {}).get("nodes", {}) or {}).get(c["name"], {})
        if node.get("backstory_moment"):
            lines.append(f"  - *Backstory ({node.get('backstory_source','')}):* "
                         f"{node['backstory_moment']}")
        if node.get("relic") or node.get("tell"):
            lines.append(f"  - *Relic:* {node.get('relic','—')} · "
                         f"*Tell:* {node.get('tell','—')}")
        if node.get("emotion_style"):
            styles = "; ".join(f"{k}: {v}" for k, v in
                               list(node["emotion_style"].items())[:6])
            lines.append(f"  - *Expresses emotion:* {styles}")
    if s.get("educational"):
        edu = s["educational"]
        lines.append("\n## Pedagogy")
        lines.append(f"- **Objective:** {edu.get('learning_objective','')}")
        lines.append(f"- **Hook:** {edu.get('hook_question','')}")
        lines.append(f"- **Misconception broken:** {edu.get('misconception','')}")
        lines.append(f"- **Core analogy:** {edu.get('core_analogy','')}")
        for i, k in enumerate(edu.get("key_ideas", []), 1):
            lines.append(f"  {i}. {k}")
    lines.append("\n## Screenplay")
    for sc in plan.get("scenes", []):
        atmo = (" · designed silence" if sc.get("voice_design") ==
                "atmospheric" else "")
        lines.append(f"\n### Scene {sc['scene_id']} — {sc.get('location','')} "
                     f"(act {sc.get('act')}, tension {sc.get('tension',0):.2f}"
                     f"{atmo})")
        lines.append(sc.get("summary", ""))
        if sc.get("connection_note"):
            lines.append(f"> **Connection:** {sc['connection_note']}")
        if sc.get("eureka_note"):
            lines.append(f"> **Eureka:** {sc['eureka_note']}")
        for sh in sc.get("shots", []):
            lines.append(f"\n**{sh['shot_id']}** ({sh.get('duration',0):.0f}s, "
                         f"{sh.get('shot_scale','')} {sh.get('camera_angle','')})")
            lines.append(f"- Action: {sh.get('action','')}")
            for ln in sh.get("lines", []):
                lines.append(f"- {ln.get('speaker','')} "
                             f"({ln.get('delivery','')}): "
                             f"\"{ln.get('text','')}\"")
            lines.append(f"- Sound: {sh.get('soundscape','')}")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# PHASE 2
# ---------------------------------------------------------------------------
def _load_plan(plan_path: str) -> Dict:
    with open(plan_path) as f:
        return json.load(f)


def produce_film(plan_path: Optional[str] = None, cfg: Dict = None,
                 only_shot_ids: Optional[set] = None) -> Optional[str]:
    llm.bind_config(cfg)
    if cfg is None:
        from .config import FILM_CONFIG as cfg  # noqa: F811
    if not plan_path:
        plan_path = find_plan(cfg)
    plan_path = resolve_plan_path(plan_path, cfg)
    plan = _load_plan(plan_path)
    story = plan["story"]
    scenes = plan["scenes"]
    style_bible = plan["style_bible"]
    slug = plan.get("slug") or _slug(story.get("title", "film"))
    dirs = _dirs(cfg, slug)

    logger.info("=" * 70)
    logger.info("FILM MAKER -- production phase: %s", story.get("title"))
    logger.info("=" * 70)

    # Map an optional shot-id selection onto its SCENES (the render unit).
    only_scene_ids = None
    if only_shot_ids:
        only_scene_ids = {sc["scene_id"] for sc in scenes
                          for sh in sc.get("shots", [])
                          if sh["shot_id"] in set(only_shot_ids)}

    # 1. Stills (variants + vision pick; idempotent). ONE still per scene
    #    -- its opening frame / persistent identity reference. Skipped
    #    entirely with use_reference_stills=False (text-only identity
    #    locks carry consistency instead).
    if cfg.get("use_reference_character_images", False):
        from .images import generate_character_refs
        generate_character_refs(plan["story"], plan.get("style_bible"),
                                os.path.join(dirs["images"], "characters"),
                                cfg)
        _save_plan(plan, plan_path)   # persist ref_image paths
    if cfg.get("use_reference_stills", True):
        openers = {sc["shots"][0]["shot_id"] for sc in scenes
                   if sc.get("shots")}
        stills_ids = (openers if only_scene_ids is None else
                      {sc["shots"][0]["shot_id"] for sc in scenes
                       if sc.get("shots")
                       and sc["scene_id"] in only_scene_ids})
        logger.info("[pipeline] %d scene-opening still(s) needed (one per "
                    "scene).", len(stills_ids))
        generate_stills(scenes, story, style_bible, dirs["images"], cfg,
                        only_shot_ids=stills_ids)
    else:
        logger.info("[pipeline] use_reference_stills=False: skipping "
                    "Z-Image entirely; identity rides in the prompts.")

    # 2. H3 render: ONE continuous generation per scene (idempotent;
    #    retries transient GPU errors; reuses finished clips).
    jobs = build_scene_jobs(scenes, story, style_bible, dirs, cfg,
                            only_scene_ids=only_scene_ids,
                            wardrobe=plan.get("wardrobe"),
                            locations=plan.get("locations"))
    render_all(jobs, dirs, cfg, scenes=scenes)

    # Persist clip paths + render status back into the plan.
    status = {j["scene_id"]: bool(j.get("_rendered")) for j in jobs}
    for sc in scenes:
        if sc["scene_id"] in status:
            sc["rendered"] = status[sc["scene_id"]]
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)

    # 3. Assemble.
    if not cfg.get("concat_final_video", True):
        logger.info("[pipeline] concat_final_video=False; shot clips are in "
                    "%s", dirs["clips"])
        return None
    final = assemble_film(scenes, dirs, cfg, slug)
    if final and not cfg.get("keep_shot_clips", True):
        shutil.rmtree(dirs["clips"], ignore_errors=True)
    return final


# ---------------------------------------------------------------------------
# Targeted regeneration
# ---------------------------------------------------------------------------
def regenerate_scenes(plan_path: Optional[str] = None, cfg: Dict = None,
                      scene_ids: List[int] = None,
                      regenerate_stills: bool = False) -> Optional[str]:
    if not scene_ids:
        raise ValueError("regenerate_scenes: pass scene_ids=[...] (plan_path/cfg are now optional, but scene_ids is required).")
    """Delete the clips (and optionally stills) for the given scenes and
    re-produce just those shots, then reassemble the film."""
    if cfg is None:
        from .config import FILM_CONFIG as cfg  # noqa: F811
    if not plan_path:
        plan_path = find_plan(cfg)
    plan_path = resolve_plan_path(plan_path, cfg)
    plan = _load_plan(plan_path)
    targets = set()
    for sc in plan["scenes"]:
        if sc["scene_id"] in set(scene_ids):
            for sh in sc.get("shots", []):
                targets.add(sh["shot_id"])
    return regenerate_shots(plan_path, cfg, sorted(targets),
                            regenerate_stills=regenerate_stills)


def regenerate_shots(plan_path: Optional[str] = None, cfg: Dict = None,
                     shot_ids: List[str] = None,
                     regenerate_stills: bool = False) -> Optional[str]:
    if not shot_ids:
        raise ValueError("regenerate_shots: pass shot_ids=[...] (plan_path/cfg are now optional, but shot_ids is required).")
    llm.bind_config(cfg)
    if cfg is None:
        from .config import FILM_CONFIG as cfg  # noqa: F811
    if not plan_path:
        plan_path = find_plan(cfg)
    plan_path = resolve_plan_path(plan_path, cfg)
    plan = _load_plan(plan_path)
    slug = plan.get("slug") or _slug(plan["story"].get("title", "film"))
    dirs = _dirs(cfg, slug)
    targets = set(shot_ids)
    if not targets:
        logger.warning("[regen] No shots targeted.")
        return None
    logger.info("[regen] Re-rolling %d shot(s): %s",
                len(targets), ", ".join(sorted(targets)))
    # The render unit is the SCENE: regenerating any shot re-renders its
    # whole scene (one continuous generation), so delete that scene's clip.
    scene_of = {sh["shot_id"]: sc for sc in plan["scenes"]
                for sh in sc.get("shots", [])}
    for sid in targets:
        sc = scene_of.get(sid)
        if not sc:
            continue
        clip = os.path.join(dirs["clips"],
                            f"scene_{int(sc['scene_id']):03d}.mp4")
        if os.path.exists(clip):
            os.remove(clip)
        if regenerate_stills and sc.get("shots"):
            still = os.path.join(
                dirs["images"], f"image_{sc['shots'][0]['shot_id']}.jpg")
            if os.path.exists(still):
                os.remove(still)
    # produce for just these shots, then a full reassembly (produce assembles
    # from ALL scenes' clip paths, so the film is rebuilt whole).
    return produce_film(plan_path, cfg, only_shot_ids=targets)


# ---------------------------------------------------------------------------
# Project overview
# ---------------------------------------------------------------------------
def list_projects(cfg: Dict) -> List[Dict]:
    """Scan <base_dir> and report every project's state: title, shot counts,
    stills done, clips rendered, final film. Returns the data and logs a
    readable table -- the notebook's 'where is everything' call."""
    base = os.path.abspath(cfg.get("base_dir", "./films"))
    out: List[Dict] = []
    if not os.path.isdir(base):
        logger.info("[projects] No projects yet under %s", base)
        return out
    for name in sorted(os.listdir(base)):
        root = os.path.join(base, name)
        if not os.path.isdir(root) or name.startswith("_"):
            continue
        info = {"project": name, "root": root, "title": "", "shots": 0,
                "stills": 0, "clips": 0, "final": None, "minutes": 0.0}
        plan_file = os.path.join(root, "plan.json")
        if os.path.exists(plan_file):
            try:
                with open(plan_file) as f:
                    plan = json.load(f)
                info["title"] = plan.get("story", {}).get("title", "")
                shots = [sh for sc in plan.get("scenes", [])
                         for sh in sc.get("shots", [])]
                info["shots"] = len(shots)
                info["minutes"] = round(
                    sum(sh.get("duration", 0) for sh in shots) / 60.0, 1)
            except Exception:
                pass
        stills_dir = os.path.join(root, "stills")
        if os.path.isdir(stills_dir):
            info["stills"] = len([f for f in os.listdir(stills_dir)
                                  if f.startswith("image_")
                                  and "_v" not in f])
        clips_dir = os.path.join(root, "clips")
        if os.path.isdir(clips_dir):
            info["clips"] = len([f for f in os.listdir(clips_dir)
                                 if f.endswith(".mp4")])
        finals = sorted(f for f in os.listdir(root)
                        if f.endswith("_final.mp4") or "_final_" in f)
        if finals:
            info["final"] = os.path.join(root, finals[-1])
        out.append(info)
        logger.info("[projects] %-24s %-28s %3d shots | %3d stills | "
                    "%3d clips | ~%.1f min | final: %s",
                    name, (info["title"][:26] or "-"), info["shots"],
                    info["stills"], info["clips"], info["minutes"],
                    "yes" if info["final"] else "no")
    return out


def prune_unused_stills(plan_path: Optional[str] = None, cfg: Dict = None,
                        delete: bool = False) -> List[str]:
    """Housekeeping for projects upgraded from the pre-chaining
    architecture (which synthesized a still for EVERY shot): finds stills
    that are no longer block openers -- and therefore never used as render
    references -- and moves them into stills/_unused/ (or deletes them
    outright with delete=True). Variant files (_vN) follow their base
    image. Returns the list of pruned base shot_ids."""
    if cfg is None:
        from .config import FILM_CONFIG as cfg  # noqa: F811
    if not plan_path:
        plan_path = find_plan(cfg)
    plan_path = resolve_plan_path(plan_path, cfg)
    plan = _load_plan(plan_path)
    scenes = plan["scenes"]
    slug = plan.get("slug") or _slug(plan["story"].get("title", "film"))
    dirs = _dirs(cfg, slug)
    # v6: the only still each scene needs is its opener (first shot).
    openers = {sc["shots"][0]["shot_id"] for sc in scenes
               if sc.get("shots")}
    stills_dir = dirs["images"]
    unused_dir = os.path.join(stills_dir, "_unused")
    pruned = []
    for f in sorted(os.listdir(stills_dir)):
        if not (f.startswith("image_") and f.endswith(".jpg")):
            continue
        base = f[len("image_"):-len(".jpg")]
        shot_id = base.split("_v")[0]
        if shot_id in openers:
            continue
        src = os.path.join(stills_dir, f)
        if delete:
            os.remove(src)
        else:
            os.makedirs(unused_dir, exist_ok=True)
            shutil.move(src, os.path.join(unused_dir, f))
        if "_v" not in base and shot_id not in pruned:
            pruned.append(shot_id)
    logger.info("[stills] Pruned %d non-opener still(s)%s. Block openers "
                "kept: %d.", len(pruned),
                " (deleted)" if delete else " -> stills/_unused/",
                len(openers))
    return pruned
