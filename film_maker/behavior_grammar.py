"""
film_maker.behavior_grammar
===========================
The body-language equivalent of CAMERA_GRAMMAR_BY_TENSION: LLMs converge
on the same dozen stock gestures (clenched jaw, gripped railing, a single
tear), so -- exactly as the camera grammar fixed camera repetition -- scene
authoring gets a MENU of concrete, stageable behavior per emotion, split by
expression mode (CONTAINED for suppressors, EXPRESSIVE for exploders; a
character's own emotion_style decides which column they act from), and a
deterministic post-pass counts gesture reuse across the film and flags
convergence so it can be fixed instead of hoped away.

Everything here is deliberately physical and filmable -- things a video
model can literally stage and a still can literally show.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from .llm import logger

# (emotion) -> {"contained": [...], "expressive": [...]}
# 6-10 options each; the menu injected per scene samples from the emotions
# the scene actually carries.
BEHAVIOR_GRAMMAR: Dict[str, Dict[str, List[str]]] = {
    "anger": {
        "contained": [
            "goes very still and sets an object down with exaggerated care",
            "speaks slower and quieter while aligning things on a surface "
            "into perfect rows",
            "folds a cloth or paper into smaller and smaller squares",
            "smiles at the wrong moment, eyes not joining in",
            "closes a drawer/door with controlled softness that lands "
            "louder than a slam",
            "wipes an already-clean surface in tight circles",
        ],
        "expressive": [
            "sweeps a small object off the table and immediately regrets it",
            "paces a short line, turning too sharply at each end",
            "points, catches themselves, converts the point into running "
            "a hand through their hair",
            "laughs one hard humorless syllable and turns away",
            "yanks at their own collar or sleeve like the clothes offend",
            "kicks something harmless and then has to hop, undercutting "
            "themselves",
        ],
    },
    "grief": {
        "contained": [
            "keeps working -- busy hands doing an ordinary task with "
            "extraordinary focus",
            "cooks or pours or tidies for people who haven't asked",
            "folds the absent person's belongings, smooths them twice",
            "stands in a doorway a beat too long before entering",
            "sets an extra place / pours an extra cup out of habit, then "
            "stares at it",
            "answers questions one beat late, from a long way off",
        ],
        "expressive": [
            "sits down wherever they are -- floor, stairs, curb -- as if "
            "the legs were unplugged",
            "clutches the relic or garment to their chest",
            "presses the heels of both hands into their eyes",
            "shakes their head 'no' at something no one said",
            "reaches for a phone/photo, stops halfway, drops the hand",
            "makes a sound that starts as a laugh and lands as the other "
            "thing",
        ],
    },
    "joy": {
        "contained": [
            "a slow-blooming smile fought down and losing",
            "presses lips together and looks away to hide the grin",
            "touches the good news object twice, checking it's real",
            "walks with a new lightness, almost a bounce, denying it",
            "hums or taps their idle rhythm brighter and faster than usual",
            "gives away something small -- a coin, an apple -- for no reason",
        ],
        "expressive": [
            "spins once with arms out, catches someone watching, does it "
            "again anyway",
            "picks someone up or claps both their shoulders",
            "laughs with the whole torso, head back",
            "drums a victory beat on the nearest surface",
            "runs when walking would do",
            "repeats the good sentence out loud to make it truer",
        ],
    },
    "fear": {
        "contained": [
            "goes economical -- small steps, elbows in, taking up less "
            "space",
            "checks the exit with a flick of the eyes, twice",
            "swallows before answering; the voice comes out one size small",
            "hides shaking hands by holding something with both of them",
            "keys or coins in a pocket go quiet -- the hand has clamped "
            "them still",
            "over-nods at reassurance nobody believes",
        ],
        "expressive": [
            "backs into furniture because the eyes won't leave the threat",
            "breath audible -- shallow, high, counted",
            "grabs the nearest arm without looking at whose it is",
            "talks too fast, sentences shedding their endings",
            "startles hugely at a harmless sound and can't laugh it off",
            "freezes mid-gesture like a stopped film frame",
        ],
    },
    "shame": {
        "contained": [
            "adjusts clothing that doesn't need adjusting",
            "over-tidies -- stacking, straightening, anything but eye "
            "contact",
            "agrees too quickly to everything",
            "positions an object between themselves and the other person",
            "studies their own hands like unfamiliar tools",
            "leaves a room by degrees -- half-steps backward while still "
            "talking",
        ],
        "expressive": [
            "covers face with one hand and speaks through the fingers",
            "apologizes to the furniture they bump, not the person",
            "offers restitution with both hands, physically pressing it "
            "on the other person",
            "sits abruptly and puts their head down on the nearest surface",
            "confesses to the floor, voice climbing as it comes loose",
        ],
    },
    "love": {
        "contained": [
            "orients toward the other person like a compass needle, "
            "denying it when caught",
            "keeps their drink filled / chair pulled / path clear without "
            "being asked",
            "remembers a small detail out loud and shrugs it off as "
            "nothing",
            "touches the place the other person just touched",
            "listens with the whole body leaned in, one degree past "
            "casual",
            "laughs half a beat early at their jokes",
        ],
        "expressive": [
            "closes distance in the middle of the other's sentence",
            "fixes the other's collar/hair/strap -- hands lingering",
            "gives them the last / the best / the warmest of whatever "
            "there is",
            "says the plain thing out loud and watches it land",
            "takes their hand mid-task, making the task impossible",
        ],
    },
    "wonder": {
        "contained": [
            "goes silent mid-sentence, mouth still open around the last "
            "word",
            "reaches toward it and stops short of touching",
            "removes hat / glasses / gloves as if entering a church",
            "checks the reaction of someone beside them: are you seeing "
            "this",
            "steps closer one careful foot at a time",
        ],
        "expressive": [
            "laughs in disbelief and covers their mouth",
            "turns a full slow circle to take it all in",
            "calls the others over without taking their eyes off it",
            "touches it and snatches the hand back, then touches it again",
            "narrates what they're seeing to no one",
        ],
    },
    "resolve": {
        "contained": [
            "rolls sleeves / ties hair / squares the jacket -- the small "
            "ritual of getting to work",
            "sets the relic down deliberately: not carrying it into this",
            "exhales once, long, and the shoulders drop into place",
            "checks each tool or strap in a practiced order",
            "nods once, to themselves, and moves",
        ],
        "expressive": [
            "stands so fast the chair complains",
            "crosses the room like the floor owes them distance",
            "says the plan out loud in short flat sentences, pointing at "
            "each person it touches",
            "shakes out both hands like a swimmer on the blocks",
        ],
    },
}

# Aliases so scene emotions ("dread", "heartbreak", "triumph"...) resolve
# to a bank.
_EMOTION_ALIASES = {
    "anger": "anger", "rage": "anger", "fury": "anger",
    "frustration": "anger", "resentment": "anger",
    "grief": "grief", "loss": "grief", "sorrow": "grief",
    "sadness": "grief", "heartbreak": "grief", "mourning": "grief",
    "joy": "joy", "happiness": "joy", "delight": "joy", "triumph": "joy",
    "relief": "joy", "hope": "joy", "excitement": "joy",
    "fear": "fear", "dread": "fear", "terror": "fear", "anxiety": "fear",
    "panic": "fear", "unease": "fear", "suspense": "fear",
    "shame": "shame", "guilt": "shame", "embarrassment": "shame",
    "regret": "shame", "humiliation": "shame",
    "love": "love", "tenderness": "love", "longing": "love",
    "affection": "love", "desire": "love", "attraction": "love",
    "wonder": "wonder", "awe": "wonder", "curiosity": "wonder",
    "amazement": "wonder", "discovery": "wonder",
    "resolve": "resolve", "determination": "resolve", "courage": "resolve",
    "defiance": "resolve",
}


def _resolve_emotions(raw: List[str]) -> List[str]:
    out = []
    for r in raw:
        text = str(r or "").lower()
        for word, key in _EMOTION_ALIASES.items():
            if word in text and key not in out:
                out.append(key)
    return out


def behavior_menu_block(scene_emotion: str,
                        extra_emotions: List[str] = None,
                        max_emotions: int = 3) -> str:
    """The per-scene menu injected into scene authoring: for each emotion
    the scene carries, both expression modes' options (a character's own
    emotion_style decides whether they act from the CONTAINED or the
    EXPRESSIVE column). Framed as a palette, never a checklist."""
    keys = _resolve_emotions([scene_emotion] + list(extra_emotions or []))
    if not keys:
        keys = ["resolve"]
    keys = keys[:max_emotions]
    lines = []
    for k in keys:
        bank = BEHAVIOR_GRAMMAR.get(k)
        if not bank:
            continue
        lines.append(f"{k.upper()} -- contained bodies: "
                     + "; ".join(bank["contained"][:5]))
        lines.append(f"{k.upper()} -- expressive bodies: "
                     + "; ".join(bank["expressive"][:5]))
    if not lines:
        return ""
    return ("BEHAVIOR PALETTE (concrete stageable options for this scene's "
            "emotions -- a menu, not a checklist: pick or invent IN THIS "
            "SPIRIT, matched to each character's own emotion style "
            "[suppressors act from the contained column, exploders from "
            "the expressive], and NEVER default to stock gestures like a "
            "clenched jaw, white knuckles, or a single tear):\n"
            + "\n".join(lines))


# ---------------------------------------------------------------------------
# Deterministic gesture-repetition audit (the enforcement half)
# ---------------------------------------------------------------------------
_STOCK_GESTURES = [
    "clench", "clenches", "clenched jaw", "grips", "gripping", "gripped",
    "white knuckle", "white-knuckle", "single tear", "a tear rolls",
    "takes a deep breath", "deep breath", "sighs", "runs a hand through",
    "shakes his head", "shakes her head", "nods slowly", "swallows hard",
    "eyes widen", "narrows his eyes", "narrows her eyes", "fists",
]

_STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "at",
              "with", "his", "her", "their", "its", "as", "into", "onto",
              "from", "then", "while", "over", "under", "for", "by"}


def _content_trigrams(text: str) -> List[str]:
    words = [w for w in re.findall(r"[a-z']+", (text or "").lower())
             if w not in _STOPWORDS and len(w) > 2]
    return [" ".join(words[i:i + 3]) for i in range(len(words) - 2)]


def behavior_repetition_report(scenes: List[Dict],
                               cfg: Dict = None) -> str:
    """Walk every shot's staged action and count (a) stock-gesture hits and
    (b) repeated content trigrams across DIFFERENT scenes (same-scene
    repetition can be deliberate rhythm; cross-scene repetition is
    convergence). Purely deterministic -- no LLM call. Returns a short
    report string ('' when clean); the caller logs it and lands it in the
    plan for the table read."""
    max_reuse = int((cfg or {}).get("behavior_repetition_max", 2))
    stock_hits = Counter()
    tri_scenes: Dict[str, set] = {}
    for sc in scenes:
        for sh in sc.get("shots", []):
            action = str(sh.get("action", "") or "").lower()
            for g in _STOCK_GESTURES:
                if g in action:
                    stock_hits[g] += 1
            for t in set(_content_trigrams(action)):
                tri_scenes.setdefault(t, set()).add(sc.get("scene_id"))
    lines = []
    stocked = [(g, n) for g, n in stock_hits.most_common() if n > max_reuse]
    if stocked:
        lines.append("Stock gestures over budget (rewrite from the "
                     "behavior palette or the character's idle behavior): "
                     + ", ".join(f"'{g}' x{n}" for g, n in stocked[:8]))
    converged = sorted(((t, s) for t, s in tri_scenes.items()
                        if len(s) > max_reuse),
                       key=lambda kv: -len(kv[1]))
    if converged:
        lines.append("Action phrasing repeated across scenes: "
                     + "; ".join(f"'{t}' in scenes "
                                 f"{sorted(x for x in s if x is not None)}"
                                 for t, s in converged[:6]))
    if not lines:
        return ""
    report = ("BEHAVIOR REPETITION AUDIT (deterministic): the staged "
              "actions are converging -- variety is where authored films "
              "and generated films part ways.\n" + "\n".join(lines))
    logger.warning("[behavior] %s", report.replace("\n", " | ")[:500])
    return report
