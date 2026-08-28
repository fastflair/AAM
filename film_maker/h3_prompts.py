"""
film_maker.h3_prompts
=====================
Assembles each shot's final MiniMax H3 prompt in the required 3-part
structure:

    [Shot 1: <style/framing>. <visual block with dialogue inline>.]
    overall_soundscape: <diegetic sound design>
    non_diegetic_music: N/A

Dialogue rules implemented exactly per the H3 prompting guide:
  * Speaker IDs (S1), (S2)... assigned chronologically by FIRST SPEECH
    within THIS shot (each shot is its own generation; cross-shot voice
    consistency comes from the chained audio reference, not the ID).
  * Only the language tag + exact spoken words go inside <d>...</d>; all
    visual description stays outside the tag.
  * Delivery cues are written as natural clauses around the tag ("(S1) says
    through gritted teeth: <d>[English] ...</d>").
  * NARRATOR lines use the OFFICIAL H3 voiceover syntax: the exact phrase
    "says in an off-screen voiceover" in the lead, the designed narrator
    voice texture on the first narrated line (so H3 synthesizes a distinct
    off-screen voice instead of borrowing an on-screen character's), and
    -- immediately AFTER the <d> block -- an explicit clause that every
    visible character's lips remain completely closed. Both halves are
    required: without the trailing lips-closed clause H3 reliably animates
    an on-screen mouth to the narration.
  * The reference-image anchor sentence opens the visual block so the still
    locks subject/setting/style for the whole take.
  * A hard word-budget pass keeps every assembled prompt under
    wgp_prompt_max_words, trimming style tail first and never touching a
    <d> tag.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .llm import clean_text, logger


def _speaker_ids_for_shot(shot: Dict) -> Dict[str, str]:
    """Chronological (S1), (S2)... by first speech within this shot."""
    ids: Dict[str, str] = {}
    n = 0
    for ln in shot.get("lines", []):
        sp = (ln.get("speaker") or "").strip()
        if sp and sp not in ids:
            n += 1
            ids[sp] = f"S{n}"
    return ids


_ABBREV = {"Dr.": "Doctor", "Mr.": "Mister", "Mrs.": "Missus",
           "Ms.": "Miz", "St.": "Saint", "Lt.": "Lieutenant",
           "Sgt.": "Sergeant", "Capt.": "Captain", "Prof.": "Professor",
           "Jr.": "Junior", "Sr.": "Senior", "vs.": "versus", "etc.": "and so on",
           "No.": "Number", "&": "and", "%": " percent"}
_SMALL_NUMS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
               "eight", "nine", "ten", "eleven", "twelve", "thirteen",
               "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
               "nineteen", "twenty"]


def make_speakable(text: str) -> str:
    """Normalize dialogue so H3 speaks it cleanly: expand common
    abbreviations and small standalone numerals into words (a lip-synced
    voice reading '3' or 'Dr.' is a coin flip; 'three' and 'Doctor' are
    not). Larger numbers are left for the LLM rule that already asks for
    words, but digits 0-20 are guaranteed here."""
    if not text:
        return text
    for k, v in _ABBREV.items():
        text = text.replace(k, v)

    def _num(m):
        n = int(m.group(0))
        return _SMALL_NUMS[n] if 0 <= n <= 20 else m.group(0)

    text = re.sub(r"(?<![\w.])\d{1,2}(?!\w)(?!\.\d)", _num, text)
    return re.sub(r"\s{2,}", " ", text).strip()


# The official H3 voiceover syntax has TWO required parts (per MiniMax's
# prompting guidance): the EXACT phrase "says in an off-screen voiceover"
# in the lead, and -- immediately AFTER the <d> block -- an explicit
# statement that visible lips stay closed. Omitting the trailing
# lips-closed clause makes H3 animate an on-screen mouth to the
# narration (a visible character appears to speak the narrator's line).
VOICEOVER_LIPS_CLOSED = ("while every visible character's lips remain "
                         "completely closed")


def _dialogue_clauses(shot: Dict, language: str,
                      screen_tags: Dict[str, str] = None,
                      narrator_voice: str = "") -> List[str]:
    """Render each line as '(S1) says <delivery>: <d>[Lang] words</d>'.
    The FIRST line from each speaker carries their screen_tag so H3 binds
    the (S#) voice to the correct on-screen body in multi-character shots;
    later lines stay short. NARRATOR lines use the official H3 voiceover
    syntax ("says in an off-screen voiceover" + a lips-closed clause right
    after the <d> block), with the designed narrator voice texture on the
    first narrated line so H3 synthesizes a DISTINCT off-screen voice
    instead of borrowing an on-screen character's."""
    ids = _speaker_ids_for_shot(shot)
    screen_tags = screen_tags or {}
    tagged = set()
    clauses = []
    for ln in shot.get("lines", []):
        sp = (ln.get("speaker") or "").strip()
        text = make_speakable((ln.get("text") or "").strip())
        if not sp or not text:
            continue
        sid = ids.get(sp, "S1")
        line_lang = (ln.get("language") or "").strip() or language
        delivery = (ln.get("delivery") or "").strip().rstrip(".")
        text = re.sub(r'[<>\[\]]', "", text)  # keep tag syntax unambiguous
        if sp.upper() == "NARRATOR":
            if sp not in tagged:
                voice_bit = f", {narrator_voice}," if narrator_voice else ""
                lead = (f"An unseen narrator{voice_bit} ({sid}) says in "
                        f"an off-screen voiceover")
                tagged.add(sp)
            else:
                lead = (f"The unseen narrator ({sid}) says in an "
                        f"off-screen voiceover")
            if delivery:
                lead += f", {delivery}"
            clauses.append(f"{lead}: <d>[{line_lang}] {text}</d> "
                           f"{VOICEOVER_LIPS_CLOSED}")
        else:
            tag = screen_tags.get(sp, "")
            if tag and sp not in tagged and len(ids) > 1:
                lead = f"{sp}, {tag}, ({sid}) says"
                tagged.add(sp)
            else:
                lead = f"{sp} ({sid}) says"
            if delivery:
                if (delivery.split()[0].endswith("ly")
                        or delivery.startswith(("through", "with", "in ",
                                                "under", "over"))):
                    lead += f" {delivery}"
                else:
                    lead += f", {delivery}"
            clauses.append(f"{lead}: <d>[{line_lang}] {text}</d>")
    return clauses


def _tag_speakers_in_visual(visual: str, shot: Dict) -> str:
    """First mention of each speaking character in the visual text gets its
    (S#) tag attached, tying the voice to the on-screen body."""
    ids = _speaker_ids_for_shot(shot)
    for name, sid in ids.items():
        if name.upper() == "NARRATOR":
            continue
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        if pattern.search(visual) and f"({sid})" not in visual:
            visual = pattern.sub(f"{name} ({sid})", visual, count=1)
    return visual


ANCHOR = ("The subject, setting, and visual style shown in the reference "
          "image hold for this entire continuous take, with no reset or "
          "restart partway through.")

ANCHOR_CHAINED = (
    "The reference image is the exact moment this continues from: every "
    "costume detail and damage, held object, prop position, lighting, and "
    "weather in it carries over unchanged, while the framing becomes the "
    "shot described here. Nothing resets. CRITICAL: do not restart, repeat, "
    "or re-loop any action already completed in the reference image -- the "
    "action must advance FORWARD from exactly this moment, never back to "
    "its beginning.")

ANCHOR_IDENTITY = (
    "An additional reference image shows this scene's opening frame: every "
    "character's face, hairstyle, build, and costume must keep matching it "
    "exactly -- identity never drifts from that appearance.")


def assemble_h3_prompt(shot: Dict, scene: Dict, story: Dict,
                       style_medium: str, cfg: Dict,
                       chained: bool = False,
                       overrides: Optional[Dict] = None,
                       phase_note: str = "",
                       identity_anchor: bool = False) -> str:
    """Build the full 3-part H3 prompt for one shot. `chained` marks a shot
    (or chain segment) whose reference image is a previous rendered frame
    rather than a synthesized still. `overrides` optionally substitutes
    action/motion/lines (used to render one PHASE of a long take as its own
    segment, each with distinct progressing content, instead of repeating
    the whole shot's description on every segment). `phase_note` is an
    optional short parenthetical for debugging (not shown to the model's
    semantics, kept out of the prompt itself)."""
    language = cfg.get("language", "English")
    scale = shot.get("shot_scale", "medium shot")
    angle = shot.get("camera_angle", "eye-level")
    ov = overrides or {}

    # --- visual block -------------------------------------------------------
    subject_action = clean_text(ov.get("action", shot.get("action", "")))
    motion = clean_text(ov.get("motion", shot.get("motion", "")))
    visual_bits = [ANCHOR_CHAINED if chained else ANCHOR]
    if identity_anchor and chained:
        visual_bits.append(ANCHOR_IDENTITY)
    carry = clean_text(shot.get("carry_state", ""))
    if carry:
        visual_bits.append(f"Continuity that must hold in frame: {carry}")
    visual_bits.append(subject_action)
    screen_tags = {c["name"]: c.get("screen_tag", "")
                   for c in (story.get("characters") or [])}
    narrator_voice = clean_text(
        (story.get("narrator") or {}).get("voice_texture", ""))
    dial_shot = dict(shot)
    if "lines" in ov:
        dial_shot["lines"] = ov["lines"]
    dial = _dialogue_clauses(dial_shot, language, screen_tags,
                             narrator_voice=narrator_voice)
    if dial:
        visual_bits.append(" ".join(dial))
    if motion:
        visual_bits.append(motion)
    visual = " ".join(b.strip().rstrip(".") + "." for b in visual_bits if b.strip())
    visual = _tag_speakers_in_visual(visual, shot)

    header = f"Shot 1: {style_medium}, cinematic. A {scale} at {angle}."
    block = f"[{header} {visual}]"

    # --- soundscape ---------------------------------------------------------
    sound = clean_text(shot.get("soundscape", "")) or \
        "a natural ambient bed true to the setting"

    prompt = (f"{block}\n"
              f"overall_soundscape: {sound}\n"
              f"non_diegetic_music: N/A")
    return _enforce_word_budget(prompt,
                                int(cfg.get("wgp_prompt_max_words", 350)))


def _enforce_word_budget(prompt: str, max_words: int) -> str:
    """Trim to the H3 word ceiling without ever cutting inside a <d> tag or
    dropping the soundscape/music lines. Strategy: shorten the visual block
    from its tail (style/motion prose), keeping dialogue clauses whole."""
    words = prompt.split()
    if len(words) <= max_words:
        return prompt
    lines = prompt.split("\n")
    visual = lines[0]
    tail = "\n".join(lines[1:])
    tail_words = len(tail.split())
    budget = max(40, max_words - tail_words)

    # Protect <d>...</d> spans AND the voiceover lips-closed clauses: split
    # the visual block into segments and trim only unprotected segments
    # from the end. (The lips-closed clause sits right after a </d> at the
    # prompt's tail -- exactly where tail-first trimming would eat it,
    # silently reintroducing the character-mouths-the-narration failure.)
    segments = re.split(
        r"(<d>.*?</d>|\s*" + re.escape(VOICEOVER_LIPS_CLOSED) + r")",
        visual)
    total = sum(len(s.split()) for s in segments)
    i = len(segments) - 1
    while total > budget and i >= 0:
        seg = segments[i]
        if seg.startswith("<d>") or VOICEOVER_LIPS_CLOSED in seg:
            i -= 1
            continue
        seg_words = seg.split()
        excess = total - budget
        if len(seg_words) <= excess:
            total -= len(seg_words)
            segments[i] = ""
        else:
            segments[i] = " ".join(seg_words[:len(seg_words) - excess])
            total = budget
        i -= 1
    visual = "".join(segments).rstrip()
    if not visual.endswith("]"):
        visual = visual.rstrip(". ") + ".]"
    logger.info("  [h3] Prompt trimmed to ~%d words.", total + tail_words)
    return visual + "\n" + tail


def lead_speaker(shot: Dict) -> str:
    """The character whose voice dominates this shot (first non-narrator
    speaker; falls back to the narrator; '' for silent shots). Used for
    voice-chaining audio references."""
    for ln in shot.get("lines", []):
        sp = (ln.get("speaker") or "").strip()
        if sp and sp.upper() != "NARRATOR":
            return sp
    for ln in shot.get("lines", []):
        sp = (ln.get("speaker") or "").strip()
        if sp:
            return sp
    return ""


def speaking_characters(shot: Dict) -> List[str]:
    seen = []
    for ln in shot.get("lines", []):
        sp = (ln.get("speaker") or "").strip()
        if sp and sp not in seen:
            seen.append(sp)
    return seen
