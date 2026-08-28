"""
film_maker.registers
====================
The artistic-method engine. Every register is a complete craft playbook:
how STORY is shaped, how DIALOGUE breathes, how PACING cuts, how the CAMERA
behaves at each tension band, and how the SOUNDSCAPE is designed. The same
space-station premise becomes a comedy, a horror film, or a drama purely by
which playbooks are active.

Registers COMPOSE: FILM_CONFIG["registers"] = ["scifi", "comedy"] blends the
sci-fi world method with comedy's timing machinery. The first entry is
PRIMARY (it wins pacing conflicts); later entries layer their methods in.

Every block below is written to be injected verbatim into LLM prompts at the
stage it governs — these are working instructions to a craftsperson, not
mood adjectives.
"""
from __future__ import annotations

from typing import Dict, List

# Baseline tension-band camera grammar (H3's own vocabulary), ported from the
# music pipeline and re-keyed to DRAMATIC tension instead of music energy.
CAMERA_GRAMMAR_BY_TENSION = {
    "low": {
        "feel": "calm, intimate, observational — the camera breathes with the scene",
        "moves": ["Static Shot", "Push In", "slow Pull Out", "gentle Tilt Up",
                  "slow Pan Left", "slow Pan Right", "Pedestal Up"],
        "modifiers": "with small amplitude at slow speed",
    },
    "mid": {
        "feel": "purposeful, flowing, the story gathering momentum",
        "moves": ["Tracking Shot", "Arc Shot", "Truck Left", "Truck Right",
                  "Push In", "Pan Left", "Pan Right", "Tilt Down"],
        "modifiers": "with medium amplitude",
    },
    "high": {
        "feel": "urgent, kinetic, the peak of the sequence",
        "moves": ["Zoom In", "fast Push In", "Arc Shot", "Shake Slightly",
                  "fast Tracking Shot", "Whip Pan Left", "Whip Pan Right",
                  "Roll Clockwise"],
        "modifiers": "with large amplitude at fast speed",
    },
}

REGISTERS: Dict[str, Dict] = {
    # ----------------------------------------------------------------- DRAMA
    "drama": {
        "label": "Drama",
        "story_method": (
            "Build the film on a character whose WANT (what they pursue) and "
            "NEED (the truth they must accept) are in conflict. Every scene "
            "either moves the want forward or presses on the wound that "
            "blocks the need. Escalate cost: each act's choices are harder "
            "than the last, and the climax forces choosing the need at the "
            "want's expense (or refusing it, in tragedy). Plant at least two "
            "concrete objects/images early that recur transformed at the "
            "climax and resolution."
        ),
        "dialogue_method": (
            "Subtext over statement: characters talk AROUND the wound, not "
            "about it. Let the important thing be the one not said; use "
            "interrupted sentences, deflections, and one late scene where "
            "the truth finally lands in plain words and hits like a blow "
            "because it was withheld so long. People answer the question "
            "they wish had been asked."
        ),
        "sound_method": (
            "Grounded, sparse diegetic sound. Silence is an instrument: cut "
            "the room tone almost to nothing in the two or three biggest "
            "emotional beats so a breath or a chair creak carries the "
            "moment. Weather and rooms mirror interior states without "
            "announcing it."
        ),
        "pacing": {"low": 1.15, "mid": 1.0, "high": 0.9},
        "visual_bias": "natural light motivated by real sources, faces held "
                       "long enough to read a thought change on screen",
    },
    # ---------------------------------------------------------------- COMEDY
    "comedy": {
        "label": "Comedy",
        "story_method": (
            "Comedy is structure: a flawed protagonist with an unshakeable "
            "wrong belief collides with escalating situations that punish "
            "it. Use the rule of three (setup, reinforce, subvert) at every "
            "scale — within a line, within a scene, across the film. Every "
            "scene ends on a button (a topper, a reversal, or a cut on the "
            "worst possible moment). Plant running gags that escalate on "
            "each return and pay the biggest one off inside the climax "
            "itself, where the joke and the emotional resolution are the "
            "same beat."
        ),
        "dialogue_method": (
            "Comic timing is rhythm: setups are efficient, punchlines land "
            "on the LAST word of the line, and the funniest word goes at "
            "the end. Write straight-man/wildcard dynamics; let one "
            "character take everything literally. Never explain a joke, "
            "never have characters laugh at their own lines. After a "
            "punchline, give the next shot a silent reaction beat — the "
            "laugh lives in the reaction."
        ),
        "sound_method": (
            "Sound IS a comedian: a too-long silence after a disaster, one "
            "small pathetic diegetic sound (a single falling hubcap, a "
            "deflating balloon) as the button on chaos, cheerful ambient "
            "sound continuing obliviously under misfortune. Mistimed or "
            "inappropriate room tone is a joke."
        ),
        "pacing": {"low": 0.85, "mid": 0.8, "high": 0.75},
        "post_peak_hold": True,   # hold an extra beat AFTER a punchline shot
        "visual_bias": "clean wide framings that let physical comedy read "
                       "full-body; symmetry played straight until broken",
    },
    # ---------------------------------------------------------------- HORROR
    "horror": {
        "label": "Horror",
        "story_method": (
            "Dread before shock: the audience must know the rules of the "
            "threat before it fully appears, and the protagonist must break "
            "one knowingly. Escalate by proximity — first traces, then "
            "glimpses, then presence. Grant one false relief per act and "
            "cut it short. The final confrontation costs something that "
            "cannot be restored; do not fully explain the threat."
        ),
        "dialogue_method": (
            "Under-write it. Fear shortens sentences and strips politeness. "
            "Characters narrate what they see only when it is worse than "
            "what the camera shows. Whispered lines, unanswered calls, one "
            "character insisting everything is fine long past the point of "
            "credibility."
        ),
        "sound_method": (
            "The soundscape is the monster's first body: design one "
            "signature sound for the threat (a specific scrape, a wet "
            "click, a wrong-pitched hum) and plant it faintly two scenes "
            "before its source is seen. Drop ambient sound to dead silence "
            "before every scare; let the loudest moment of a scene be "
            "something small and organic. Off-screen sound does more work "
            "than on-screen sound."
        ),
        "pacing": {"low": 1.35, "mid": 1.1, "high": 0.7},
        "visual_bias": "negative space the eye must search, doorways and "
                       "thresholds, light sources that leave the corners "
                       "unresolved, slow push-ins on nothing",
    },
    # -------------------------------------------------------------- THRILLER
    "thriller": {
        "label": "Thriller",
        "story_method": (
            "Suspense is information asymmetry: let the audience know the "
            "bomb is under the table before the characters do (Hitchcock's "
            "rule) at least twice per act. Deadlines and ticking clocks "
            "escalate; every apparent victory reveals a deeper layer of the "
            "problem. The protagonist's competence is real but always one "
            "step behind until the turn."
        ),
        "dialogue_method": (
            "Lines carry double duty — surface transaction plus concealed "
            "agenda. Characters test each other with questions they already "
            "know the answers to. Keep exposition adversarial: information "
            "is extracted, traded, or overheard, never volunteered."
        ),
        "sound_method": (
            "Rhythmic diegetic tension: clocks, footsteps, engines, "
            "breathing — real sounds that behave like a score. Cut ambient "
            "beds abruptly on reveals. Overlap the next scene's sound a "
            "half-beat early to keep the pulse moving through cuts."
        ),
        "pacing": {"low": 1.0, "mid": 0.85, "high": 0.7},
        "visual_bias": "compressed telephoto frames, obstructed vantage "
                       "points, reflections and surveillance angles",
    },
    # --------------------------------------------------------------- ROMANCE
    "romance": {
        "label": "Romance",
        "story_method": (
            "Two people who each hold the missing piece of the other's "
            "unfinished self, kept apart by a believable internal flaw, not "
            "a contrived misunderstanding. Build in touches that almost "
            "happen. The midpoint gives a taste of the life they could "
            "have; the crisis takes it away through the flaw; the climax "
            "requires a public or costly act that proves the flaw broken."
        ),
        "dialogue_method": (
            "Chemistry is specificity: they notice details about each other "
            "no one else does, and tease with precision. Banter has an "
            "undertow — every joke is also a question. The word 'love' is "
            "spent exactly once, where it costs the most."
        ),
        "sound_method": (
            "Warm close-perspective sound in their scenes together — the "
            "world's ambience recedes when they focus on each other, and "
            "floods back when the moment breaks. Small intimate sounds "
            "(fabric, rain on glass, a held breath) at high presence."
        ),
        "pacing": {"low": 1.2, "mid": 1.0, "high": 0.95},
        "visual_bias": "shrinking distance across the film: singles become "
                       "shared frames become close two-shots; golden and "
                       "blue hour, faces lit by practicals",
    },
    # ----------------------------------------------------------------- SCIFI
    "scifi": {
        "label": "Science Fiction",
        "story_method": (
            "One clear speculative premise, rigorously followed: change one "
            "thing about the world and derive everything else honestly from "
            "it. The technology or phenomenon must externalize the "
            "protagonist's inner question — the film argues an idea through "
            "images. Deliver at least two moments of genuine scale and awe, "
            "each attached to an emotional turn, never as empty spectacle."
        ),
        "dialogue_method": (
            "World-building through implication: characters treat the "
            "extraordinary as routine and the routine as precious. No "
            "exposition dumps — jargon is used casually and explained only "
            "by consequence. Let one character speak for wonder and one "
            "for cost."
        ),
        "sound_method": (
            "Invent the world's sonic signature: engines, interfaces, and "
            "atmospheres get specific, consistent voices reused across "
            "scenes so the world feels built. Contrast vast exterior "
            "silences or thin atmospheres against dense interior hums. "
            "Alien or advanced elements sound organic-plus-wrong, not "
            "generic bleeps."
        ),
        "pacing": {"low": 1.25, "mid": 1.0, "high": 0.85},
        "visual_bias": "scale contrast — tiny human figures against vast "
                       "structures; hard single-source light in space, "
                       "volumetric atmosphere on surfaces",
    },
    # ------------------------------------------------------------------ NOIR
    "noir": {
        "label": "Noir",
        "story_method": (
            "A compromised protagonist takes one morally gray job that "
            "unravels into a web implicating everyone, including them. "
            "Every ally has an angle; every answer opens a worse question. "
            "The ending resolves the mystery but not the rot — the "
            "protagonist wins the truth and loses something they wanted "
            "more."
        ),
        "dialogue_method": (
            "Hard-edged economy: wit as armor, threat delivered politely, "
            "confession delivered like an accusation. Optional sparse "
            "voice-over from the protagonist, past tense, wry, unreliable "
            "at the edges."
        ),
        "sound_method": (
            "Night city as orchestra: rain, neon buzz, distant trains, a "
            "phone ringing in another room. Interiors are close and dead; "
            "exteriors wet and alive. Footsteps on hard surfaces carry "
            "menace."
        ),
        "pacing": {"low": 1.15, "mid": 1.0, "high": 0.85},
        "visual_bias": "chiaroscuro, blinds-slatted light, wet streets "
                       "doubling every light source, smoke in beams",
    },
    # ------------------------------------------------------------- ADVENTURE
    "adventure": {
        "label": "Adventure",
        "story_method": (
            "A concrete goal across escalating terrain, with set pieces "
            "that each demand a DIFFERENT competence (wits, nerve, "
            "strength, trust). The map itself is a character — geography "
            "creates the obstacles. Companions carry the theme; the "
            "treasure is redefined by the journey's cost."
        ),
        "dialogue_method": (
            "Momentum talk: plans made mid-motion, banter under pressure, "
            "camaraderie built through competence noticed aloud. Short "
            "lines during action, breath and humor between set pieces."
        ),
        "sound_method": (
            "Big honest physical sound: environments at full presence "
            "(wind, water, stone, machinery), impacts with weight. Each "
            "new location announces itself sonically in its first shot."
        ),
        "pacing": {"low": 1.0, "mid": 0.9, "high": 0.75},
        "visual_bias": "epic wides that establish geography before action "
                       "enters it, motivated aerials, horizon lines",
    },
    # --------------------------------------------------------------- TRAGEDY
    "tragedy": {
        "label": "Tragedy",
        "story_method": (
            "The protagonist's greatest strength is the flaw; the audience "
            "must see the exit door the character cannot. Build inevitability "
            "with rhymed scenes — late scenes that visually echo early ones "
            "with the meaning inverted. Grant one true moment of grace "
            "before the fall, and let the ending land with dignity, not "
            "punishment."
        ),
        "dialogue_method": (
            "Dramatic irony in the mouth: characters make promises the "
            "audience knows will break, use words whose second meaning "
            "they cannot yet hear. In the final movement, strip the "
            "eloquence to plain, short truth."
        ),
        "sound_method": (
            "A recurring diegetic sound motif (bells, tide, a machine) "
            "that returns at each turn of the wheel, slightly transformed "
            "— slower, lower, or finally silent."
        ),
        "pacing": {"low": 1.3, "mid": 1.05, "high": 0.95},
        "visual_bias": "formal, stable compositions that tighten and "
                       "darken as choices close; empty frames after exits",
    },
    # ---------------------------------------------------------------- WONDER
    "wonder": {
        "label": "Wonder / Family",
        "story_method": (
            "See the world at child height: an ordinary life cracked open "
            "by something impossible that responds to sincerity. Stakes "
            "are emotional, not violent — losing the friend, the belief, "
            "the home. Adults who don't see it aren't villains; they're "
            "the future the protagonist is resisting. Earn the tears with "
            "joy first."
        ),
        "dialogue_method": (
            "Direct, sincere, lightly funny. Children speak with concrete "
            "logic that cuts through adult evasion. The impossible thing "
            "communicates in its own consistent way (sounds, light, "
            "gesture), never plain speech."
        ),
        "sound_method": (
            "The impossible thing gets a beautiful signature sound "
            "(glass-chime, whale-warmth, wind-through-strings) developed "
            "across the film. Everyday sounds rendered lovingly — cereal, "
            "bicycles, screen doors — so the ordinary world is worth "
            "saving."
        ),
        "pacing": {"low": 1.1, "mid": 0.95, "high": 0.85},
        "visual_bias": "low child-height camera, warm practicals, bicycle-"
                       "speed tracking, light as a character",
    },
    # ------------------------------------------------------------- EXPLAINER
    "explainer": {
        "label": "Explainer / Educational",
        "story_method": (
            "Teach through story, never lecture: a character with relatable "
            "stakes NEEDS the concept to solve a real problem. Open with a "
            "curiosity gap (a concrete question the audience can't yet "
            "answer but immediately wants to). Dramatize the INTUITIVE "
            "WRONG MODEL first and let it visibly fail -- refuting the "
            "misconception is how understanding sticks. Then build the true "
            "model through ONE core analogy carried consistently across the "
            "whole film (never mix vehicles), concrete before abstract, ONE "
            "new idea per scene, each idea earned by the story's events. "
            "The eureka is the climax; the resolution shows the concept "
            "USED to win something that matters to the character."
        ),
        "dialogue_method": (
            "The learner asks the exact questions the audience is thinking, "
            "half a beat before they think them. The guide answers with "
            "images and demonstrations, not definitions -- jargon appears "
            "only AFTER its meaning has been seen, then gets named almost "
            "casually. Wrong guesses are honored, never mocked; a wrong "
            "guess that's wrong in an interesting way teaches more than a "
            "right answer. Delight is allowed: the guide's own wonder at "
            "the idea is contagious."
        ),
        "sound_method": (
            "Give the CONCEPT its own signature sound that develops as "
            "understanding grows -- muddy and fragmented while the wrong "
            "model holds, resolving into clarity at the click. Demonstration "
            "moments get intimate, close-perspective foley (the audience "
            "hears the mechanism). Silence right before the click; the "
            "world's sound floods back with the release."
        ),
        "pacing": {"low": 1.15, "mid": 1.0, "high": 0.9},
        "visual_bias": "demonstrations staged in the physical world of the "
                       "story (never floating diagrams); scale shifts that "
                       "make the invisible visible; the core analogy's "
                       "imagery recurring and evolving",
    },
    # ---------------------------------------------------------------- MATURE
    "mature_sensual": {
        "label": "Mature / Sensual",
        "story_method": (
            "Desire as dramatic engine: attraction complicates loyalty, "
            "power, or grief, and every intimate beat changes the story "
            "state — someone risks, reveals, or decides. Sensuality is "
            "characterization: HOW each person approaches closeness "
            "exposes their wound. Never gratuitous; every charged scene "
            "must be replaceable by nothing else in the plot."
        ),
        "dialogue_method": (
            "Tension in restraint: charged pauses, double meanings, the "
            "gap between what is offered and what is asked. Low volume, "
            "close distance. Consent and desire read in the lines, not "
            "around them."
        ),
        "sound_method": (
            "Extreme close sound perspective: breath, fabric, skin, a "
            "swallowed word — mixed above the room. The world outside "
            "the two people drops away and returns as a cold slap when "
            "reality intrudes."
        ),
        "pacing": {"low": 1.35, "mid": 1.1, "high": 1.0},
        "visual_bias": "shallow focus, warm low-key practicals, partial "
                       "framings that suggest more than they show, "
                       "lingering holds",
        "requires_rating": "mature",
    },
}

REGISTER_ALIASES = {
    "comedy": "comedy", "funny": "comedy", "humor": "comedy",
    "drama": "drama", "dramatic": "drama",
    "horror": "horror", "scary": "horror", "dread": "horror",
    "thriller": "thriller", "suspense": "thriller", "mystery": "noir",
    "romance": "romance", "romantic": "romance", "love": "romance",
    "scifi": "scifi", "sci-fi": "scifi", "science fiction": "scifi",
    "noir": "noir", "detective": "noir",
    "adventure": "adventure", "action": "adventure",
    "tragedy": "tragedy",
    "wonder": "wonder", "family": "wonder", "fantasy": "wonder",
    "explainer": "explainer", "educational": "explainer",
    "education": "explainer", "documentary": "explainer",
    "science": "explainer",
    "mature": "mature_sensual", "sensual": "mature_sensual",
    "erotic": "mature_sensual",
}


def resolve_registers(names: List[str], content_rating: str = "teen") -> List[str]:
    """Normalize user-supplied register names, dropping any gated by rating."""
    out = []
    for n in names or ["drama"]:
        key = REGISTER_ALIASES.get(str(n).strip().lower())
        if key is None:
            continue
        gate = REGISTERS[key].get("requires_rating")
        if gate and content_rating != gate:
            continue
        if key not in out:
            out.append(key)
    return out or ["drama"]


def _blend(names: List[str], field: str, header: str) -> str:
    lines = [header]
    for i, n in enumerate(names):
        r = REGISTERS[n]
        role = "PRIMARY" if i == 0 else "LAYERED"
        lines.append(f"[{role} — {r['label']}] {r[field]}")
    return "\n".join(lines)


def story_method_block(names: List[str]) -> str:
    return _blend(names, "story_method",
                  "STORY CRAFT (obey these playbooks; the PRIMARY register "
                  "wins any conflict):")


def dialogue_method_block(names: List[str]) -> str:
    return _blend(names, "dialogue_method",
                  "DIALOGUE CRAFT (write every line under these methods):")


def sound_method_block(names: List[str]) -> str:
    return _blend(names, "sound_method",
                  "SOUND DESIGN CRAFT (design every overall_soundscape under "
                  "these methods; sound is a storytelling instrument, not "
                  "wallpaper):")


def visual_bias_block(names: List[str]) -> str:
    return _blend(names, "visual_bias",
                  "VISUAL BIAS (let this inflect composition and light):")


def pacing_multiplier(names: List[str], band: str) -> float:
    """Primary register's pacing curve scales the base cut-length targets."""
    r = REGISTERS[names[0]]
    return float(r.get("pacing", {}).get(band, 1.0))


def post_peak_hold(names: List[str]) -> bool:
    return any(REGISTERS[n].get("post_peak_hold") for n in names)


def camera_grammar_block(band: str, names: List[str]) -> str:
    g = CAMERA_GRAMMAR_BY_TENSION.get(band, CAMERA_GRAMMAR_BY_TENSION["mid"])
    moves = ", ".join(g["moves"])
    return (f"CAMERA REGISTER for this shot (tension = {band.upper()}: "
            f"{g['feel']}):\n"
            f"- Choose ONE dominant camera move from: {moves}.\n"
            f"- Default modifiers unless the moment calls otherwise: "
            f"{g['modifiers']}.\n"
            f"- {visual_bias_block(names)}")


def content_rating_block(rating: str) -> str:
    if rating == "all_ages":
        return ("CONTENT: all-ages. No graphic violence, no sexual content, "
                "no profanity; peril and emotion are fine.")
    if rating == "mature":
        return ("CONTENT: mature. Adult themes, sensuality, intensity, and "
                "consequence are available where the story earns them; use "
                "them with intent, never as filler.")
    return ("CONTENT: teen-appropriate. Real stakes and emotion; no graphic "
            "content, mild language at most.")
