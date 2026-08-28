"""
film_maker.graph
================
The emotional-intelligence CharacterGraph, ported self-contained from the
novel/comic brain. This is what makes dialogue and behavior feel authored:

  * CharacterNode  — archetype, core wound, core longing, defense mechanism,
                     shadow, voice signature, lexical habits, speech rhythm,
                     metaphor pool (the imagery THIS character reaches for).
  * RelationshipEdge — directed A→B: trust/affection/respect/attraction/
                     fear/resentment/envy/empathy (−5..+5), power dynamics,
                     unspoken truths, grievances, secrets kept — plus
                     THEORY OF MIND: what A *believes* B feels, which when
                     wrong produces dramatic irony for free.
  * CharacterGraph — the container plus prompt-block renderers the
                     screenplay engine injects into every dialogue call.

Everything is JSON-serializable so the whole graph lives in plan.json and
survives the plan → edit → produce round trip.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .llm import get_llm, safe_json_dict, safe_json_list, clean_text, logger


@dataclass
class CharacterNode:
    name: str = ""
    archetype: str = ""
    core_wound: str = ""
    core_longing: str = ""
    defense_mechanism: str = ""
    shadow: str = ""
    voice_signature: str = ""
    lexical_habits: List[str] = field(default_factory=list)
    rhythm: str = ""
    metaphor_pool: List[str] = field(default_factory=list)
    humor_style: str = ""
    fear_response: str = ""       # how fear physically shows on them
    joy_response: str = ""        # how joy physically shows on them
    # --- connection layer: what makes an audience recognize themselves ---
    backstory_moment: str = ""    # ONE concrete remembered scene from a
                                  # universal human experience, grounding the
                                  # wound (2-3 sentences, filmable)
    backstory_source: str = ""    # the universal experience it draws from
    emotion_style: Dict[str, str] = field(default_factory=dict)
                                  # per-emotion expression mode for THIS
                                  # personality: anger/grief/joy/fear/shame/
                                  # love -> how it shows on them
    tell: str = ""                # involuntary physical tell when the wound
                                  # is touched (audiences learn to read it)
    relic: str = ""               # small personal object tied to the memory,
                                  # carried or kept; can appear on screen
    # --- spoken voice: how they actually SOUND in person ---
    dialect: str = ""             # region/culture/class/trade voice, e.g.
                                  # "East Texas drawl", "broad Australian",
                                  # "Miami Latina, English-dominant"
    speech_markers: List[str] = field(default_factory=list)
                                  # signature spoken markers: "y'all",
                                  # "mate", "fixin' to", "mija", "eh?"
    code_switching: str = ""      # if bilingual/bidialectal: when and how
                                  # they switch (e.g. "slips into Spanish
                                  # for endearments and when overwhelmed")
    # --- behavioral idiom: what this BODY does because of who it is -------
    passions: List[str] = field(default_factory=list)
                                  # 1-3 things they love: music, engines,
                                  # cooking, birds, cards, boxing...
    idle_behavior: str = ""       # what hands/body do when unoccupied,
                                  # rooted in a passion and THEIRS alone:
                                  # "drums a beat on whatever's nearest --
                                  # a can, a railing, his own knee";
                                  # "whistles the same six notes when
                                  # thinking"; "walks a coin across her
                                  # knuckles"
    sonic_signature: str = ""     # the sound their PRESENCE makes (can
                                  # announce them off-screen): jangling
                                  # keys, a dragged bootheel, humming
                                  # under the breath, a tapped rhythm
    skill_display: str = ""       # a physical competence that shows in
                                  # small confident actions (the "competence"
                                  # care hook made concrete and repeatable)
    social_energy: float = 0.0    # -5 (edges of rooms, speaks to objects)
                                  # .. +5 (works the room, fills silences,
                                  # touches shoulders, remembers names)
    comic_toolkit: Dict[str, str] = field(default_factory=dict)
                                  # only for characters with humor: modes
                                  # (sarcasm/exaggeration/literalism/...),
                                  # emphasis_habit (which words they lean
                                  # on), timing (pause-before-punch,
                                  # deadpan hold), running_gag they own


@dataclass
class RelationshipEdge:
    source: str = ""
    target: str = ""
    trust: float = 0.0
    affection: float = 0.0
    respect: float = 0.0
    attraction: float = 0.0
    fear: float = 0.0
    resentment: float = 0.0
    envy: float = 0.0
    empathy: float = 0.0
    power_type: str = ""              # e.g. "employer", "older sibling"
    perceived_power: float = 0.0      # −5 (subordinate) .. +5 (dominant)
    unspoken_truths: List[str] = field(default_factory=list)
    grievances: List[str] = field(default_factory=list)
    secrets_kept_from: List[str] = field(default_factory=list)
    tom_believes_trust: float = 0.0
    tom_believes_affection: float = 0.0
    tom_believes_respect: float = 0.0
    tom_believes_fear: float = 0.0
    feared_trajectory: str = ""
    hoped_trajectory: str = ""


class CharacterGraph:
    def __init__(self):
        self.nodes: Dict[str, CharacterNode] = {}
        self.edges: Dict[str, RelationshipEdge] = {}   # "A|B" directed

    # ---------------------------------------------------------------- access
    def add_node(self, node: CharacterNode):
        self.nodes[node.name] = node

    def add_edge(self, edge: RelationshipEdge):
        self.edges[f"{edge.source}|{edge.target}"] = edge

    def get_node(self, name: str) -> Optional[CharacterNode]:
        if name in self.nodes:
            return self.nodes[name]
        low = (name or "").lower()
        for nm, nd in self.nodes.items():
            if nm.lower() in low or low in nm.lower():
                return nd
        return None

    def get_edge(self, a: str, b: str) -> Optional[RelationshipEdge]:
        return self.edges.get(f"{a}|{b}")

    # ------------------------------------------------------- prompt renderers
    def voice_guide(self, name: str) -> str:
        node = self.get_node(name)
        parts = [f"{name}:"]
        if node:
            for label, val in (("archetype", node.archetype),
                               ("wound", node.core_wound),
                               ("longing", node.core_longing),
                               ("backstory", node.backstory_moment),
                               ("relic", node.relic),
                               ("tell (wound touched)", node.tell),
                               ("dialect", node.dialect),
                               ("code-switching", node.code_switching),
                               ("defense", node.defense_mechanism),
                               ("shadow", node.shadow),
                               ("voice", node.voice_signature),
                               ("rhythm", node.rhythm),
                               ("humor", node.humor_style)):
                if val:
                    parts.append(f"  {label}={str(val)[:180]}")
            if node.speech_markers:
                parts.append(f"  says things like: "
                             f"{', '.join(node.speech_markers[:5])}")
            if node.passions:
                parts.append(f"  loves: {', '.join(node.passions[:3])}")
            if node.idle_behavior:
                parts.append(f"  idle hands/body: {node.idle_behavior[:120]}")
            if node.sonic_signature:
                parts.append(f"  their presence sounds like: "
                             f"{node.sonic_signature[:100]}")
            if node.skill_display:
                parts.append(f"  visible competence: "
                             f"{node.skill_display[:100]}")
            if node.social_energy:
                se = float(node.social_energy)
                parts.append(f"  social energy {se:+.0f} "
                             f"({'works the room' if se > 0 else 'edges of the room'})")
            if node.comic_toolkit:
                ct = "; ".join(f"{k}: {v}" for k, v in
                               list(node.comic_toolkit.items())[:4])
                parts.append(f"  comic toolkit -> {ct}")
            if node.emotion_style:
                styles = "; ".join(f"{k}: {v}" for k, v in
                                   list(node.emotion_style.items())[:6])
                parts.append(f"  expresses emotion as -> {styles}")
            if node.lexical_habits:
                parts.append(f"  habits={', '.join(node.lexical_habits[:4])}")
            if node.metaphor_pool:
                parts.append(f"  imagery from={', '.join(node.metaphor_pool[:3])}")
        return "\n".join(parts)

    def relationship_block(self, a: str, b: str) -> str:
        e = self.get_edge(a, b) or self.get_edge(b, a)
        if e is None:
            return ""
        flip = e.source != a
        dims = [("trust", e.trust), ("affection", e.affection),
                ("respect", e.respect), ("attraction", e.attraction),
                ("fear", e.fear), ("resentment", e.resentment),
                ("envy", e.envy), ("empathy", e.empathy)]
        hot = sorted(dims, key=lambda kv: abs(float(kv[1] or 0)), reverse=True)
        chosen = [f"{k} {float(v):+.0f}" for k, v in hot
                  if abs(float(v or 0)) >= 1.5][:4]
        src, dst = (e.source, e.target)
        out = [f"{src} -> {dst}: " + (", ".join(chosen) if chosen else "neutral")]
        if e.power_type or e.perceived_power:
            out.append(f"    power: {e.power_type or '-'} "
                       f"({float(e.perceived_power):+.0f})")
        if e.unspoken_truths:
            out.append(f"    unspoken: {'; '.join(e.unspoken_truths[:2])}")
        if e.grievances:
            out.append(f"    grievances: {'; '.join(e.grievances[:2])}")
        if e.secrets_kept_from:
            out.append(f"    secrets kept: {'; '.join(e.secrets_kept_from[:2])}")
        tom = []
        for k in ("trust", "affection", "respect", "fear"):
            tv = float(getattr(e, f"tom_believes_{k}", 0) or 0)
            if abs(tv) >= 1.5:
                tom.append(f"{k} {tv:+.0f}")
        if tom:
            out.append(f"    {src} BELIEVES {dst} feels: " + ", ".join(tom))
        if e.feared_trajectory:
            out.append(f"    {src} fears it becomes: {e.feared_trajectory[:80]}")
        if e.hoped_trajectory:
            out.append(f"    {src} hopes it becomes: {e.hoped_trajectory[:80]}")
        if flip:
            out.insert(0, f"(edge stored as {src}->{dst}; read accordingly)")
        return "\n".join(out)

    def chemistry_block(self, speakers: List[str]) -> str:
        """Attraction/affection-driven SOCIAL REFLEX direction for the
        characters present. This is where 'people who like each other flirt
        and blush' becomes stageable behavior: for every present pair with
        meaningful attraction or deep affection, emit how compliments land,
        how proximity behaves, and how one-sidedness or secrecy changes the
        expression -- always rendered through that character's own emotion
        styles. Jealousy emerges where attraction meets envy or a feared
        trajectory."""
        lines = []
        seen = set()
        for a in speakers:
            for b in speakers:
                if a == b or (a, b) in seen:
                    continue
                seen.add((a, b))
                e = self.get_edge(a, b)
                if e is None:
                    continue
                attraction = float(e.attraction or 0)
                affection = float(e.affection or 0)
                node = self.get_node(a)
                love_style = ""
                if node and node.emotion_style:
                    love_style = node.emotion_style.get("love") or \
                        node.emotion_style.get("joy") or ""
                if attraction >= 2.0:
                    hidden = bool(e.secrets_kept_from or e.unspoken_truths)
                    believes_less = (float(e.tom_believes_affection or 0)
                                     < affection - 1.0)
                    mode = []
                    mode.append(
                        f"a compliment or moment of notice from {b} LANDS "
                        f"visibly on {a}"
                        + (f" ({love_style})" if love_style else
                           " (blush, ducked head, a too-quick deflection, a "
                           "smile fought down)"))
                    mode.append(f"{a} finds small reasons for proximity to "
                                f"{b}, holds eye contact a beat too long, "
                                f"and mirrors {b}'s posture without noticing")
                    if hidden:
                        mode.append(f"{a} is HIDING it, so the attraction "
                                    f"leaks only through the body and the "
                                    f"tell -- never through words")
                    if believes_less:
                        mode.append(f"{a} believes {b} feels less, so the "
                                    f"flirting is hesitant: offered, then "
                                    f"half-withdrawn, tested as a joke")
                    lines.append(f"- {a} is drawn to {b}: " + "; ".join(mode))
                elif affection >= 3.0:
                    lines.append(
                        f"- {a} deeply cares for {b}: praise from {b} "
                        f"straightens {a}'s spine and softens the voice; "
                        f"teasing between them reads as affection; {a} "
                        f"tracks {b}'s state across the room without "
                        f"meaning to")
                if attraction >= 2.0 and (float(e.envy or 0) >= 2.0
                                          or e.feared_trajectory):
                    lines.append(
                        f"- when {b}'s attention goes elsewhere, {a}'s "
                        f"jealousy shows as a glance held too long and a "
                        f"subject changed too briskly -- never named aloud")
        # Dyad rhythm: two-handers get CHOREOGRAPHY, not two solo
        # performances -- who initiates, who mirrors, who breaks eye
        # contact first, derived from social energy, power, and attraction.
        pairs_done = set()
        for a in speakers:
            for b in speakers:
                if a == b or (b, a) in pairs_done or (a, b) in pairs_done:
                    continue
                pairs_done.add((a, b))
                na, nb = self.get_node(a), self.get_node(b)
                if na is None or nb is None:
                    continue
                sa = float(getattr(na, "social_energy", 0) or 0)
                sb = float(getattr(nb, "social_energy", 0) or 0)
                if abs(sa - sb) < 2.0:
                    continue
                lead, follow = (a, b) if sa > sb else (b, a)
                e = self.get_edge(follow, lead)
                drawn = bool(e and float(e.attraction or 0) >= 2.0)
                lines.append(
                    f"- DYAD RHYTHM {lead} & {follow}: {lead} initiates -- "
                    f"closes distance, starts and re-starts the talk, fills "
                    f"silences; {follow} responds -- mirrors posture half a "
                    f"beat late, "
                    + ("holds eye contact a moment too long before looking "
                       "away" if drawn else "breaks eye contact first")
                    + f". Stage their scenes as this choreography, never as "
                    f"two people performing separately.")
        if not lines:
            return ""
        return ("CHEMISTRY & SOCIAL REFLEXES (stage these as visible human "
                "behavior -- these are the moments audiences fall for):\n"
                + "\n".join(lines))

    def behavior_block(self, speakers: List[str]) -> str:
        """BEHAVIORAL IDIOM direction for the characters present: what
        each body does with time, objects, sound, and the room because of
        WHO IT IS -- independent of the current emotion. This is the layer
        that makes a music-lover drum a beat on a can, a comic land a
        punchline with a held deadpan, a charismatic character work the
        room. Injected into scene staging (actions), still physicality,
        and -- via sonic signatures -- the soundscape design."""
        lines = []
        any_comic = False
        for s in speakers:
            node = self.get_node(s)
            if node is None:
                continue
            bits = []
            if node.idle_behavior:
                bits.append(
                    f"when {s}'s hands are free, reach for THEIR idle "
                    f"behavior before any generic gesture: "
                    f"{node.idle_behavior}. And let it REACT to the scene "
                    f"-- it speeds up under stress, softens in comfort, "
                    f"and STOPS DEAD at a shock (behavior that modulates "
                    f"with tension is characterization; behavior that "
                    f"just loops is wallpaper)")
            if node.sonic_signature:
                bits.append(
                    f"{s}'s presence has a sound ({node.sonic_signature}) "
                    f"-- it can announce them a beat before they enter "
                    f"frame, and its absence can mark that something is "
                    f"wrong")
            if node.skill_display:
                bits.append(
                    f"let {s}'s competence show in small confident "
                    f"physical actions: {node.skill_display}")
            se = float(getattr(node, "social_energy", 0) or 0)
            if se >= 2.0:
                bits.append(
                    f"{s} WORKS the room: enters toward people, uses "
                    f"names, touches a shoulder, fills the first silence, "
                    f"draws other characters' eye-lines")
            elif se <= -2.0:
                bits.append(
                    f"{s} keeps to the EDGES: hugs frame borders, "
                    f"addresses objects or the middle distance instead of "
                    f"faces, lets silences sit")
            if node.comic_toolkit:
                any_comic = True
                ct = node.comic_toolkit
                tk = []
                if ct.get("modes"):
                    tk.append(f"modes: {ct['modes']}")
                if ct.get("emphasis_habit"):
                    tk.append(f"emphasis: {ct['emphasis_habit']}")
                if ct.get("timing"):
                    tk.append(f"timing: {ct['timing']}")
                if ct.get("running_gag"):
                    tk.append(f"owns the running gag: {ct['running_gag']}")
                bits.append(f"{s}'s comedy machinery -- {'; '.join(tk)}")
            if bits:
                lines.append(f"- {s}: " + ". ".join(bits) + ".")
        if not lines:
            return ""
        block = ("BEHAVIORAL IDIOM (each body's signature behavior -- use "
                 "these to give hands, feet, and rooms something TRUE to "
                 "do; a character's passion should leak into their "
                 "physical life, the way a drummer can't sit near a "
                 "surface without tapping it):\n" + "\n".join(lines))
        if any_comic:
            block += (
                "\nCOMIC EXECUTION: the timing lives in the DELIVERY "
                "CUES, which the voice literally performs -- write them "
                "as playable direction (\"flat, then hitting the word "
                "'fine' way too hard\", \"a beat of silence first, then "
                "deadpan\", \"speeding up as the excuse collapses\"). Put "
                "word emphasis in the CUE, never in capital letters "
                "inside the spoken text. Sarcasm is subtext made audible "
                "-- aim it along the theory-of-mind edges (a character "
                "is most sarcastic with the person they wrongly believe "
                "disrespects them). Never explain the joke; the shot "
                "after a punchline opens on a silent reaction.")
        return block

    def scene_subtext_block(self, speakers: List[str]) -> str:
        """Everything the dialogue writer needs about who's in the room."""
        blocks = []
        for s in speakers:
            g = self.voice_guide(s)
            if g:
                blocks.append(g)
        seen = set()
        for a in speakers:
            for b in speakers:
                if a == b or (a, b) in seen:
                    continue
                seen.add((a, b))
                rb = self.relationship_block(a, b)
                if rb:
                    blocks.append(rb)
        return "\n".join(blocks)

    # --------------------------------------------------------- serialization
    def to_dict(self) -> Dict:
        return {"nodes": {k: asdict(v) for k, v in self.nodes.items()},
                "edges": {k: asdict(v) for k, v in self.edges.items()}}

    @classmethod
    def from_dict(cls, d: Dict) -> "CharacterGraph":
        g = cls()
        for k, v in (d.get("nodes") or {}).items():
            g.nodes[k] = CharacterNode(**{f: v.get(f, CharacterNode().__getattribute__(f))
                                          for f in CharacterNode.__dataclass_fields__})
        for k, v in (d.get("edges") or {}).items():
            g.edges[k] = RelationshipEdge(
                **{f: v.get(f, RelationshipEdge().__getattribute__(f))
                   for f in RelationshipEdge.__dataclass_fields__})
        return g


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _safe_float(v) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def build_character_graph(characters: List[Dict], story: Dict,
                          register_block: str) -> CharacterGraph:
    """Two LLM passes: deep nodes for every character, then the full directed
    relationship matrix with theory-of-mind. Fails soft to shallow defaults."""
    graph = CharacterGraph()
    names = [c["name"] for c in characters]
    logger.info("[graph] Building EI nodes for %d character(s)...", len(names))

    char_lines = "\n".join(
        f"- {c['name']} ({c.get('role','')}): {c.get('summary','')[:220]}"
        for c in characters)

    prompt = f"""
Design the deep psychology of this film's cast. For each character produce an
emotional-intelligence node that a dialogue writer can act from.

Story: {story.get('logline','')}
Themes: {', '.join(story.get('themes', []))}
{register_block}

EMPATHY GROUNDING (critical): every character's wound and longing must be
rooted in ONE concrete, remembered, FILMABLE moment drawn from universal
human experience -- the kind of thing the audience has lived or watched
someone live. Draw from experiences like: being bullied or humiliated at
school, a day in court or a night in a cell, an unfair punishment for a
small mistake, picking a fight and regretting it, being the last one picked,
a parent's absence at the moment it mattered, the birth of a child, a
wedding (their own or one they watched from the edge), making a first real
friend, the moment an invention or creation finally worked, losing a job,
teaching someone something and watching it click, caring for someone dying.
Choose what fits THIS story and register; make it specific (a place, a
sound, one sensory detail), not generic. The audience should recognize
themselves in it.

DIFFERENT PERSONALITIES EXPRESS EMOTION DIFFERENTLY (critical): for each
character define HOW their personality shows each core emotion -- a
suppressor's anger goes quiet and precise while an exploder's fills the
room; one person's grief is busy hands and cooking for everyone, another's
is stillness; joy can be a contained glow or infectious noise. Make each
cast member's expression styles CONTRAST with the others, so the same event
lands differently on every face in the room.

AUTHENTIC SPOKEN VOICE (critical -- every line is literally SPOKEN aloud by
the film): give each character the voice they would actually have in
person, rooted in their region, culture, class, and trade. A rural Texan
says "y'all" and "fixin' to"; an Australian says "mate" and "reckon"; a
bilingual Latina slips into Spanish for endearments and exclamations
("mija", "ay, Dios"); a career lawyer speaks in qualified clauses even at
home. Carry dialect through WORD CHOICE, IDIOM, and RHYTHM with only light
eye-dialect spelling (y'all, gonna, 'em) -- never dense phonetic spelling.
Dialect is identity and music, written with an insider's affection, never
mockery. Voices must be instantly distinguishable with eyes closed.

BEHAVIORAL IDIOM (critical -- character is what a body DOES, not just what
a mouth says): give each character 1-3 PASSIONS that fit this story, and
derive from them a signature IDLE BEHAVIOR -- what their hands and body do
when unoccupied, theirs alone: a music-lover drums paradiddles on any flat
surface or whistles the same six notes while thinking; a mechanic can't
hold an object without checking how it's made; a card player walks a coin
across her knuckles. Give each a SONIC SIGNATURE -- the sound their
presence makes (jangling keys, a dragged bootheel, humming under the
breath) -- and a small visible COMPETENCE. Rate each character's SOCIAL
ENERGY from -5 (edges of rooms, speaks to objects, lets silences sit) to
+5 (works the room, fills silences first, touches shoulders, remembers
names) -- a charismatic character flirts openly and socially, an insecure
one offers and half-withdraws; make the cast CONTRAST. For any character
whose humor_style is not \"none\", build their COMIC TOOLKIT: which modes
they use (sarcasm, exaggeration, literalism, self-deprecation, deadpan),
their emphasis habit (which words they stretch or lean on -- \"that went
GREAT\"), their timing (a half-beat pause before the punch; a deadpan hold
after), and the running gag they personally own.

Cast:
{char_lines}

Return ONLY a raw JSON array, one object per character, in the same order:
{{"name": "<exact name>",
  "archetype": "<2-4 words>",
  "core_wound": "<the defining hurt driving behavior, 1 sentence>",
  "core_longing": "<what they most want but can't ask for, 1 sentence>",
  "backstory_moment": "<2-3 sentences: the ONE concrete remembered scene
     from a universal experience that created the wound or longing --
     filmable, sensory, specific>",
  "backstory_source": "<the universal experience category it draws from>",
  "relic": "<a small physical object they keep, tied to that memory
     (a bent medal, a torn ticket stub, a child's drawing) -- or ''>",
  "tell": "<the involuntary physical tell when the wound is touched, <=10
     words (thumb worrying a ring, a laugh half a beat too late)>",
  "dialect": "<their spoken voice's home: region/culture/class/trade, <=10
     words>",
  "speech_markers": ["<3-5 signature SPOKEN markers they actually say:
     'y'all', 'mate', 'fixin' to', 'mija', 'per my understanding'>"],
  "code_switching": "<if bilingual/bidialectal: when and how they switch
     (which emotions pull them to which language), else ''>",
  "defense_mechanism": "<how they protect the wound, <=10 words>",
  "shadow": "<the trait they deny in themselves, <=10 words>",
  "voice_signature": "<how they sound: register, vocabulary, tempo, 1 sentence>",
  "lexical_habits": ["<3-4 signature words/phrases they reach for>"],
  "rhythm": "<clipped | flowing | halting | musical | blunt ...>",
  "metaphor_pool": ["<2-3 domains their imagery comes from (their trade, their past)>"],
  "humor_style": "<dry | deflecting | none | warm | dark | slapstick ...>",
  "emotion_style": {{"anger": "<how anger shows on THIS person, <=10 words>",
                    "grief": "<...>", "joy": "<...>", "fear": "<...>",
                    "shame": "<...>", "love": "<...>"}},
  "fear_response": "<how fear physically shows on THIS body, <=10 words>",
  "joy_response": "<how joy physically shows on THIS body, <=10 words>",
  "passions": ["<1-3 things they love that fit this story>"],
  "idle_behavior": "<what hands/body do when unoccupied, rooted in a
     passion, <=18 words>",
  "sonic_signature": "<the sound their presence makes, <=10 words>",
  "skill_display": "<a competence visible in small physical actions, <=12
     words>",
  "social_energy": 0,
  "comic_toolkit": {{"modes": "<their comedy modes, or omit the whole
                      object if humor_style is none>",
                    "emphasis_habit": "<which words they lean on>",
                    "timing": "<their delivery timing habit>",
                    "running_gag": "<the gag they own, or ''>"}}}}
"""
    for item in safe_json_list(get_llm(prompt, temperature=0.7)):
        name = clean_text(item.get("name", ""))
        if not name:
            continue
        graph.add_node(CharacterNode(
            name=name,
            archetype=clean_text(item.get("archetype", "")),
            core_wound=clean_text(item.get("core_wound", "")),
            core_longing=clean_text(item.get("core_longing", "")),
            backstory_moment=clean_text(item.get("backstory_moment", "")),
            backstory_source=clean_text(item.get("backstory_source", "")),
            relic=clean_text(item.get("relic", "")),
            tell=clean_text(item.get("tell", "")),
            dialect=clean_text(item.get("dialect", "")),
            speech_markers=[clean_text(str(x)) for x in
                            (item.get("speech_markers") or []) if x][:5],
            code_switching=clean_text(item.get("code_switching", "")),
            defense_mechanism=clean_text(item.get("defense_mechanism", "")),
            shadow=clean_text(item.get("shadow", "")),
            voice_signature=clean_text(item.get("voice_signature", "")),
            lexical_habits=[clean_text(x) for x in
                            (item.get("lexical_habits") or [])][:5],
            rhythm=clean_text(item.get("rhythm", "")),
            metaphor_pool=[clean_text(x) for x in
                           (item.get("metaphor_pool") or [])][:4],
            humor_style=clean_text(item.get("humor_style", "")),
            emotion_style={str(k): clean_text(str(v)) for k, v in
                           (item.get("emotion_style") or {}).items()
                           if v},
            fear_response=clean_text(item.get("fear_response", "")),
            joy_response=clean_text(item.get("joy_response", "")),
            passions=[clean_text(str(x)) for x in
                      (item.get("passions") or []) if x][:3],
            idle_behavior=clean_text(item.get("idle_behavior", "")),
            sonic_signature=clean_text(item.get("sonic_signature", "")),
            skill_display=clean_text(item.get("skill_display", "")),
            social_energy=_safe_float(item.get("social_energy", 0)),
            comic_toolkit={str(k): clean_text(str(v)) for k, v in
                           (item.get("comic_toolkit") or {}).items()
                           if v and str(v).strip()},
        ))
    for c in characters:                       # fallback shallow nodes
        if c["name"] not in graph.nodes:
            graph.add_node(CharacterNode(name=c["name"],
                                         archetype=c.get("role", "")))

    logger.info("[graph] Building relationship matrix with theory-of-mind...")
    node_lines = "\n".join(
        f"- {n.name}: wound={n.core_wound[:80]} | longing={n.core_longing[:80]}"
        for n in graph.nodes.values())
    prompt = f"""
Design the DIRECTED relationship matrix for this cast. For every ordered pair
(A, B) where a real relationship exists (skip pairs who never meaningfully
interact), produce an edge. Values are -5..+5. THEORY OF MIND fields describe
what A BELIEVES B feels about A -- make at least two beliefs meaningfully
WRONG somewhere in the matrix, because misreading is where drama lives.

Story: {story.get('logline','')}
{register_block}

Cast psychology:
{node_lines}

Return ONLY a raw JSON array of edge objects:
{{"source": "A", "target": "B",
  "trust": 0, "affection": 0, "respect": 0, "attraction": 0,
  "fear": 0, "resentment": 0, "envy": 0, "empathy": 0,
  "power_type": "<the axis of power between them, or ''>",
  "perceived_power": 0,
  "unspoken_truths": ["<what A knows but won't say to B>"],
  "grievances": ["<what A holds against B>"],
  "secrets_kept_from": ["<what A hides from B>"],
  "tom_believes_trust": 0, "tom_believes_affection": 0,
  "tom_believes_respect": 0, "tom_believes_fear": 0,
  "feared_trajectory": "<where A fears this goes>",
  "hoped_trajectory": "<where A hopes this goes>"}}
"""
    for item in safe_json_list(get_llm(prompt, temperature=0.7, large=True)):
        src, dst = item.get("source", ""), item.get("target", "")
        if not src or not dst or src == dst:
            continue

        def _f(k):
            try:
                return float(item.get(k, 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        def _l(k):
            return [clean_text(str(x)) for x in (item.get(k) or [])
                    if str(x).strip()][:3]

        graph.add_edge(RelationshipEdge(
            source=src, target=dst,
            trust=_f("trust"), affection=_f("affection"),
            respect=_f("respect"), attraction=_f("attraction"),
            fear=_f("fear"), resentment=_f("resentment"),
            envy=_f("envy"), empathy=_f("empathy"),
            power_type=clean_text(item.get("power_type", "")),
            perceived_power=_f("perceived_power"),
            unspoken_truths=_l("unspoken_truths"),
            grievances=_l("grievances"),
            secrets_kept_from=_l("secrets_kept_from"),
            tom_believes_trust=_f("tom_believes_trust"),
            tom_believes_affection=_f("tom_believes_affection"),
            tom_believes_respect=_f("tom_believes_respect"),
            tom_believes_fear=_f("tom_believes_fear"),
            feared_trajectory=clean_text(item.get("feared_trajectory", "")),
            hoped_trajectory=clean_text(item.get("hoped_trajectory", "")),
        ))
    logger.info("[graph] %d nodes, %d edges.", len(graph.nodes), len(graph.edges))
    return graph
