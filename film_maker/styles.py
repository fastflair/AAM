"""
film_maker.styles
=================
Author-selectable STYLE layers, orthogonal to the registers (a horror film
can be watercolor or photoreal; a comedy can be handheld or classical).
Each knob accepts a preset key OR free text (free text is used verbatim).

  visual_style   — the rendering medium family. Locks the style bible's
                   medium and inflects palette/lighting/grammar choices.
  camera_style   — the film's camera personality, layered under the
                   tension-band grammar (the grammar says WHAT moves are
                   available per band; the camera style says HOW this film
                   executes them).
  dialogue_style — the film's line-writing texture, layered under the
                   register dialogue methods (register wins conflicts; the
                   style refines the sentences themselves). Includes the
                   Twain-craft vernacular mode: dialect carried by word
                   choice, idiom, and rhythm with only light eye-dialect,
                   because every line is literally SPOKEN by the model.
"""
from __future__ import annotations

from typing import Dict

VISUAL_STYLES: Dict[str, str] = {
    "photorealistic": (
        "photorealistic live-action cinematography: physically-plausible "
        "light, true skin and fabric micro-texture, real lens behavior "
        "(bokeh, subtle chromatic falloff), no illustration stylization"),
    "cgi": (
        "high-end 3D CGI feature-animation render: sculpted forms, "
        "subsurface-scattered skin, cinematic raytraced light, painterly "
        "color keys, expressive but dimensionally consistent characters"),
    "graphic_novel": (
        "graphic-novel illustration brought to motion: bold confident ink "
        "linework, dramatic spot blacks, screentone and hatching texture, "
        "flat-but-rich color fills, compositions that read like splash "
        "panels"),
    "watercolor": (
        "hand-painted watercolor animation: soft pigment blooms, visible "
        "paper grain, wet-edge color bleeding, luminous washes with "
        "loose-but-controlled linework, light carried by the white of the "
        "paper"),
    "anime": (
        "cinematic anime: clean expressive linework, painted backgrounds "
        "with atmospheric depth, dramatic lighting cels, wind-in-hair and "
        "fabric emphasis, emotionally heightened facial acting"),
    "oil_painting": (
        "living oil painting: visible impasto brushwork, old-master "
        "chiaroscuro, glazed color depth, edges dissolving into painterly "
        "atmosphere"),
    "stop_motion": (
        "stop-motion puppet animation: handcrafted miniature sets with "
        "real-material texture (felt, wood grain, clay fingerprints), "
        "slightly stepped motion charm, practical miniature lighting"),
    "noir_bw": (
        "black-and-white film noir photography: silver-gelatin contrast, "
        "hard key light through blinds and smoke, deep blacks, wet-street "
        "reflections"),
    "storybook": (
        "illustrated children's storybook come alive: gentle rounded "
        "shapes, warm gouache textures, cozy light, imperfect charming "
        "linework"),
}

CAMERA_STYLES: Dict[str, str] = {
    "classical": (
        "classical composed cinematography: tripod-stable frames, moves "
        "motivated and geometric (dolly, crane, measured pans), cuts carry "
        "the energy while the camera stays disciplined; compositions built "
        "on strong lines and balanced weight"),
    "handheld": (
        "handheld documentary energy: a subtle constant breathing drift "
        "even in 'static' shots, human imperfect reframes that chase the "
        "action a half-beat late, quick corrective pans to reactions; "
        "intimacy over polish"),
    "epic": (
        "sweeping epic camera: gliding crane and aerial moves that reveal "
        "scale, long confident pushes, foreground elements sweeping past "
        "the lens, horizon-conscious wides; the camera itself feels vast"),
    "tableau": (
        "static tableau style: locked-off symmetrical frames the action "
        "moves through, meaning built by staging within the frame rather "
        "than camera motion; the rare camera move lands like an event"),
    "kinetic": (
        "kinetic contemporary camera: whip pans, snap zooms, arcs that "
        "orbit the action, speed changes; the camera is a participant in "
        "the energy, never merely an observer"),
    "floating": (
        "floating dreamlike camera: slow weightless drifts, gentle arcs "
        "with no hard stops, moves that begin and end imperceptibly; the "
        "camera as a curious spirit in the room"),
}

DIALOGUE_STYLES: Dict[str, str] = {
    "vernacular": (
        "VERNACULAR (Twain-craft): every character speaks in the authentic "
        "voice of their region, culture, class, and trade -- dialect "
        "carried by WORD CHOICE, IDIOM, and SENTENCE RHYTHM first ('y'all', "
        "'mate', 'fixin' to', 'fair dinkum', 'mija'), with only LIGHT "
        "eye-dialect spelling (y'all, gonna, 'em, ain't) and never dense "
        "phonetic spelling, since every line is literally spoken aloud. "
        "Grammar bends the way real speech bends. Dialect is identity and "
        "music, never mockery -- write each voice with the affection of an "
        "insider. Code-switching is welcome where a character's background "
        "supports it: emotion pulls a bilingual speaker toward their first "
        "language for endearments, oaths, and prayers."),
    "spare": (
        "SPARE: short declarative sentences, concrete nouns, almost no "
        "adverbs; emotion carried by what is omitted -- the iceberg under "
        "the line. Repetition used deliberately, like a bell. Characters "
        "say less than they know and far less than they feel."),
    "rapid_wit": (
        "RAPID WIT: fast overlapping exchanges, characters finishing and "
        "hijacking each other's sentences, wordplay under pressure, "
        "intelligence as flirtation and combat; the argument IS the "
        "relationship. Keep individual lines short so the volley stays "
        "playable at speaking speed."),
    "naturalistic": (
        "NATURALISTIC: real-speech texture -- false starts, self-"
        "corrections, trailing sentences, 'I mean--', small verbal tics "
        "per character, people talking slightly past each other; the "
        "meaning lives between the lines, imperfectly."),
    "lyrical": (
        "LYRICAL: heightened, image-rich speech with rhythm and internal "
        "echo; characters reach for metaphor drawn from their own metaphor "
        "pools; earns its poetry by staying rooted in concrete things the "
        "speaker has touched."),
    "hardboiled": (
        "HARDBOILED: compressed, wry, armored speech; similes with teeth; "
        "threat and tenderness both delivered deadpan; nobody explains "
        "themselves twice."),
}


def _resolve(value: str, presets: Dict[str, str]) -> str:
    key = (value or "").strip()
    if not key:
        return ""
    norm = key.lower().replace(" ", "_").replace("-", "_")
    for k, v in presets.items():
        if norm == k or norm in k or k in norm:
            return v
    return key  # free text: use verbatim


def visual_style_block(cfg: Dict) -> str:
    style = _resolve(cfg.get("visual_style", ""), VISUAL_STYLES)
    if not style:
        return ""
    return (f"AUTHOR-LOCKED VISUAL STYLE (non-negotiable; the film's medium "
            f"MUST realize this, and palette/lighting/grammar are chosen to "
            f"serve it): {style}")


def camera_style_block(cfg: Dict) -> str:
    style = _resolve(cfg.get("camera_style", ""), CAMERA_STYLES)
    if not style:
        return ""
    return (f"AUTHOR-LOCKED CAMERA STYLE (the film's camera personality -- "
            f"execute every tension-band move in this manner): {style}")


def dialogue_style_block(cfg: Dict) -> str:
    style = _resolve(cfg.get("dialogue_style", ""), DIALOGUE_STYLES)
    if not style:
        return ""
    return (f"AUTHOR-LOCKED DIALOGUE STYLE (the line-writing texture; "
            f"register craft wins structural conflicts, this style shapes "
            f"the sentences themselves): {style}")
