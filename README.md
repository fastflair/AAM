# Film Maker — idea → animated film (MiniMax H3)

Turns a rough human idea into a 15+ minute animated film with spoken
dialogue, performances, and a designed diegetic soundscape — all rendered by
MiniMax H3 through Wan2GP, **keeping H3's own generated audio** as the
film's sound. You add music in post (`non_diegetic_music: N/A` on every
shot, by design).

The notebook stays tiny: set config, `plan_film()`, edit `plan.json`,
`produce_film()`. See `FilmMaker.ipynb`.

## Pipeline

```
IDEA
 └─ story.py          concept → cast (locked visual descriptions + voice
                      textures) → truth arc (want/need/lie) → act-structured
                      beat sheet with a real tension curve → CRITIC pass
                      (attack + rewrite the weakest beats) → setup/payoff
                      ledger + through-line object
 └─ graph.py          EI CharacterGraph: wounds, longings, defenses, shadow,
                      voice signatures; directed relationship edges with
                      trust/fear/resentment/attraction, power, unspoken
                      truths, secrets, and THEORY OF MIND (deliberately
                      wrong beliefs → dramatic irony)
 └─ screenplay.py     beats → scenes → shots. Pacing engine: shot durations
                      follow tension through the register's pacing curve.
                      Whole-scene EI dialogue with delivery cues, spoken-word
                      budgets tied to shot length (~2.2 w/s so lip-sync never
                      rushes), a polish pass on the talkiest scenes, and the
                      deterministic repetition guard.
 └─ cinematography.py locked style bible · designed location sets with
                      verbatim locks + signature details + signature
                      AMBIENCES · wardrobe continuity plan · GLOBAL shot
                      composition plan (scale/angle/light with hard
                      anti-repetition + face-readability rules for dialogue
                      shots) · structured image prompts with all locks
                      reproduced verbatim · era lock + anachronism scrub ·
                      per-shot camera/motion clauses (tension-band grammar) ·
                      per-shot soundscapes composed under the registers'
                      sound-design methods, with 2–3 film-wide SOUND MOTIFS
                      planted and transformed across scenes
 └─ h3_prompts.py     the 3-part H3 prompt per shot:
                        [Shot 1: …visual block… (S1) says …: <d>[English] …</d> …]
                        overall_soundscape: …
                        non_diegetic_music: N/A
                      speaker IDs chronological per shot, dialogue only
                      inside <d>, off-screen narrator phrasing, word budget
                      enforced without ever cutting inside a <d> tag
 └─ images.py         Z-Image stills: N variants per shot (default 4), a
                      vision-model pick against a concrete checklist
                      (dialogue shots require a readable, unobstructed
                      mouth); variants kept on disk so you can override
 └─ render.py         Wan2GP batch adapter (ported from the proven music
                      pipeline: manifest → single wgp.py --process per wave,
                      conda-wrapped, stdout-parsed task→file mapping that
                      survives sliding windows, transient-GPU retries) with:
                        • H3 audio KEPT — conform trims video+audio together
                        • VOICE CHAINING in waves: wave 1 renders silent
                          shots + each character's first solo shot with no
                          audio ref; those clips' audio is harvested into a
                          voice bank; wave 2 renders everything else with
                          the lead speaker's banked clip as audio_guide
                          (multi-speaker shots get a concatenated ref).
                          The establishing clip is reused verbatim forever —
                          the voice cannot drift through copies.
 └─ assembly.py       audio-preserving concat (hard cuts default; optional
                      scene-boundary A/V crossfades)
```

## Registers (the artistic-method engine)

`FILM_CONFIG["registers"] = ["scifi", "comedy"]` — first is primary. Each
register in `registers.py` is a working craft playbook injected at the stage
it governs: **story method** (comedy's rule-of-three and running-gag payoff
inside the climax; horror's dread-before-shock and false relief; romance's
almost-touches), **dialogue method** (punchlines on the last word + silent
reaction beats; fear under-writing; charged restraint), **sound method**
(silence before scares, a threat's signature sound planted two scenes early,
receding ambience in intimate scenes), **pacing curves** (comedy cuts ~25%
faster; horror holds ~35% longer at low tension and snaps at high), and
**visual bias**. `content_rating` gates the mature register.

## Operational notes

* Same hardware assumptions as the music generator: 4090 24GB + 128GB RAM,
  Wan2GP fork at `~/repos/Wan2GP` in the `wan2gp` conda env, Turbo LoRA
  auto-installed (8 steps; set `wgp_use_turbo_lora=False` → ~20 steps).
* **Idempotent everywhere.** Re-running `produce_film` reuses finished
  stills, voice refs, and clips. `regenerate_scenes(plan, cfg, [ids])`
  deletes only those clips, re-renders them (pure H3 re-roll, per your
  spec), and reassembles.
* Overriding a still by hand: delete `stills/image_<shot>.jpg`, copy your
  preferred `_v*.jpg` variant over that name, re-produce.
* Voice bank lives at `film_output/<slug>/voice_bank/`. Drop in your own
  `<Character>.wav` (2–15s, mono) before wave 2 to force a voice.
* API keys come from `OPENAI_API_KEY` / `XAI_API_KEY` / `HF_TOKEN` env vars —
  nothing hardcoded. (Rotate the keys that were in the old script.)
* Frame math: target = round(duration×24); rendered at the next valid
  5+17k grid length (≤737 = one H3 window, no seams); trimmed back to the
  exact target with audio intact.

## Install

Planning phase: `pip install openai json-repair ftfy unidecode tiktoken`.
Production adds the music generator's stack: `torch diffusers soundfile
huggingface_hub pillow` + ffmpeg/ffprobe on PATH + your working Wan2GP env.

## Long scenes and long takes

Long **scenes** were always handled the filmic way: as sequences of ≤30s
shots (cuts), each a single-window H3 generation — cuts are seamless by
definition, and voice chaining carries vocal identity across them.

Long **takes** (one continuous shot beyond H3's 30.7s single window) are now
supported two ways, chosen by `long_take_mode`:

* **`sliding_window`** (default) — the full `video_length` is submitted in
  one task and Wan2GP's built-in H3 sliding-window continuation stitches the
  windows internally (its FL2VA-style last-frames→next-window mechanism),
  with continuous H3 audio. The stdout parser already keeps only each task's
  FINAL window, so nothing else changes. The motion prompt for any shot over
  30s is written in 2–3 temporal PHASES with the camera continuously moving
  through phase boundaries, which masks the faint seam a window join can show.
* **`chain`** — explicit segmenting: each ≤30s segment renders separately,
  continuing from the previous segment's **last frame** (`image_start`) and
  using its **tail audio** as the reference (voice + ambience continuity),
  then segments are concatenated and conformed. More control, at the cost of
  a possible visual pop at joins. The continuation field names are config
  knobs (`wgp_chain_start_image_prompt_type`, `wgp_chain_start_image_field`)
  — verify them once against your fork with `wgp_dry_run=True` and an
  "Export Settings" JSON before a real chained run.

The pacing engine auto-promotes up to `long_take_budget` (default 3)
low-tension scenes into 30–60s oners where the register's craft favors a
held shot (horror dread builds, drama resolutions); fast-cutting registers
get at most one. You can also hand-edit any shot's `duration` past 30 in
plan.json and the renderer routes it automatically.

## Craft passes (v2)

* **Color script** — a per-scene grade progression within the locked palette
  (warmth drains toward the low point, the climax gets the film's most
  extreme grade), injected into every image prompt.
* **Cut design** — every scene boundary is an authored editorial choice:
  hard contrast, a match cut (incoming frame echoes the outgoing shape or
  gesture; used sparingly at meaning-bearing turns), or a **sound bridge** —
  the next scene's ambience faintly pre-enters under the outgoing shot's
  soundscape, a J-cut built inside H3's own generated audio.
* **Dynamics map** — a deterministic loudness/density arc from the tension
  curve: the scene before the peak is deliberately hushed (the held breath),
  the peak is the densest, the final scene settles.
* **Performance physicality** — each character's `fear_response` /
  `joy_response` / defense mechanism from the EI graph now shapes how
  emotion appears on *their* body in the stills, instead of generic
  sadness/joy.
* **Character introductions** — a principal's first on-screen appearance is
  designed as a character-defining entrance (met mid-behavior, never idle).
* **Opening/closing rhyme** — the style bible's opening and closing image
  intentions are realized on the film's literal first and last frames.
* **Speaker binding + speakable dialogue** — in multi-character shots, each
  speaker's first line carries their `screen_tag` ("Cole, the wiry radioman
  with the cracked glasses, (S2) says…") so H3 binds the voice to the right
  body; dialogue is normalized to spoken words (Dr.→Doctor, 3→three).
* **Planning checkpoints** — every planning stage is cached under a hash of
  (idea, registers, rating, length); a crash or rate-limit resumes without
  repaying finished stages (`plan_film(..., resume=False)` to force fresh).
* **Table read** — `table_read.md` lands next to the plan: a frank story-
  editor pass naming the weakest scenes, flat dialogue, pacing lulls, and
  unpaid setups (non-destructive — you decide what to edit in plan.json).

## Connection layer (v3) — making the audience care

* **Grounded backstories** — every character's wound is rooted in ONE
  concrete, filmable memory drawn from *universal human experience* (being
  humiliated at school, a night in a cell, an unfair punishment, a first
  invention finally working, a wedding watched from the edge, a birth, a
  first real friend...). Each comes with a **relic** (a small object tied to
  the memory) and an involuntary **tell** that fires when the present rhymes
  with the past.
* **Backstory echoes, not flashback dumps** — the connection plan surfaces
  each memory obliquely 2–3 times (relic glimpsed, a sentence that stops,
  the tell firing) and pays it off in full exactly once, near the
  character's hardest choice. The audience assembles the memory from
  fragments — which is what makes it theirs.
* **Care hooks** — the proven bonding levers (undeserved misfortune,
  kindness unobserved, competence, vulnerability, humor, longing shown not
  told) assigned to specific first-quarter scenes as stageable micro-beats,
  rationed deliberately. The antagonist gets one: understood ≠ excused.
* **Personality-specific emotion** — each character gets a per-emotion
  expression map (a suppressor's anger goes quiet and precise; one person's
  grief is busy hands, another's is stillness), designed to CONTRAST across
  the cast so the same event lands differently on every face. Consumed by
  dialogue, stills, and reaction beats.
* **Reaction beats** — empathy is transmitted through faces reacting: after
  every high-tension event the plan names whose face we hold on, expressed
  in *their* style.
* **Mirror moments** — 1–2 scenes staged inside instantly-recognized life
  textures (waiting rooms, school corridors, the walk to a front door with
  news) so viewers see their own lives in the frame.
* **Eureka ladder** — realizations engineered so the audience solves them
  one beat before the character: three visible clues planted innocently, a
  false floor (the near-miss), the CLICK as a visual reframe (two planted
  images collide into meaning — no explaining dialogue), then RELEASE in the
  character's own emotion style plus the action the understanding unlocks.

## Educational mode (v3)

`FILM_CONFIG["film_mode"] = "educational"` — the idea becomes a TOPIC:

```python
FILM_CONFIG["film_mode"] = "educational"
FILM_CONFIG["registers"] = ["wonder"]        # explainer is auto-added first
plan_film("How vaccines train the immune system", FILM_CONFIG)
```

A pedagogy spec is built before the story: learning objective, a concrete
**hook question** (curiosity gap), the intuitive **misconception** the
protagonist acts on and watches fail (refutational teaching), **ONE core
analogy** carried consistently through the whole film, a **key-idea ladder**
(one idea per scene, strict dependency order, each SHOWN inside the analogy,
never lectured), and a payoff where the understanding wins something the
character personally needs. The eureka ladder carries the concept's click as
the climax; the resolution has the protagonist re-explain it imperfectly to
someone else — teaching it is the proof they own it. The critic pass and
table read both add a comprehension audit ("could a smart 12-year-old
re-explain it after watching?").

## Robustness (v3.1) — no more silent truncation loss

* **Bigger budgets** — default completion budgets raised to 60k tokens
  (100k for `large=True` calls), sized for reasoning models where the
  budget also covers reasoning tokens. Knobs: `llm_max_completion_tokens`,
  `llm_max_completion_tokens_large`.
* **Automatic truncation retry** — any response with
  `finish_reason == "length"` is retried with a doubled budget up to
  `llm_truncation_retry_cap` (160k default) before a partial is ever
  accepted, and truncation retries don't consume failure attempts. A
  truncated JSON array silently loses its tail when repaired — this makes
  that failure mode self-healing instead of a warning in the log.
* **Batched bulk calls** — the two calls whose response size grows with the
  film (the global shot-composition plan: 130+ objects on a 20-minute film;
  the anachronism scrub) now run through a generic
  `batched_json_call` helper: ordered batches (`shot_plan_batch_size`,
  `scrub_batch_size`), missing-item repair passes, and — critically — a
  context recap threaded between batches (last 10 assignments + most-used
  scale/angle pairs) so film-wide anti-repetition rules survive batching
  instead of each batch planning blind.
* The payoff-ledger prompt now carries an explicit brevity contract (the
  call that truncated in the field).

## Projects & storyboard context (v3.2)

* **`project_name`** — set it and everything for one film (planning cache,
  plan.json, plan.md, table_read.md, stills, voice bank, clips, final) lives
  under `<base_working_dir>/<project>/` + `<base_output_dir>/<project>/`.
  Different names = independent films side by side. The planning cache
  moves inside the project folder and carries a fingerprint of (idea,
  registers, rating, length, tone, mode) — change any of those and stale
  stages are auto-invalidated instead of mixed in.
* **Storyboard context everywhere** — a compact one-line-per-scene film map
  (with the current scene marked `>>`, plus ONER/connection/eureka flags) is
  now threaded into every sequential and batched call:
  scene authoring additionally receives a rolling **authored digest**
  (staged actions + key dialogue from every scene already written, token-
  clamped) so setups land and callbacks connect; shot-composition batches
  get the film map + full assignment recap; image-prompt calls get the map
  so planted imagery recurs *transformed*, never duplicated by accident;
  motion/soundscape calls get the map + the last few designed soundscapes
  so the sound develops across scenes instead of resetting.

## Style layers & human texture (v3.3)

* **`visual_style`** — author-locks the medium family (presets: watercolor,
  graphic_novel, photorealistic, cgi, anime, oil_painting, stop_motion,
  noir_bw, storybook — or free text). The style bible must realize it, with
  a hard code-level guarantee that the locked style anchors the medium even
  if the LLM drifts.
* **`camera_style`** — the film's camera personality (classical, handheld,
  epic, tableau, kinetic, floating — or free text), layered under the
  tension-band grammar in both the shot-composition plan and every motion
  clause: the grammar says *what* moves are available, the style says *how
  this film executes them*.
* **`dialogue_style`** — line-writing texture (default **vernacular**: the
  Twain craft — dialect carried by word choice, idiom, and rhythm, light
  eye-dialect only since every line is literally spoken, written with an
  insider's affection and never as mockery; also spare, rapid_wit,
  naturalistic, lyrical, hardboiled, or free text).
* **Authentic spoken voice** — every character now gets a `dialect`,
  signature `speech_markers` ("y'all", "mate", "fixin' to", "mija"), and a
  `code_switching` profile (which emotions pull a bilingual speaker toward
  which language). Dialogue rules require lines a stranger could attribute
  with eyes closed. **Per-line language switching**: brief embedded words
  stay in the default-language tag; a line spoken fully in another language
  sets `"language": "Spanish"` and renders as `<d>[Spanish] …</d>` so H3
  speaks it natively.
* **Chemistry & social reflexes** — derived from the graph's attraction/
  affection edges: compliments from someone a character is drawn to LAND
  visibly (a blush fought down, a too-quick deflection — in that
  character's own love/joy style); proximity-seeking, held eye contact,
  unconscious posture mirroring; hidden attraction leaks only through the
  body; one-sided belief (theory-of-mind) makes flirting hesitant —
  offered, then half-withdrawn as a joke; jealousy appears where attraction
  meets envy. Injected into scene staging, the dialogue polish pass, and
  the still-image physicality so blushes and stolen glances are literally
  visible in the frames.

## Render monitoring (v3.4)

* **Readable raw filenames** — the moment wgp.py saves a clip it is renamed
  in place from the unwieldy `{datetime}_seed{seed}_{full prompt}.mp4` to:
  `scene_04_shot_02_seed....mp4`, with `_window_2of3` on multi-window long
  takes and `_segment_02` on chain-mode segments — so `wgp_render/raw/` is
  scannable while a batch is still running. Each task's final file is still
  moved to `shot_clips/shot_<id>.mp4` for assembly as before.
* **Task heartbeat** — a `[wgp] >>> Task 17/80 started: shot 06_03 (251
  frames, 10.4s)` line on every task start ties wgp.py's progress to your
  shot list.
* **Wave-labeled manifests** — `batch_manifest_wave1.json`,
  `batch_manifest_wave2.json`, `batch_manifest_chain_...json` are kept side
  by side instead of overwriting one file, so a failed wave can be inspected
  after the fact.
* **Live `render_status.json`** — refreshed in `wgp_render/` after every
  wave: total/rendered counts and per-shot status, mode, duration, and clip
  path — watchable from another terminal during multi-hour batches.

## Unified project layout (v3.5)

One self-contained folder per film, replacing the film_work/film_output split:

```
<base_dir>/<project_name>/            (default base_dir: ./films)
  plan.json  plan.md  table_read.md  planning_cache.json
  stills/         reference stills + variants
  voice_bank/     established character voice refs
  clips/          conformed per-shot clips (with H3 audio)
  render/         wgp manifests, raw/ live renders, render_status.json
  <project>_final.mp4
```

* **Automatic migration** — an existing pre-v3.5 project (film_work/<slug> +
  film_output/<slug>) is moved into the unified folder the first time it's
  touched (shot_clips→clips, wgp_render→render); empty legacy dirs are
  removed. Old plan paths still work: produce/regenerate resolve them to the
  migrated location automatically. Idempotent and verified by tests.
* **`list_projects(FILM_CONFIG)`** — one call prints a table of every film:
  title, shot count, stills done, clips rendered, planned minutes, final
  present — and returns the data.
* **Final versioning** — reassembly (e.g. after `regenerate_scenes`) keeps
  the previous final as `<project>_final_<timestamp>.mp4` instead of
  overwriting, so versions can be compared (`keep_previous_finals` knob).

## Scene cohesion — continuity chaining (v4)

The anti-slop architecture change: shots in the same setting no longer
spring from independent synthesized stills (which reset torn sleeves, held
props, and weather at every cut). Three reinforcing mechanisms:

* **Continuity blocks** — consecutive scenes sharing a SETTING form one
  block. Only each block's OPENING shot gets a synthesized still; every
  later shot uses the **last sharp frame** of the previous shot's rendered
  clip as its H3 reference image (sharpest of the final ~12 frames by
  edge-energy, so motion blur never poisons the chain). The reference
  carries costume state, damage, props, lighting, and weather; the prompt
  (with the chained anchor: "the reference image is the exact moment this
  shot continues from... nothing resets") directs the new framing per the
  shot-composition plan. Returning to a location later is a NEW block —
  a fresh still, since time has passed.
* **Physical-state ledger** — a continuity-supervisor pass walks the whole
  film maintaining cumulative state (a ripped sleeve STAYS ripped until we
  see it changed; mud, blood, bandages, dropped objects, overturned
  furniture all persist). Each shot's `carry_state` is written into both
  the still prompts and the H3 prompts, so text and chained frames agree —
  and block-opening stills of LATER scenes inherit accumulated state too.
* **Round-based rendering** — round k renders the k-th shot of every block
  (one wgp.py call per round), so each chained shot's predecessor frame
  exists before it renders. Voice references harvest progressively after
  every round. Trade-off vs the old 2 waves: more wgp startups (= max
  shots per block, typically 8–12). Regenerating a mid-block shot chains
  off its existing neighbors on disk; handoff frames auto-refresh when a
  predecessor clip is newer. `scene_continuity_chaining: False` restores
  the old per-shot-still, 2-wave behavior.

Note: existing plans work as-is (blocks are computed at produce time), but
the physical-state ledger is a planning-stage pass — re-run `plan_film`
(the cache resumes everything except the shot_design stage if you delete
that key from planning_cache.json) to add `carry_state` to an older plan.

## GPU/RAM tuning: understanding mmgp's "partial pinning" message (v4.1)

If you see:
```
Switching to partial pinning since full requirements for pinned models is
X MB while estimated available reservable RAM is Y MB. You may increase
'perc_reserved_mem_max' to force full pinning.
```

**Partial pinning is not broken** — it only slows RAM↔VRAM transfer during
model load, with zero effect on output quality or correctness. Before
raising the percentage, do the math mmgp already gave you:

1. `Y / current_perc` = your reservable RAM at 100%.
2. `required_perc = X / that_100%_figure`.
3. Only raise `wgp_perc_reserved_mem_max` to `required_perc` if it leaves
   **~15–20% headroom** for the OS, ffmpeg, and this notebook process. A
   `required_perc` above ~0.85 means don't push the percentage further —
   instead: switch `wgp_model_type` to the pruned checkpoint
   (`minimax_h3_ref2va_pruned`), drop `wgp_profile` to `4`
   (LowRAM_LowVRAM), or increase actual system RAM (check WSL2's own cap in
   `.wslconfig` — it defaults to 50% of host RAM, a common surprise).

`wgp_profile` and `wgp_perc_reserved_mem_max` are now separate config knobs
(previously baked into one `wgp_extra_cli_args` list) so they're easy to
tune independently; raising the percentage past 0.85 logs a warning.
Confirm your actual RAM with `free -h` before assuming any figure.

### WSL2 users: check the WSL memory cap before tuning anything

If Windows Task Manager shows your full RAM (e.g. 128GB) but `free -h`
*inside* Ubuntu shows roughly half of that, the shortfall isn't your
hardware — **WSL2 defaults to allocating only 50% of host RAM** to the
Linux VM. mmgp's "reservable RAM" math runs inside that capped VM, so it
sees a ceiling your machine doesn't actually have, and pushing
`wgp_perc_reserved_mem_max` upward just squeezes harder against a false
limit. Fix the cap instead:

1. Create/edit `%UserProfile%\.wslconfig` on Windows:
   ```ini
   [wsl2]
   memory=110GB
   ```
   (leave headroom for Windows itself — don't set this to 100% of host RAM)
2. `wsl --shutdown` from PowerShell, then reopen your Ubuntu terminal.
3. `free -h` to confirm the new total is visible.
4. Re-run the render and redo the reservable-RAM math above at the new
   ceiling — full pinning, or a much smaller `perc_reserved_mem_max` bump,
   may now fit with real headroom instead of none.

Note: Windows Task Manager's "Shared GPU memory" figure is a separate
WDDM mechanism for Windows-native GPU processes borrowing system RAM as
VRAM overflow — H3 rendering runs through WSL2's own CUDA stack and
doesn't use that pool, so it isn't the number to watch here; `free -h`
inside WSL is.

## Fixed: long-take action restarting every window (v4.2)

**Symptom:** in a long take split across multiple render windows (e.g. a
person walking out a door), each window replayed the SAME beginning of the
action instead of continuing it — three windows, three identical "walks out
the door" instead of one continuous exit.

**Root cause, confirmed from render logs:** the default `long_take_mode` was
`"sliding_window"` — Wan2GP's own internal window-splitting, which submits
the shot as ONE task and sends **one static text prompt across every
internal window**. H3 has no way to be told "this is window 2, continue
forward" — only frame overlap argues for continuation, and when the prompt
keeps re-describing the same action, the text conditioning wins and the
action restarts. A second bug compounded it: our own `"chain"` mode (built
for exactly this) was reusing the identical prompt for every segment too.

**Fix, three parts:**
* **`long_take_mode` now defaults to `"chain"`** — explicit segments we
  control end-to-end, each getting its own distinct prompt. `sliding_window`
  is still available but now documented as best-effort only, reserved for
  driftlike atmospheric takes with no discrete action (where "continuing"
  and "repeating" look the same).
* **Progressive phase authoring** (`plan_long_take_phases` in
  cinematography.py) — any shot needing multiple segments gets its single
  action broken into genuinely progressive beats by an LLM pass explicitly
  told to write it "like consecutive sentences of one unbroken commentary,
  never separate replays of the same moment": segment 1 might be "grips the
  handle and pulls the door open," segment 2 "steps through the threshold as
  the door slams behind her." Dialogue lines are assigned to the ONE phase
  where they're actually spoken, never repeated across segments.
* **Fixed prompt reuse in chain rendering** — `_render_chain_group` now uses
  each segment's own phase-mapped prompt instead of the shot's single prompt
  duplicated across every segment. The chained anchor itself also gained an
  explicit instruction: *"do not restart, repeat, or re-loop any action
  already completed... advance FORWARD, never back to its beginning."*
  Older plans without authored phases still get a same-text fallback, but
  now with that anti-restart instruction and a phase counter per segment
  rather than pure duplication — with a logged suggestion to clear the
  `shot_design` planning-cache stage and re-run for the full fix.

**For your current lighthouse_portal project:** delete the `"shot_design"`
key from `planning_cache.json` and re-run `plan_film` to author phases
properly (everything else resumes from cache); or just re-render as-is to
get the improved fallback phrasing immediately.

## Reading the round schedule (v4.3)

Round 1 rendering scene 1 then jumping to scene 4 is CORRECT, not a skip:
consecutive scenes sharing a location form ONE continuity block, and round
1 renders only each block's opening shot (the only shots with synthesized
stills). If scenes 1-3 share a setting, their block spans all three —
scene 2's and 3's shots are later positions in that chain and render in
later rounds, each continuing from its predecessor's last frame (which is
the whole cohesion mechanism). A **block map** is now logged at render
start showing exactly which scenes share each block and which rounds their
shots land in, and `render_status.json` carries per-shot `block` and
`round` fields.

## Stills that aren't block openers (v4.4)

If `stills/` contains images for shots that never render from a still
(e.g. `image_02_01.jpg` when scene 2 chains from scene 1), one of two
things is true:

1. **Leftovers from the pre-chaining architecture**, which synthesized a
   still for every shot. Current runs only generate block openers, but
   idempotent skipping means old files stay on disk. Clean them up with
   `prune_unused_stills(plan_path, FILM_CONFIG)` — moves non-opener stills
   (and their variants) to `stills/_unused/` (`delete=True` removes them).
2. **They ARE openers** — the scene's location differs from its neighbor's,
   making it its own block. The produce log now prints the exact opener
   list so this is checkable at a glance. Note also that long-take oners
   in chain mode render as separate `chain_*` wgp calls AFTER a round's
   batch, so a round-1 batch manifest legitimately skips their task
   numbers.

## Revised: scene = setting = continuity unit (v4.5)

The continuity block is now the SCENE itself, matching the intended model:
one scene = one specific setting = one synthesized opening still; a change
of setting requires a scene change, which is a cut to a fresh still; and
within a scene (arbitrarily long) every shot chains from its predecessor's
last sharp frame. Round-1 task order now equals scene order — task 2 is
scene 2's opener with image 2, always.

The previous behavior (consecutive scenes sharing a location fused into one
chain, skipping the later scene's still) is demoted to the opt-in
`chain_across_scenes: True` — it saved a still but made the schedule
illegible, and cross-scene physical continuity is already handled the right
way: the state ledger carries costume/damage/prop state textually into the
next scene's opener still prompt. Empty/missing locations never merge even
with the flag on.

Rendering logs now also enumerate, per round, which shots are in the batch
manifest vs which long takes render AFTER it as separate `chain_*` calls —
so nothing ever again looks "skipped."

## Narration & emotional voice (v4.6)

Fixes the voiceless-scenes problem and builds the narration layer properly:

* **Voice is now the default** — the old "some shots should be silent" rule
  (which the LLM over-obeyed) is replaced: every scene needs a voice —
  dialogue where characters share the frame, narration where a character is
  alone or the moment is internal. Total silence is rationed craft (at most
  one fully silent shot per scene), never a habit.
* **A designed narrator persona** (`design_narrator`) — before any scene is
  written, the film gets a narrator with a specific relationship to the
  story (e.g. the protagonist remembering it years later) and a voice
  texture a voice director could cast. `narration: "auto" | "rich" | "off"`
  and free-text `narration_style` in config.
* **Narration weave** (`weave_narration`) — a post-pass that walks the
  authored film and inserts NARRATOR beats where they earn their place:
  a MANDATORY opening orientation (who/where/what's at stake, planting the
  central question), interiority at key emotional moments — *the quiet part
  said out loud*, what a character thinks but would never say — coverage
  for any scene with no voice at all, and a MANDATORY closing bookend that
  answers the opening. Every addition fits inside the shot's remaining
  spoken-word budget (never crowding lip-synced dialogue timing), verified
  by code-level clamping.
* **Emotion-matched delivery, enforced everywhere** — every line, dialogue
  and narration alike, must carry a cue naming the emotional color AND
  vocal quality: private/sensual thoughts as an intimate near-whisper,
  anger hot and driving, grief slow and weighted, wonder hushed. The polish
  pass rewrites flat cues; the narrator's voice chains like any character's
  (its first solo shot establishes the banked voice).
* **Voice coverage audit** — after the screenplay is written, a
  deterministic report logs voiced-shot percentage, voiceless scenes,
  flat/missing delivery cues, and whether the film opens and closes with
  narration; the table read gained a matching editorial audit item.

To apply to an existing plan: delete the `"screenplay"` and `"shot_design"`
keys from `planning_cache.json` and re-run `plan_film` (story, graph, and
style stages resume from cache).

## Designed silence — atmospheric scenes (v4.7)

Refinement of the voice layer: some scenes SHOULD be wordless — relief on
breaking waves after a peak, a pure-action surge, a held closeup of a face
in grief where any word would be smaller than the face. The fix isn't a
looser rule; it's making silence a DECISION:

* **Voice plan** (`design_voice_plan`) — before authoring, each scene is
  deliberately designated `voiced` or `atmospheric` (rationed ~1 per 5
  scenes, placed where the emotional journey needs breath; the first and
  last scenes always stay voiced so the opening orientation and closing
  bookend narration have a home). The designation and its reasoning land
  in plan.json (`voice_design`, `voice_note`) — hand-editable like
  everything else.
* **Everything respects it**: atmospheric scenes are authored with NO
  dialogue and NO narration, the emotion written into bodies and faces;
  the narration weave skips them; and their **soundscape is composed as
  the scene's voice** — richer and more specific than a voiced scene's,
  with an emotional arc across the shots that lands what words would have
  carried.
* **The audit distinguishes intent from accident**: designed-silent scenes
  are reported as such (no warning); a voiceless scene *without* the
  designation is flagged as a gap — either mark it atmospheric in
  plan.json or give it a voice.

## Fixed: true single-window limit is 362 frames (v4.8)

Render logs revealed the fork's actual H3 sliding-window size is **362
frames (~15.1s @ 24fps)**, not the 737 (~30s) previously assumed — so
ordinary 15–30s shots were being silently split into internal sliding
windows ("2 Windows will be generated"), reintroducing the
action-restart risk on shots meant to be single-window.

Corrected defaults: `max_shot_seconds: 15.0`, `wgp_frames_maximum` /
`wgp_video_length_max_frames: 362` (exactly on the 5+17k frame grid),
`cut_target_low_seconds: 13.0`. Every normal shot now fits ONE window;
anything longer routes to chain mode with progressive per-segment
prompts; a 41s oner becomes 3 chained segments of ≤362 frames, and
planning-time phase counts align with render-time segment counts
(verified).

**Applying to an existing plan:** shot durations are set at the
screenplay stage, so clear `"screenplay"` and `"shot_design"` from
planning_cache.json and re-run `plan_film` for shots authored to the
15s rhythm (recommended). Re-producing without replanning also works —
any >15s shot auto-routes to chain mode with the anti-restart fallback
prompts — but re-planned shots with authored phases are better. Shots
already rendered with "2 Windows" in the log are the ones to spot-check
for restarts and re-roll.

## Efficiency: lockstep chain rendering + call forecast (v4.9)

With the 362-frame window, long takes segment more — so segment rendering
efficiency now matters. Two structural wins:

* **Segment-0 rides in the round's main batch.** A chain long take's first
  segment has no predecessor dependency, so rendering it as its own wgp
  call was pure wasted model-startup time. It now joins the round's batch
  manifest alongside normal shots.
* **Lockstep segment waves.** Segments of *different* long takes depend
  only on their own predecessors, so continuation segments render one wgp
  call per SEGMENT INDEX across ALL long takes (`round1_chainseg2` holds
  seg-1 of every oner), instead of one call per segment per take. Two
  3-segment oners in a round: was ~6 wgp invocations, now 3 (verified by
  test). Frame/tail-audio extraction still happens per group between
  waves, so continuity is unchanged.
* **Call forecast** — render start now logs `~N wgp.py call(s) forecast`
  so a batch's wall-clock cost is predictable before it starts, and
  plan.md tags designed-silence scenes (`· designed silence`) for review.

## v5.0 — windowed long takes, identity anchoring, behavioral idiom

Three changes: a render bug fix, a new default long-take mode built on the
fork's per-window prompting + multi-frame overlap, and a behavioral-idiom
layer that gives every character a physical life (and funny characters a
working comedy machine).

### Fixed: chained shots silently skipped on fresh projects

`build_shot_jobs` carried a duplicated, unconditional still-existence
check left over from pre-v4.5 (when every shot had its own still). Under
v4.5 semantics only **block openers** get synthesized stills, so on a
fresh project every chained shot hit the second check and was skipped
with a "missing still" warning — the film rendered as openers only.
Chained shots now skip the disk check entirely (their reference frame
resolves at round time) and their fallback still points at the **block
opener's** still, the only one guaranteed to exist.

### Long takes: `long_take_mode: "windowed"` (new default)

Chain mode fixed the action-restart problem with per-segment progressive
prompts, but paid for it at every join: a single static handoff frame
carries **no motion vectors and no audio phase**, so velocity hitches and
ambience seams live exactly at segment boundaries. The fork's sliding
windows carry **multi-frame overlap** (motion + audio context) across the
join *and* accept **one prompt line per window** — both halves of the
solution at once:

* One task per long take (it rides the round's main batch — no lockstep
  segment waves, fewer wgp calls), with `sliding_window_size` /
  `sliding_window_overlap` set from config and the per-window prompt
  switch enabled (`wgp_windowed_extra_task_fields`, default
  `{"multi_prompts_gen_type": 1}`).
* The prompt is one **line per window**, joined by
  `wgp_windowed_prompt_separator`: each window gets its own authored
  progressive phase (window 2 describes what happens *next*, with its own
  dialogue lines, never re-describing window 1) plus an explicit
  "part i of N … never restart" continuation clause.
* `plan_long_take_phases` now authors **one phase per window** via
  `estimate_phase_count()`, which accounts for overlap (window count ≠
  chain segment count for the same duration). Older plans degrade
  gracefully: phases spread proportionally, unplaced lines land in
  window 1, and a warning tells you which cache stage to clear.
* Field names are fork-dependent knobs — verify once with
  `wgp_dry_run=True` against a UI "Export Settings" JSON, exactly like
  the chain-mode fields. `"chain"` remains the proven fallback and
  `"sliding_window"` remains for pure atmospheric drift.

### Identity re-anchoring (`chain_identity_anchor: True`)

The most visible artifact of long chains is identity drift — faces
morphing copy-of-a-copy by shot 8. Chained shots now send a **second
Ref2VA reference** alongside the handoff frame: the block opener's still,
i.e. the scene's canonical appearance. The prompt tells the model the
extra reference is the scene's opening frame and identity must keep
matching it. The fork preserves reference memories across sliding
windows, so the anchor holds through windowed long takes too.

### Behavioral idiom — character is what a body *does*

The EI graph now builds each character's **behavioral idiom**: 1–3
passions and a signature **idle behavior** derived from them (the
music-lover drums paradiddles on whatever's nearest), a **sonic
signature** (their presence has a sound — it can announce them off-screen,
and its absence can mean something's wrong), a visible **competence**,
a **social energy** rating (−5 edges-of-rooms … +5 works-the-room), and —
for funny characters — a **comic toolkit** (modes, emphasis habit, timing,
their own running gag). It flows everywhere the older EI fields flow:

* **Scene staging** — `behavior_block()` directs idle behavior *before*
  generic gestures, and makes it modulate (quickens under stress, stops
  dead at a shock); comic timing is written into playable delivery cues
  (emphasis in the cue, never CAPS in the line; punchlines land on the
  last word; sarcasm aims along ToM edges; the shot after a punchline
  opens on a silent reaction). Dyads with a social-energy gap get
  **rhythm choreography** (who initiates, who mirrors, who breaks eye
  contact first).
* **Stills** — idle behavior and competence are staged mid-motion in the
  frame; social energy shapes placement (room's center vs. edges).
* **Sound** — sonic signatures and musical idle behaviors are offered to
  the motif designer as **leitmotif candidates**: a character's whistled
  tune planted early, transformed by emotional state, echoed back by
  someone else at the resolution.
* **Anti-slop** — `behavior_grammar.py` injects a per-scene **behavior
  palette** (concrete stageable options per emotion, contained vs.
  expressive columns matched to each character's emotion style), and a
  deterministic post-pass audits gesture convergence across scenes
  (stock gestures + repeated action phrasing), logged and appended to
  `table_read.md`. Knobs: `behavior_grammar`, `behavior_repetition_max`.

Existing plans keep working (old graphs deserialize with empty idiom
fields). To adopt the new layer on an existing project, clear the
`graph`, `screenplay`, and `shot_design` stages from
`planning_cache.json` (or re-plan); to adopt windowed phase counts alone,
clearing `shot_design` is enough.

## v5.1 — fixed: characters mouthing the narrator's lines

Narrated shots could show a visible character lip-syncing the narration
in something like their own voice. The old prompt used ad-hoc phrasing
("an unseen narrator's voice speaks from off-screen, no visible mouth
moving") — but MiniMax's H3 prompting guidance specifies an exact
two-part voiceover syntax, and both parts were missing:

1. The lead must use the **exact phrase** `says in an off-screen
   voiceover`.
2. **Immediately after** the `<d>` block, the prompt must state that the
   visible characters' lips remain completely closed. Without this
   trailing clause H3 reliably animates an on-screen mouth to the
   narration.

Narrator lines now render as:

```
An unseen narrator, <voice_texture>, (S1) says in an off-screen
voiceover, <delivery>: <d>[English] ...</d> while every visible
character's lips remain completely closed.
```

Three details beyond the required syntax:

* The **designed narrator voice** finally reaches the model: the first
  narrated line of each shot carries `story["narrator"]["voice_texture"]`
  (from `design_narrator`), so H3 synthesizes a distinct off-screen
  timbre instead of borrowing an on-screen character's voice. Later
  lines in the same shot stay short but keep the exact voiceover phrase,
  and every narrator block gets its own lips-closed clause.
* The **word-budget trimmer** now protects the lips-closed clause the
  same way it protects `<d>` spans — the clause sits right after a
  `</d>` at the prompt's tail, exactly where tail-first trimming used to
  eat it on long prompts, which would have silently reintroduced the bug.
* The fix flows through every prompt path automatically: single shots,
  chain segments, and windowed long takes (where the clause survives the
  one-line-per-window flattening).

No plan changes needed — prompts are assembled at render time, so
existing projects pick the fix up on their next render.

## v6.0 — scenes render as ONE continuous generation (chaining removed)

Last-frame chaining is **gone**. Extracting a clip's final sharp frame and
feeding it back as the next shot's start was the pipeline's main quality
leak — a copy-of-a-copy step that degraded every hop — and the fork no
longer needs it: WanGP's MiniMax H3 builds long, multi-shot videos
natively. The render unit is now the **scene**.

### How a scene renders now

One wgp task per scene, directed **window by window** through the fork's
sliding-window engine using its inline commands (`docs/PROMPTS.md`):

```
[/duration=10s] <shot 1 prompt>
[/duration=13.3s,/new_shot] <long-take phase 1>
[/duration=13.3s,/overlap=17] <phase 2 — continues forward, never restarts>
[/duration=13.3s,/overlap=17] <phase 3>
[/duration=8s,/new_shot] <shot 3 prompt>
```

* **Every shot boundary is a native `[/new_shot]` hard cut** — the cut
  happens *inside* the model with full identity memory, not by
  concatenating separately rendered clips.
* **Long takes span windows with `[/overlap=N]`** — multi-frame overlap
  hands real motion and phase-continuous audio across the join, each
  window carrying its own authored progressive phase.
* **Identity is held by the model**: the scene's Z-Image still is both
  the **Start Image** (pins frame 1) and a **persistent Ref2VA
  reference** that the fork keeps across every window. Nothing drifts
  copy-of-a-copy, because nothing is copied.
* Audio is one continuous H3 track per scene — no more per-shot seams.

Every scene needs (at most) **one** synthesized still: its opener.
Cross-scene continuity remains textual via the physical-state ledger,
exactly as before. Voice consistency across scenes keeps the two-wave
scheduler: wave 1 renders silent scenes plus each character's first
solo-voiced scene, banks those voices, wave 2 attaches them.

### Optional: skip Z-Image entirely

`use_reference_stills: False` removes the stills stage completely. The
scene's first window then opens with a detailed **identity-lock block**
(each character's face/hair/build description, wardrobe, distinctive
props, and the location's look — the same anchors a Z-Image prompt would
carry, stated verbally), and the model's cross-window memory holds it.
Cheaper and simpler; identity is typically a touch less pinned than with
a reference still. `scene_start_image: False` keeps the still as a
reference only, without pinning frame 1.

### What was removed

`scene_continuity_chaining`, `chain_across_scenes`, `chain_frame_samples`,
`chain_identity_anchor`, `long_take_mode`, `enable_long_takes`, the
`wgp_chain_*` fields, the continuity-rounds scheduler, sharp-frame
handoff, and chain-segment rendering. `regenerate_shots` now re-renders
the containing scene (the render unit); clips are `clips/scene_NNN.mp4`.
Verify the fork-dependent field names once with `wgp_dry_run=True`
(`wgp_scene_start_image_field`, `wgp_windowed_*`) — the familiar ritual.

## v6.1 — fixed: stills-free run crashed WanGP's prompt-template parser

First stills-free production run failed with `Error processing prompt
template: Unknown variable ... Skipping` on every task. Three fixes:

* **Braces never reach the manifest.** WanGP parses `{...}` in prompts as
  template variables (its macro system) and skips any task containing an
  unknown one. All scene prompts now convert `{`/`}` to parentheses;
  `[...]` is untouched (only `[/...]` is a command).
* **The identity lock now uses the machinery built for it**: each
  character's `visual_lock` (the canonical 60-100 word description
  written to be reproduced verbatim in Z-Image prompts) plus
  `wardrobe_for(name, scene_id)` — the scene-current wardrobe state as a
  string. Previously it dumped the raw wardrobe evolution list (Python
  dicts, i.e. braces) into the prompt, which is what tripped the parser.
* **Voice banking survives the narrator.** Establishing used to require
  a solo-speaker scene; the narrator speaks nearly everywhere, so no
  voice was ever banked (`wave 1 = 0`). A scene now establishes every
  voice heard there for the first time (a mixed clip is a valid H3
  reference — it locks every (S#) voice in it), so the narrator and
  leads bank from scene 1 and stay consistent across the film.

## v6.2 — RAM cap: long scenes split into task parts at shot boundaries

One-task-per-scene made some tasks 1,500-2,500 frames; WanGP holds a
task's entire output (frames, latents, H3 audio) in memory until it
finishes, which exhausted system RAM (WSL2 + notebook unresponsive at
96%+ memory, GPU starved by paging into shared memory). Scenes longer
than `scene_max_task_seconds` (default 45s) now split into consecutive
task parts **at shot boundaries** — a hard cut carries no motion or
audio continuity, so parts joined at cuts lose nothing and mid-take
stitching stays gone. Part 1 uses the scene still as Start Image +
reference; later parts (which open on a cut) carry it as an identity
reference only. Assembly concatenates parts in order. Single shots
longer than the cap stay whole with a warning. If RAM still pegs, lower
the cap, lower `wgp_perc_reserved_mem_max`, and check your `.wslconfig`
memory ceiling (see the v4.1 section) so Windows keeps headroom.

## v6.3 — fixed: monitor KeyError stacked duplicate wgp processes

The v6.2 run failed with `Error running wgp.py: 'target_seconds'`: the
render monitor still read a per-shot field scene jobs didn't carry. The
KeyError killed the monitoring loop while wgp kept rendering, every task
was marked "no output parsed," and the retry pass launched a SECOND wgp
on top of the live one — doubling memory pressure (98% RAM) with no
progress ever recorded. Fixed threefold: scene jobs carry
`target_seconds`; both call sites fall back to `duration`; and if the
monitor ever dies mid-run it now kills the wgp child before retrying, so
duplicates can never stack again. Before rerunning after any crashed
wave, check for strays: `pkill -f wgp.py`.

## v6.4 — fixed: wave 2 rejected (visual refs must be >= audio refs)

Wave 2 failed validation on every task in stills-free mode: `MiniMax H3
requires at least as many reference images and videos as audio
references (found 0 visual and 1 audio)` — each reference voice must
bind to a reference face. Voice banking now harvests a companion FRAME
of each speaker from the same establishing clip (voice_bank/<name>.png
beside the wav), and stills-free tasks attach those frames as the
visual references paired with the voice — face and voice from the same
rendered footage, which also gives later scenes a rendered-footage
identity anchor. The scene still (when present) already satisfied the
rule and still exclusively pins frame 1. If no frame is available for a
voice, the audio ref is dropped for that task (valid text-only task)
rather than tripping validation. Resumed projects backfill missing
frames from already-rendered clips automatically.

## v6.5 — `find_plan()`: produce without re-planning

The plan path is now derived from config, so a restarted notebook never
needs `plan_film()` just to recover the path:

```python
final = produce_film(cfg=FILM_CONFIG)                 # auto-finds plan
final = produce_film(find_plan(FILM_CONFIG), FILM_CONFIG)   # explicit
regenerate_scenes(cfg=FILM_CONFIG, scene_ids=[4, 11])       # also works
```

`find_plan(cfg)` returns `base_dir/<slug(project_name)>/plan.json`
(most-recent plan under `base_dir` if no project_name), with a clear
error if no plan exists yet. `plan_path` is now optional on
`produce_film`, `regenerate_scenes`, `regenerate_shots`, and
`prune_unused_stills`.

### v6.5 addendum — `plan_film` is now idempotent

With a `project_name` set and a finished `plan.json` on disk,
`plan_film(idea, cfg)` returns the existing plan path immediately (zero
LLM calls) when the creative brief is unchanged — so a notebook that
runs the plan cell then the produce cell works safely top-to-bottom on
every restart. If the idea/config fingerprint *changed*, it raises
instead of silently invalidating the cache and re-planning over your
finished film (previously a reworded idea string did exactly that).
Re-plan deliberately with `resume=False`, or start a new film with a
new `project_name`.
