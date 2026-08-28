"""
film_maker.config
=================
One dict the notebook edits; everything else reads from it.

H3 / Wan2GP settings are ported directly from the proven music-video
generator (a 24GB-class card such as a 4090; system RAM varies by machine
-- verify yours with `free -h` and see wgp_perc_reserved_mem_max below
before tuning memory flags), same conda-wrapped
wgp.py --process batch call, same 5+17k frame grid). The film-specific
additions:

  * registers                 — genre/register list ("comedy", "horror", ...)
                                driving every artistic-method layer.
  * target_minutes            — total film length; the pacing engine sizes
                                scene and shot counts from it.
  * voice_chaining            — establish each character's voice once (first
                                clean solo shot), then feed that clip as the
                                H3 audio reference on every later shot they
                                lead. Rendering happens in WAVES so the
                                establishing clips exist before dependents.
  * image_variants            — stills per shot; a vision pass picks the best.
  * keep_h3_audio             — the whole point: H3's generated audio IS the
                                film's audio (dialogue + diegetic sound).
                                non_diegetic_music is always N/A; you score
                                the film in post.
"""
import os

FILM_CONFIG = {
    # ------------------------------------------------------------------ LLM
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "openai_model": "gpt-5.4-nano",
    "openai_model_large": "gpt-5.4-nano",
    "grok_api_key": os.getenv("XAI_API_KEY", ""),
    "grok_model": "grok-4.3",
    "use_grok": False,
    "llm_retry_limit": 3,
    # Completion budgets. On reasoning models the completion budget also
    # covers reasoning tokens, so these sit well above the expected visible
    # output. When a response still comes back truncated (finish_reason ==
    # 'length'), get_llm automatically retries with a DOUBLED budget up to
    # llm_truncation_retry_cap before accepting a partial.
    "llm_max_completion_tokens": 60000,
    "llm_max_completion_tokens_large": 100000,
    "llm_truncation_retry_cap": 160000,
    # Bulk per-item calls (one JSON object per shot/prompt) are batched so a
    # single response can never grow past what any model returns reliably;
    # missing items are re-requested in repair passes.
    "shot_plan_batch_size": 36,
    "scrub_batch_size": 30,
    # Vision model used ONLY for picking the best of N reference stills.
    # Any OpenAI-compatible vision chat model; falls back to seed 0 if empty
    # or if the call fails.
    "vision_model": "gpt-5.4-nano",

    # --------------------------------------------------------------- Project
    # Set a project name to keep everything for one film -- plan, cache,
    # stills, voice bank, clips, final -- under a single folder pair:
    #   <base_working_dir>/<project_name>/  and  <base_output_dir>/<project_name>/
    # Different names = fully independent films side by side. Left empty,
    # the folder is derived from the generated film title.
    "project_name": "",

    # --------------------------------------------------------------- Story
    # "story" (default) — the idea is a story seed.
    # "educational" — the idea is a TOPIC to teach ("how vaccines train the
    #   immune system"). A pedagogy spec (learning objective, one core
    #   analogy, the misconception to break, a key-idea ladder) is built
    #   first, the explainer register is auto-added, and the eureka ladder
    #   carries the concept's click as the film's climax.
    "film_mode": "story",
    # Ordered list; first entry is the PRIMARY register, later entries blend.
    # Available keys: see film_maker.registers.REGISTERS
    "registers": ["drama"],
    # Free-text tonal guidance layered on top of the registers (optional).
    "tone_notes": "",
    # Content register: "all_ages" | "teen" | "mature". Mature unlocks the
    # sensual/intense artistic-method layers; all_ages keeps everything clean.
    "content_rating": "teen",
    "target_minutes": 16.0,
    "cast_size": "small",          # small | medium  (H3 identity limits favor small)
    "language": "English",         # goes inside every <d>[English] ...</d> tag
    # ---- Style layers (preset key or free text; empty = LLM's choice) ----
    # Presets in film_maker.styles: VISUAL_STYLES / CAMERA_STYLES /
    # DIALOGUE_STYLES. Free text is used verbatim.
    "visual_style": "",     # "watercolor" | "graphic_novel" | "photorealistic"
                            # | "cgi" | "anime" | "oil_painting" | "stop_motion"
                            # | "noir_bw" | "storybook" | free text
    "camera_style": "",     # "classical" | "handheld" | "epic" | "tableau"
                            # | "kinetic" | "floating" | free text
    "dialogue_style": "vernacular",
                            # "vernacular" (Twain-craft dialect) | "spare" |
                            # "rapid_wit" | "naturalistic" | "lyrical" |
                            # "hardboiled" | free text
    # ---- Narration -------------------------------------------------------
    # "auto" (default): a designed narrator persona opens the film with
    #   orientation narration (who/where/what's at stake), voices
    #   interiority at key emotional beats ("the quiet part out loud" --
    #   what a character is thinking/feeling, delivered in the emotion's
    #   own register: intimate and low for private thoughts, hot for anger,
    #   slow for grief), covers scenes where no one has a scene partner,
    #   and closes with a bookend reflection.
    # "rich": the same, woven through most scenes -- a strongly narrated
    #   film (documentary/fable/noir texture).
    # "off": no narrator; dialogue only.
    "narration": "auto",
    # Optional free-text persona override, e.g. "first-person retrospective,
    # the keeper herself years later, weathered and wry" or "omniscient,
    # warm, David Attenborough-like wonder". Empty = designed from the
    # registers and story.
    "narration_style": "",

    # ---------------------------------------------------------- Screenplay
    # Pacing engine bounds (seconds per shot). Same clamp as the music
    # pipeline: H3 single-window render range.
    "min_shot_seconds": 5.0,
    # One H3 window in this Wan2GP fork is 362 frames (~15.1s @ 24fps) --
    # confirmed from render logs ("Sliding Window Size (362)"). Every normal
    # shot must fit ONE window, or wgp silently splits it into sliding
    # windows and the action can restart at each window boundary. Anything
    # longer than this becomes a long take, spanning several windows of
    # its scene's generation with progressive per-window prompts.
    "max_shot_seconds": 15.0,
    # Target shot length per tension band; register pacing curves modulate.
    "cut_target_low_seconds": 13.0,
    "cut_target_mid_seconds": 11.0,
    "cut_target_high_seconds": 6.5,
    # Max spoken words per shot (H3 lip-sync stays clean with short lines;
    # ~2.5 words/sec of screen time is the physical ceiling).
    "max_dialogue_words_per_10s": 22,
    "dialogue_repetition_window": 6,     # shots
    "dialogue_repetition_threshold": 0.72,

    # --------------------------------------------------------------- Stills
    "image_model_repo": "ykarout/Z-Image-Turbo-FP8-Full",
    "hf_token": os.getenv("HF_TOKEN", ""),
    "image_variants": 4,           # per shot; vision pass picks the best
    "image_steps": 9,
    "h3_width": 1280,
    "h3_height": 704,
    "h3_fps": 24.0,

    # ---- Scene rendering (v6: one continuous generation per scene) -------
    # The render unit is the SCENE. Each scene renders as ONE H3 task
    # through Wan2GP's sliding-window engine, directed window by window:
    # one prompt line per window, a native [/duration=Xs,/new_shot] hard
    # cut at every shot boundary, and [/overlap=N] continuations inside
    # long takes (multi-frame overlap carries real motion + audio context
    # across the join). Identity is held by the model's own reference
    # memories, which persist across every window. LAST-FRAME CHAINING IS
    # GONE: no frame is ever extracted from one clip to seed another --
    # that copy-of-a-copy step was the main quality leak.
    #
    # Each scene starts from ONE synthesized Z-Image still: its Start
    # Image (pins the opening frame) and its persistent Ref2VA identity
    # reference. Cross-scene continuity stays textual (the physical-state
    # ledger writes carried state into the next scene's still prompt).
    "use_reference_stills": True,
    # False = skip Z-Image ENTIRELY: no stills stage, no image
    # conditioning. Identity then rides in the prompt -- a detailed
    # identity-lock block (faces, hair, build, wardrobe, location look)
    # leads the scene's first window, the same way it would lead a
    # Z-Image prompt. Cheaper and simpler; visual identity is typically a
    # touch less pinned than with a reference still. Try both.
    "scene_start_image": True,   # also pin frame 1 to the still (Start
                                 # Image), not just use it as a reference
    "wgp_scene_start_image_field": "image_start",       # fork-dependent
    "wgp_scene_start_image_prompt_type": "S",           # fork-dependent
    "wgp_video_prompt_type_textonly": "",  # stills-free video_prompt_type

    # ---- Behavioral idiom & behavior grammar (anti-slop, performance) ----
    # Each character's BEHAVIORAL IDIOM (passions -> signature idle
    # behavior, sonic signature, visible competence, social energy, comic
    # toolkit) flows into scene staging, stills, and sound motifs;
    # behavior_grammar adds a per-scene BEHAVIOR PALETTE and a
    # deterministic gesture-repetition audit (appended to table_read.md).
    "behavior_grammar": True,
    "behavior_repetition_max": 2,

    # ------------------------------------------------------------ Rendering
    "keep_h3_audio": True,
    "voice_chaining": True,        # audio-ref voice locking (disable to skip waves)
    # ---- Long takes ------------------------------------------------------
    # A shot longer than one window simply spans several windows of its
    # scene's generation, each window carrying its own authored
    # progressive PHASE (plan_long_take_phases) and an [/overlap] join.
    "long_take_max_seconds": 60.0,
    # RAM ceiling per wgp task: WanGP holds a task's whole output video
    # (frames + latents + H3 audio) in memory until it finishes, so
    # scenes longer than this split into consecutive task PARTS at shot
    # boundaries (a hard cut carries no continuity, so nothing is lost
    # and mid-take stitching stays gone). ~45s = ~1100 frames is a safe
    # ceiling for 24GB VRAM + 128GB RAM under WSL2; raise it on bigger
    # boxes, lower it if system RAM still pegs at 95%+ while rendering.
    "scene_max_task_seconds": 45.0,
    # ---- Dialogue fit + scene buttons ------------------------------------
    # Shot durations are FLOORED at render time to the performed speech
    # time (words at speech_words_per_second + a turn gap per line) plus
    # settle room, so no window ends mid-sentence. The last shot of each
    # scene additionally holds a designed closing BEAT after the final
    # line -- an emotion-matched button (held glance, thinning sound,
    # decisive nod) written into the prompt -- so scenes land instead of
    # cutting off.
    "speech_words_per_second": 2.3,
    "dialogue_turn_gap_seconds": 0.6,
    "dialogue_settle_seconds": 1.2,
    "scene_button": True,
    # Hook architecture (planning): one LLM pass gives every scene an
    # ENTRY hook (open mid-motion), an authored EXIT hook (replaces the
    # stock emotion button) and the OPEN QUESTION it leaves hanging --
    # chained so questions escalate by act and pay off at the finale.
    "hook_design": True,
    # Voice reference mode: "lead" (default) sends ONE clean banked wav
    # (the part's lead speaker) -- a concatenated multi-voice wav
    # switches speakers mid-stream and reliably garbles H3's speech into
    # gibberish. "combo" restores the old concatenation. If any garble
    # persists, lower scene_max_task_seconds so every scene fits one
    # model run with fewer windows.
    "voice_ref_mode": "lead",
    "scene_button_seconds": 2.5,
    # ---- Sliding-window knobs (fork-dependent; verify once w/ dry run) ---
    # Window size in frames; empty/None = wgp_video_length_max_frames.
    "wgp_windowed_window_frames": None,
    # Multi-frame overlap at long-take continuation joins: the frames of
    # shared motion + audio context that make the join smooth. One grid
    # step (17 = ~0.7s) is a solid starting point.
    "wgp_windowed_overlap_frames": 17,
    # Task-dict field names carrying window size/overlap in your fork:
    "wgp_windowed_size_field": "sliding_window_size",
    "wgp_windowed_overlap_field": "sliding_window_overlap",
    # Separator between per-window prompt lines (the fork consumes one
    # line per sliding window):
    "wgp_windowed_prompt_separator": "\n",
    # Extra task fields switching the fork into one-prompt-line-per-window
    # mode ("Each Line Will be used for a new Sliding Window"); adjust
    # after the one-time dry-run verification if named differently.
    "wgp_windowed_extra_task_fields": {"multi_prompts_gen_type": 1},
    # Auto-planned oners per film: the pacing engine promotes up to this many
    # low-tension scenes (where the register's craft favors a held take) to a
    # single continuous long take. You can also just edit any shot's
    # "duration" past 30 in plan.json and the renderer handles it.
    "long_take_budget": 3,
    "wgp_repo_dir": os.path.expanduser("~/repos/Wan2GP"),
    "wgp_python_executable": "python",
    "wgp_conda_env": "wan2gp",
    "wgp_conda_base_env": "base",
    "wgp_script": "wgp.py",
    "wgp_model_type": "minimax_h3_ref2va",
    "wgp_video_prompt_type": "KI",
    "wgp_num_inference_steps": 8,      # 8 with the Turbo LoRA, ~20 without
    "wgp_frames_offset": 5,
    "wgp_frames_steps": 17,
    "wgp_frames_minimum": 107,
    # 362 = the fork's true single-window maximum (5 + 17*21, on the frame
    # grid; ~15.08s @ 24fps). Shots at or under this render in exactly one
    # window; longer targets span several windows of the scene generation.
    "wgp_frames_maximum": 362,
    "wgp_video_length_max_frames": 362,
    "wgp_audio_ref_min_seconds": 2.0,
    "wgp_audio_ref_max_seconds": 15.0,
    "wgp_prompt_max_words": 350,
    "wgp_extra_cli_args": [],   # extra raw flags, appended AFTER the ones below
    # ---- mmgp memory profile tuning ---------------------------------------
    # These build the --profile / --perc-reserved-mem-max flags for you (set
    # wgp_extra_cli_args above for anything beyond these two).
    #   wgp_profile: 1 HighRAM_HighVRAM (no VRAM cap, wants ~48GB+ RAM) |
    #     2 HighRAM_LowVRAM (pin everything reservable, cap VRAM -- best for
    #     a 24GB card with plenty of RAM) | 3 LowRAM_HighVRAM | 4 LowRAM_LowVRAM
    #     (pin only the transformer, extra quantization -- use this on a
    #     RAM-constrained box instead of pushing perc_reserved_mem_max to its
    #     limit) | 5 VeryLowRAM_LowVRAM (no pinning, slowest, last resort).
    "wgp_profile": "2",
    # Fraction of TOTAL system RAM mmgp is allowed to reserve for pinning.
    # mmgp logs the exact numbers on every run ("full requirements for
    # pinned models is X MB while estimated available reservable RAM is Y
    # MB at the current perc") -- read Y/perc_current to get RAM at 100%,
    # then required_perc = X / (that 100% figure). Only raise this if
    # required_perc leaves comfortable headroom (~15-20%) for the OS, ffmpeg,
    # and this notebook process; if required_perc is above ~0.85, don't push
    # perc higher -- switch wgp_model_type to the pruned checkpoint instead,
    # lower wgp_profile to 4, or increase actual/WSL RAM. Partial pinning
    # (the default 0.5) is NOT broken -- it only slows RAM<->VRAM transfer
    # during model load, with zero effect on output quality.
    "wgp_perc_reserved_mem_max": "0.5",
    "wgp_dry_run": False,
    "wgp_call_timeout_seconds": 14400,   # 15+ min films = long batches
    "wgp_max_render_attempts": 3,
    "wgp_retry_sleep_seconds": 20,
    "wgp_env_overrides": {"MPLBACKEND": "Agg"},
    "wgp_env_unset": [],
    "wgp_cleanup_intermediate_window_files": True,
    # Turbo LoRA (same caveat as the music pipeline: published against fl2va)
    "wgp_use_turbo_lora": True,
    "wgp_turbo_lora_repo": "larryvrh/MiniMax-H3-Turbo-Lora",
    "wgp_turbo_lora_filename": "minimax_h3_turbo_4step_ema_ckpt850.safetensors",
    "wgp_lora_subdir": os.path.join("loras", "minimax_h3"),
    "wgp_lora_search_dirs": ["loras"],
    "wgp_lora_multiplier": "1.0",

    # ------------------------------------------------------------- Assembly
    "concat_final_video": True,
    "keep_shot_clips": True,
    # "cut" (default — H3 clips carry their own sound; hard cuts read as
    # editing) or "crossfade" (0.4s A/V crossfade at SCENE boundaries only).
    "scene_transition": "cut",
    "crossfade_seconds": 0.4,

    # ------------------------------------------------------------ Locations
    # ONE self-contained folder per film:
    #   <base_dir>/<project>/
    #     plan.json, plan.md, table_read.md, planning_cache.json
    #     stills/           reference stills (+ variants)
    #     voice_bank/       established character voice refs
    #     clips/            conformed per-shot clips (with H3 audio)
    #     render/           wgp manifests, raw/ live renders, render_status.json
    #     <project>_final.mp4   (previous finals kept timestamped -- see below)
    "base_dir": "./films",
    # When reassembling (e.g. after regenerate_scenes), the existing final is
    # kept as <slug>_final_<timestamp>.mp4 so versions can be compared.
    "keep_previous_finals": True,
    # Legacy split layout (pre-v3.5). Only read for AUTO-MIGRATION: if a
    # project exists under these and not under base_dir, its contents are
    # moved into the unified folder on first touch.
    "base_working_dir": "./film_work",
    "base_output_dir": "./film_output",
}
