"""
film_maker.llm
==============
Self-contained LLM plumbing, ported from the music generator + the
comic pipeline's token-budget module:

  * get_llm(prompt, ...)        — retrying chat completion (OpenAI or Grok).
  * get_vision(prompt, images)  — vision call for the best-of-N still pick.
  * safe_json_list / safe_json_dict — robust JSON extraction (json_repair).
  * estimate_tokens / clamp_text / guard_prompt — budget enforcement so no
    prompt ever balloons past the model ceiling (the 400/413 failure mode
    the comic pipeline hit in production).
  * clean_text — ftfy + unidecode + entity-key de-leaking, same guarantees
    as the music pipeline's clean_scene_text.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

logger = logging.getLogger("film_maker")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

try:
    import json_repair
except ImportError:  # degrade to stdlib json
    json_repair = None

try:
    import ftfy
except ImportError:
    ftfy = None
try:
    from unidecode import unidecode
except ImportError:
    unidecode = None


# ---------------------------------------------------------------------------
# Token budget (slimmed port of comic_book_token_budget)
# ---------------------------------------------------------------------------
GLOBAL_MAX_PROMPT_TOKENS = int(os.getenv("LLM_GLOBAL_MAX_PROMPT_TOKENS", "200000"))
TRUNCATION_MARKER = " …[trimmed]"

_ENCODER = None
_ENCODER_TRIED = False


def _get_encoder():
    global _ENCODER, _ENCODER_TRIED
    if _ENCODER_TRIED:
        return _ENCODER
    _ENCODER_TRIED = True
    try:
        import tiktoken
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENCODER = None
    return _ENCODER


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    if not isinstance(text, str):
        text = str(text)
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, (len(text) + 3) // 4)


def clamp_text(text: str, max_tokens: int, where: str = "") -> str:
    if not text:
        return text or ""
    if not isinstance(text, str):
        text = str(text)
    if estimate_tokens(text) <= max_tokens:
        return text
    cut = text[: max(0, max_tokens * 4 - len(TRUNCATION_MARKER))]
    while cut and estimate_tokens(cut + TRUNCATION_MARKER) > max_tokens:
        cut = cut[: int(len(cut) * 0.9)]
    sp = cut.rfind(" ", int(len(cut) * 0.8))
    if sp > 0:
        cut = cut[:sp]
    logger.warning("[TokenBudget] Clamped %s to ~%d tokens.", where or "field",
                   max_tokens)
    return cut.rstrip() + TRUNCATION_MARKER


def guard_prompt(prompt: str, max_tokens: int = GLOBAL_MAX_PROMPT_TOKENS,
                 origin: str = "") -> str:
    cost = estimate_tokens(prompt)
    if cost <= max_tokens:
        return prompt
    notice = ("\n\n[NOTE: a large context block was omitted here to fit the "
              "model's limit; instructions above and the request below are "
              "intact.]\n\n")
    body = max(1, max_tokens - estimate_tokens(notice))
    head = clamp_text(prompt, int(body * 0.6), where=f"{origin}:head")
    tail = prompt[-(body - int(body * 0.6)) * 4:]
    logger.error("[TokenBudget] GLOBAL GUARD tripped for %s (~%d tokens).",
                 origin or "an LLM call", cost)
    result = head + notice + tail
    if estimate_tokens(result) > max_tokens:
        result = clamp_text(result, max_tokens, where=f"{origin}:final")
    return result


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------
_openai_client = None
_grok_client = None
_CFG: Dict = {}


def bind_config(cfg: Dict) -> None:
    """Call once at pipeline start so every module shares one config."""
    global _CFG
    _CFG = cfg


def _client(use_grok: bool):
    global _openai_client, _grok_client
    from openai import OpenAI
    if use_grok:
        if _grok_client is None:
            key = _CFG.get("grok_api_key") or os.getenv("XAI_API_KEY", "")
            if not key:
                raise RuntimeError("XAI_API_KEY not set but use_grok=True.")
            _grok_client = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
        return _grok_client
    if _openai_client is None:
        key = _CFG.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        _openai_client = OpenAI(api_key=key)
    return _openai_client


FILMMAKER_SYSTEM_MESSAGE = (
    "You are a master filmmaker: screenwriter, director, cinematographer, and "
    "sound designer in one, with an Academy-Award-level command of story "
    "structure, subtext, comic timing, dread, tenderness, and spectacle. You "
    "write concretely and visually: everything you describe can be seen or "
    "heard on screen. You never use em-dashes. When asked for JSON you return "
    "ONLY raw JSON with no markdown fences and no commentary, keep every "
    "string value concise, and never echo input data back beyond the "
    "requested copied keys."
)


def get_llm(prompt: str, temperature: float = 0.7, large: bool = False,
            system_message: str = None, max_completion_tokens: int = None,
            use_grok: Optional[bool] = None) -> str:
    """Retrying completion. Budgets come from config; a response that comes
    back truncated (finish_reason == 'length') is automatically retried with
    a doubled budget up to llm_truncation_retry_cap, because a truncated
    JSON array silently loses its tail items when repaired downstream."""
    if use_grok is None:
        use_grok = bool(_CFG.get("use_grok", False))
    model = (_CFG.get("grok_model") if use_grok else
             (_CFG.get("openai_model_large") if large else _CFG.get("openai_model")))
    client = _client(use_grok)
    prompt = guard_prompt(prompt, origin="get_llm")
    retries = int(_CFG.get("llm_retry_limit", 3))
    if max_completion_tokens is None:
        max_completion_tokens = int(_CFG.get(
            "llm_max_completion_tokens_large" if large
            else "llm_max_completion_tokens",
            100000 if large else 60000))
    cap = max(max_completion_tokens,
              int(_CFG.get("llm_truncation_retry_cap", 160000)))
    budget = max_completion_tokens
    last_partial = ""
    attempt = 0
    while attempt < retries:
        try:
            token_key = "max_tokens" if use_grok else "max_completion_tokens"
            kwargs = dict(
                messages=[{"role": "system",
                           "content": system_message or FILMMAKER_SYSTEM_MESSAGE},
                          {"role": "user", "content": prompt}],
                model=model, temperature=temperature,
                **{token_key: budget},
            )
            if use_grok:
                kwargs["reasoning_effort"] = "none"
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            if resp.choices[0].finish_reason == "length":
                last_partial = content if len(content) > len(last_partial) \
                    else last_partial
                if budget < cap:
                    new_budget = min(cap, budget * 2)
                    logger.warning(
                        "LLM response truncated at %d-token budget "
                        "(%d chars visible); retrying with %d tokens...",
                        budget, len(content), new_budget)
                    budget = new_budget
                    # A truncation retry does not consume a failure attempt.
                    continue
                logger.error(
                    "LLM response STILL truncated at the %d-token cap "
                    "(%d chars). Returning the partial -- downstream JSON "
                    "parsing may lose tail items; raise "
                    "llm_truncation_retry_cap or batch this call.",
                    budget, len(content))
                return content
            if content.strip():
                return content
            attempt += 1
        except Exception as e:
            attempt += 1
            wait = min(2 ** (attempt + 1), 30)
            logger.warning("LLM attempt %d failed: %s. Waiting %ds.",
                           attempt, e, wait)
            time.sleep(wait)
    logger.error("All %d LLM attempts failed.", retries)
    return last_partial


def batched_json_call(items: List[dict], build_prompt, id_key: str,
                      batch_size: int, temperature: float = 0.8,
                      large: bool = False, repair_passes: int = 2,
                      context_fn=None, label: str = "") -> Dict:
    """Generic bulk-call pattern for 'one JSON object per input item':
    splits items into batches (so no single response can outgrow what a
    model returns reliably), threads shared context between batches via
    context_fn(results_so_far) -- which is how film-wide rules like
    anti-repetition survive batching -- collects results keyed by id_key,
    and re-requests any items the model skipped in repair passes.

    build_prompt(batch_items, context_str) -> prompt string.
    Returns {id_value: item_dict}."""
    results: Dict = {}

    def _run(batch, temp):
        ctx = context_fn(results) if context_fn else ""
        for it in safe_json_list(get_llm(build_prompt(batch, ctx),
                                         temperature=temp, large=large)):
            if not isinstance(it, dict):
                continue
            k = it.get(id_key)
            if k is not None and k not in results:
                results[k] = it

    for start in range(0, len(items), batch_size):
        _run(items[start:start + batch_size], temperature)
    for rp in range(repair_passes):
        missing = [it for it in items if it.get(id_key) not in results]
        if not missing:
            break
        logger.info("[batch%s] Repair pass %d: %d item(s) missing.",
                    f":{label}" if label else "", rp + 1, len(missing))
        for start in range(0, len(missing), batch_size):
            _run(missing[start:start + batch_size],
                 min(1.0, temperature + 0.05))
    still = [it.get(id_key) for it in items if it.get(id_key) not in results]
    if still:
        logger.warning("[batch%s] %d item(s) never returned after repairs "
                       "(deterministic fallbacks will fill them): %s",
                       f":{label}" if label else "", len(still), still[:12])
    return results


def get_vision(prompt: str, images: List, temperature: float = 0.2) -> str:
    """Vision completion over PIL images (used for best-of-N still pick).
    Returns '' on any failure so callers can fall back deterministically."""
    model = _CFG.get("vision_model") or _CFG.get("openai_model")
    if not model:
        return ""
    try:
        content = [{"type": "text", "text": prompt}]
        for img in images:
            buf = io.BytesIO()
            small = img.copy()
            small.thumbnail((768, 768))
            small.convert("RGB").save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        client = _client(False)
        resp = client.chat.completions.create(
            model=model, temperature=temperature, max_completion_tokens=800,
            messages=[{"role": "user", "content": content}],
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("Vision call failed (%s); falling back.", e)
        return ""


# ---------------------------------------------------------------------------
# JSON parsing (port of safe_json_parse / safe_json_dict)
# ---------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t


def _loads(text: str):
    if json_repair is not None:
        return json_repair.loads(text)
    return json.loads(text)


def safe_json_list(response_text: str) -> List[dict]:
    cleaned = _strip_fences(response_text)
    try:
        data = _loads(cleaned)
        if isinstance(data, list):
            if data and isinstance(data[0], list):
                data = [x for sub in data for x in sub]
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    inner = [d for d in v if isinstance(d, dict)]
                    if inner:
                        return inner
            return [data]
    except Exception as e:
        logger.warning("JSON list parse failed: %s", e)
    m = re.search(r"\[.*\]", response_text or "", re.DOTALL)
    if m:
        try:
            data = _loads(m.group())
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except Exception:
            pass
    return []


def safe_json_dict(response_text: str) -> dict:
    cleaned = _strip_fences(response_text)
    try:
        data = _loads(cleaned)
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
    except Exception as e:
        logger.warning("JSON dict parse failed: %s", e)
    m = re.search(r"\{.*\}", response_text or "", re.DOTALL)
    if m:
        try:
            d = _loads(m.group())
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Text cleanup (port of clean_scene_text)
# ---------------------------------------------------------------------------
def clean_text(text: str, entity_bible: Optional[dict] = None,
               strip_non_ascii: bool = True) -> str:
    if not text:
        return text
    if ftfy is not None:
        text = ftfy.fix_text(text)
    if strip_non_ascii and unidecode is not None:
        text = unidecode(text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if entity_bible:
        for key, entry in entity_bible.items():
            desc = (entry.get("description") or "").strip()
            if not desc:
                continue
            pattern = re.compile(rf"\b{re.escape(key)}(?:'s)?\b", re.IGNORECASE)
            if not pattern.search(text):
                continue
            first_done = False
            short = desc.split(",")[0].strip()

            def _sub(m):
                nonlocal first_done
                poss = "'s" if m.group(0).lower().endswith("'s") else ""
                if not first_done:
                    first_done = True
                    return desc + poss
                return short + poss

            text = pattern.sub(_sub, text)
    return text
