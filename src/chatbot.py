"""
chatbot.py
-----------
AI chatbot assistant for concrete mix design / SCC / model-usage questions.

Design goals:
  * Provider-agnostic: works with an Anthropic (Claude) or OpenAI API key,
    whichever the user supplies in the sidebar. No key is hardcoded.
  * Optional web access: when enabled (or auto-detected as needed), pulls a
    handful of live web results via DuckDuckGo (the `ddgs` package — the
    successor to the old `duckduckgo_search` package, which now emits a
    deprecation warning and proxies to `ddgs`) and feeds short summarized
    snippets to the LLM as grounding context.
  * Degrades gracefully: if no API key is configured, falls back to a small
    rule-based FAQ so the tab is still useful rather than just erroring out.
"""

from __future__ import annotations

from typing import Optional

SYSTEM_PROMPT = """You are an expert assistant embedded in a Self-Compacting \
Concrete (SCC) Mix Design web application. You help civil/structural \
engineering students and practitioners with:
  - SCC mix design principles (EFNARC guidelines, filling ability, passing \
ability, segregation resistance, viscosity)
  - Concrete materials science (cement chemistry, w/c ratio, aggregates, \
superplasticizers/admixtures, curing, strength development)
  - How to use THIS application (Predict Strength mode, Reverse Design mode, \
Admin retraining, the charts and downloads it provides)
  - General structural/geotechnical engineering concepts when asked

Answer concisely and technically, like a knowledgeable colleague — use \
bullet points for multi-part answers. If you are given "Web search results" \
context, ground your answer in it and note that the information came from a \
live web search. If a question is outside civil/structural engineering and \
this app's scope, answer briefly if you can, but steer the conversation back \
to concrete/SCC topics when relevant. Never fabricate specific standards \
clauses or numeric limits you're not confident about — say so instead."""

WEB_SEARCH_TRIGGER_KEYWORDS = [
    "latest", "current", "recent", "today", "this year", "news",
    "update", "updated", "price", "cost of", "market", "standard revision",
    "new code", "2025", "2026", "2027",
]


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------
def needs_web_search(query: str) -> bool:
    """Lightweight heuristic: does this question likely need current info?"""
    q = query.lower()
    return any(kw in q for kw in WEB_SEARCH_TRIGGER_KEYWORDS)


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Run a DuckDuckGo web search and return a list of
    {"title", "href", "body"} dicts. Returns [] (never raises) on any
    failure so the chatbot can continue without web grounding.
    """
    try:
        try:
            from ddgs import DDGS  # current package name
        except ImportError:
            from duckduckgo_search import DDGS  # older package name, still works
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return results
    except Exception:
        return []


def format_search_context(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["Web search results:"]
    for i, r in enumerate(results, start=1):
        title = r.get("title", "").strip()
        body = r.get("body", "").strip()
        href = r.get("href", "").strip()
        lines.append(f"{i}. {title} — {body} (Source: {href})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------
def _call_anthropic(api_key: str, model_name: str, system_prompt: str,
                     history: list[dict], max_tokens: int = 1024) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": m["role"], "content": m["content"]} for m in history],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_openai(api_key: str, model_name: str, system_prompt: str,
                  history: list[dict], max_tokens: int = 1024) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system_prompt}] +
                 [{"role": m["role"], "content": m["content"]} for m in history],
    )
    return response.choices[0].message.content


def get_llm_response(
    history: list[dict],
    provider: str,
    api_key: str,
    model_name: str,
    web_results: Optional[list[dict]] = None,
) -> str:
    """
    `history` is a list of {"role": "user"|"assistant", "content": str}
    representing the running conversation (most recent message last).
    Raises the underlying exception on failure — callers should catch it
    and show a friendly st.error().
    """
    system_prompt = SYSTEM_PROMPT
    if web_results:
        system_prompt += "\n\n" + format_search_context(web_results)

    if provider == "Anthropic (Claude)":
        return _call_anthropic(api_key, model_name, system_prompt, history)
    elif provider == "OpenAI":
        return _call_openai(api_key, model_name, system_prompt, history)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Offline fallback (no API key configured)
# ---------------------------------------------------------------------------
_FAQ = [
    (["what is scc", "what is self-compacting", "self compacting concrete"],
     "Self-Compacting Concrete (SCC) is a highly flowable, non-segregating "
     "concrete that spreads and fills formwork under its own weight, fully "
     "encapsulating reinforcement without any mechanical vibration. It "
     "relies on a high paste/fines content and a superplasticizer (usually "
     "polycarboxylate ether-based) for flowability, plus viscosity-"
     "modifying admixtures or a well-controlled fine-aggregate content for "
     "segregation resistance."),
    (["filling ability", "passing ability", "segregation resistance"],
     "SCC is characterized by three fresh-state properties: (1) Filling "
     "ability — flows and fills formwork under self-weight (tested via "
     "slump-flow), (2) Passing ability — flows through congested "
     "reinforcement without blocking (tested via L-box/J-ring), and (3) "
     "Segregation resistance — coarse aggregate stays uniformly "
     "distributed (tested via sieve segregation / GTM test)."),
    (["water cement ratio", "w/c ratio", "wc ratio"],
     "SCC mixes typically use a water/cement (or water/powder) ratio around "
     "0.28–0.45. Lower ratios generally raise strength but must be "
     "balanced with enough superplasticizer to keep the mix flowable "
     "without segregating."),
    (["superplasticizer", "admixture", "hrwr"],
     "Superplasticizers (high-range water reducers, typically "
     "polycarboxylate ether-based for SCC) disperse cement particles "
     "electrostatically/sterically, letting you cut the water content "
     "while keeping high flowability — this is what makes low w/c, "
     "high-strength SCC mixes practical."),
    (["retrain", "admin", "upload dataset", "how do i train"],
     "In Admin Mode: enter the admin password, upload a CSV with columns "
     "Cement, Water, Fine_Aggregate, Coarse_Aggregate, Superplasticizer, "
     "Age, and Compressive_Strength, then click 'Retrain Model'. The app "
     "will clean the data, retrain, show updated R²/RMSE plus diagnostic "
     "charts, and let you save the new model to disk."),
    (["reverse", "target strength", "mix design generator"],
     "Reverse Design mode takes a target compressive strength and searches "
     "a large space of physically-plausible mixes (respecting realistic "
     "w/c ratio, aggregate, and superplasticizer dosage ranges), predicts "
     "each candidate's strength with the trained model, and returns 3 "
     "diverse options (economical / balanced / high-performance) closest "
     "to your target."),
]


def offline_fallback_response(query: str) -> str:
    """Very small keyword-matched FAQ used when no LLM API key is set."""
    q = query.lower()
    for keywords, answer in _FAQ:
        if any(kw in q for kw in keywords):
            return answer
    return (
        "I don't have an LLM API key configured right now, so I can only "
        "answer a few common questions (try asking about what SCC is, "
        "filling/passing ability, w/c ratio, superplasticizers, or how to "
        "use Admin retraining / Reverse Design). Add an Anthropic or "
        "OpenAI API key in the sidebar for full conversational answers."
    )
