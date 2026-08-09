"""LinguaMate — English Conversation Assistant.

A Streamlit app that helps non-native English speakers find the exact right words
to say in any real-world situation by providing tone-adjusted suggestions and
natural native phrasing tips powered by the Groq API.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from a local .env file (if present) so the
# GROQ_API_KEY fallback works without any extra setup.
load_dotenv()

PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "llama-3.3-70b-versatile"

AUDIENCE_OPTIONS = [
    "Coworker / Boss",
    "Friend / Peer",
    "Customer / Client",
    "General / Stranger",
]

SYSTEM_PROMPT = """\
You are an expert English communication coach. Your goal is to help non-native \
speakers express themselves naturally and confidently in English.

Given the context, intent, and target audience, generate:
1. Three distinct variations written as natural sentences the user can copy:
   - Professional / Polished
   - Casual / Friendly
   - Direct & Concise
2. Smart Nuance & Vocabulary Tips (explain native idioms, tone subtle points, \
or why certain words work better).
3. A "Pronunciation / Stress Hint" for key difficult words used in the suggestions.

Return the response in valid JSON format with the following keys:
{
  "professional": "...",
  "casual": "...",
  "direct": "...",
  "nuance_tips": ["...", "..."],
  "pronunciation_hints": ["...", "..."]
}

Rules:
- Return ONLY the JSON object. No prose, no markdown fences, no commentary.
- Each variation must be a complete, ready-to-send sentence or short paragraph.
- Provide at least 2 nuance_tips and 2 pronunciation_hints.
- Tailor tone, vocabulary, and formality to the target audience.
- Every entry in nuance_tips and pronunciation_hints MUST include a concrete \
example phrase or sentence in parentheses, in quotes, or in italics. For example:
  "Use 'unwell' instead of 'sick' for a more formal tone (e.g., 'I'm feeling unwell today')."
  "Stress the second syllable: com-PLAIN (kəm-PLAYN)."
  If a tip would not naturally take an example, then count it as not present and \
return a different tip that does.
"""


def get_api_key() -> str | None:
    """Resolve the Groq API key from sidebar input, secrets, or env vars."""
    sidebar_key = st.session_state.get("groq_api_key", "").strip()
    if sidebar_key:
        return sidebar_key

    try:
        secrets_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        secrets_key = None
    if secrets_key:
        return str(secrets_key).strip() or None

    env_key = os.environ.get("GROQ_API_KEY")
    if env_key:
        return env_key.strip() or None

    return None


def build_user_message(context: str, intent: str, audience: str) -> str:
    intent_text = (
        intent.strip() or "(not specified — infer the most likely intent from the context)"
    )
    return (
        f"Target audience: {audience}\n\n"
        f"Context / scenario: {context.strip()}\n\n"
        f"What I want to convey: {intent_text}"
    )


def call_groq(client: Groq, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Call Groq chat completions with JSON mode, falling back across models."""
    last_error: Exception | None = None
    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=900,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
            return json.loads(content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise RuntimeError(f"Both primary and fallback models failed. Last error: {last_error}")


def render_suggestion_card(label: str, icon: str, body: str) -> None:
    with st.container(border=True):
        st.subheader(f"{icon} {label}")
        st.write(body)


# A "concrete example" inside a tip is detected if the text contains any of:
#   - a parenthetical phrase, e.g.  (e.g., "...") or (kəm-PLAYN)
#   - double- or single-quoted phrase, e.g.  'unwell' or "I'm running late"
#   - italicized text, e.g.  *unwell*
#   - an explicit prefix like "e.g." / "example:" / "such as"
_PAREN_EXAMPLE_RE = re.compile(r"\([^)]{2,}\)")
_QUOTED_EXAMPLE_RE = re.compile(r"[\"']([^\"']{2,})[\"']")
_ITALIC_EXAMPLE_RE = re.compile(r"(?<!\*)\*[^\*]{2,}\*(?!\*)")
_PREFIX_EXAMPLE_RE = re.compile(
    r"\b(e\.g\.|example:|for example|such as|like this)\b", re.IGNORECASE
)


def has_example(text: str) -> bool:
    """Heuristic check: does this tip/hint contain a recognizable example phrase?"""
    if not text or not text.strip():
        return False
    if _PAREN_EXAMPLE_RE.search(text):
        return True
    if _ITALIC_EXAMPLE_RE.search(text):
        return True
    if _PREFIX_EXAMPLE_RE.search(text):
        return True
    if _QUOTED_EXAMPLE_RE.search(text):
        return True
    return False


def find_missing_examples(
    data: dict[str, Any], keys: tuple[str, ...] = ("nuance_tips", "pronunciation_hints")
) -> list[tuple[str, int, str]]:
    """Return [(key, index, text), ...] for items that lack an example."""
    missing: list[tuple[str, int, str]] = []
    for key in keys:
        items = data.get(key) or []
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, str) or not has_example(item):
                missing.append((key, idx, item if isinstance(item, str) else ""))
    return missing


def call_groq_with_examples(
    client: Groq,
    messages: list[dict[str, str]],
    data: dict[str, Any],
) -> dict[str, Any]:
    """Call Groq; if tips/hints are missing examples, retry once with a fix-up prompt."""
    missing = find_missing_examples(data)
    if not missing:
        return data

    bullets = "\n".join(f"- {key}[{idx}]: {text or '(empty)'}" for key, idx, text in missing)
    followup = {
        "role": "user",
        "content": (
            "Some entries below are missing a concrete example phrase. Rewrite "
            "ONLY those entries so each one includes a clear example (in "
            "parentheses, quotes, or italics, e.g. '(e.g., \"I'm running late\")' "
            "or '*com-PLAIN*'). Keep all other entries exactly as they were.\n\n"
            f"Entries that need an example:\n{bullets}\n\n"
            "Return the full JSON object again with the same keys."
        ),
    }
    response = client.chat.completions.create(
        model=PRIMARY_MODEL,
        messages=[*messages, {"role": "assistant", "content": json.dumps(data)}, followup],
        temperature=0.5,
        max_tokens=900,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    return json.loads(content)


def render_results(data: dict[str, Any]) -> None:
    variations = [
        ("Professional / Polished", ":material/business_center:", "professional"),
        ("Casual / Friendly", ":material/chat_bubble:", "casual"),
        ("Direct & Concise", ":material/bolt:", "direct"),
    ]
    for label, icon, key in variations:
        value = data.get(key)
        if value:
            render_suggestion_card(label, icon, str(value))

    missing = {key for key, _, _ in find_missing_examples(data)}

    nuance_tips = data.get("nuance_tips") or []
    if nuance_tips:
        with st.container(border=True):
            st.subheader(":material/lightbulb: Smart nuance & vocabulary tips")
            for tip in nuance_tips:
                st.markdown(f"- {tip}")
            if "nuance_tips" in missing:
                st.caption(
                    ":material/warning: Some tips above don't include an "
                    "example — the model couldn't produce one in time.",
                    icon=":material/warning:",
                )

    pronunciation_hints = data.get("pronunciation_hints") or []
    if pronunciation_hints:
        with st.container(border=True):
            st.subheader(":material/record_voice_over: Pronunciation / stress hints")
            for hint in pronunciation_hints:
                st.markdown(f"- {hint}")
            if "pronunciation_hints" in missing:
                st.caption(
                    ":material/warning: Some pronunciation hints above don't "
                    "include an example — the model couldn't produce one in time.",
                    icon=":material/warning:",
                )


def main() -> None:
    st.set_page_config(
        page_title="LinguaMate",
        page_icon=":material/translate:",
        layout="centered",
    )

    st.title("LinguaMate")
    st.caption(
        "Find the exact right words to say in any real-world situation — "
        "tone-adjusted suggestions and native phrasing tips, powered by Groq."
    )

    # --- Sidebar: API key configuration -------------------------------
    with st.sidebar:
        st.header("Settings")
        st.text_input(
            "Groq API key",
            type="password",
            key="groq_api_key",
            help="Your key is kept in this session only and never stored.",
            placeholder="gsk_...",
        )
        st.caption(
            "We fall back to `GROQ_API_KEY` from environment variables or "
            "`.env` if nothing is entered here."
        )

    api_key = get_api_key()
    if not api_key:
        st.warning(
            "Please provide your Groq API key in the sidebar to start getting suggestions.",
            icon=":material/key:",
        )
        st.stop()

    # --- Main input form ----------------------------------------------
    with st.form("linguamate_form", border=False):
        st.subheader("Tell LinguaMate about your situation")

        context = st.text_area(
            "Context / scenario",
            placeholder=(
                "Example: My boss asked for a status update on a delayed "
                "project during our weekly standup."
            ),
            height=100,
        )
        intent = st.text_area(
            "What you want to convey (optional)",
            placeholder=(
                "Example: The delay is due to an unexpected bug, but the task "
                "will be ready by tomorrow afternoon. "
                "Leave blank to let LinguaMate infer your intent from the context."
            ),
            height=100,
        )
        audience = st.selectbox("Target audience / relationship", AUDIENCE_OPTIONS)

        submitted = st.form_submit_button(
            "Get smart suggestions",
            icon=":material/rocket_launch:",
            type="primary",
        )

    # --- Results -------------------------------------------------------
    if not submitted:
        st.info(
            "Fill in the form above and press **Get smart suggestions** to "
            "see three tone-adjusted variations, plus nuance and pronunciation tips.",
            icon=":material/tips_and_updates:",
        )
        st.stop()

    if not context.strip():
        st.warning(
            "Please describe the situation in the context field so the model "
            "can tailor its suggestions.",
            icon=":material/warning:",
        )
        st.stop()

    results_slot = st.container()
    with results_slot:
        with st.spinner("Crafting your suggestions...", show_time=True):
            try:
                client = Groq(api_key=api_key)
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_message(context, intent, audience),
                    },
                ]
                data = call_groq(client, messages)
                if find_missing_examples(data):
                    data = call_groq_with_examples(client, messages, data)
            except json.JSONDecodeError:
                st.error(
                    "The model returned an unexpected response. Please try again.",
                    icon=":material/error:",
                )
                return
            except Exception as exc:  # noqa: BLE001
                st.error(
                    f"Something went wrong while contacting Groq: {exc}",
                    icon=":material/error:",
                )
                return

        st.subheader("Suggested responses")
        render_results(data)


if __name__ == "__main__":
    main()
