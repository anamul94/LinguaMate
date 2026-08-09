---
name: tweak-prompt
description: Rewrite the SYSTEM_PROMPT constant in streamlit_app.py while preserving the JSON contract and the example-enforcement rule that has_example / find_missing_examples / call_groq_with_examples depend on. Use when the user wants to adjust coaching tone, add a new rule, or rephrase the instructions — but NOT for changes that affect the JSON shape, keys, or the example rule (those need a coordinated code change).
---

# Tweak the LinguaMate system prompt

The user wants to edit the `SYSTEM_PROMPT` constant at the top of `streamlit_app.py`. This skill makes a safe, contract-preserving edit.

## What is load-bearing (do NOT change)

- The five JSON keys and their types: `professional` (string), `casual` (string), `direct` (string), `nuance_tips` (array of strings), `pronunciation_hints` (array of strings).
- The rule that **every entry** in `nuance_tips` and `pronunciation_hints` contains a concrete example (parenthetical, quoted phrase, italicized token, or `e.g.`/`such as` prefix). The regexes in `has_example` and the auto-regenerate call in `call_groq_with_examples` assume this. If the new rule is incompatible, stop and tell the user the change requires updating `has_example` / `call_groq_with_examples` too.
- `response_format={"type": "json_object"}` on the Groq call.
- The instruction to return ONLY the JSON object with no prose or markdown fences.

## Workflow

1. Read `streamlit_app.py` and confirm the current `SYSTEM_PROMPT` value. Do not assume.
2. State the change in one sentence back to the user before editing — confirm you understood the intent (coaching tone shift, new rule, removed rule, etc.).
3. Edit only the `SYSTEM_PROMPT` triple-quoted string. Do not rename the constant, do not touch `PRIMARY_MODEL` / `FALLBACK_MODEL`, and do not modify the helper functions unless the user explicitly asked.
4. After the edit, re-run the smoke checks:
   ```
   .venv/bin/python -c "import ast; ast.parse(open('streamlit_app.py').read()); print('syntax OK')"
   .venv/bin/python -c "import streamlit_app; print('import OK')"
   ```
5. Tell the user what changed in plain English and remind them to refresh the running Streamlit tab (Streamlit hot-reloads, but a manual reload is safer after prompt changes).

## When to escalate

If the user asks for any of these, stop and explain that this skill alone is insufficient — `has_example` / `call_groq_with_examples` / `find_missing_examples` also need to change:

- Removing the example rule.
- Changing the JSON keys or their types.
- Switching off JSON-mode output.
- Returning markdown instead of raw JSON.

In those cases, ask whether they want a coordinated change or a follow-up edit.
