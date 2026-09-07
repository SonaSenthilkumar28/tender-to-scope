# Tender-to-Scope Analyser

Turns a procurement document into a structured delivery scope, then flags what is
ambiguous, missing or contradictory **before** anyone commits to delivering it.

Built because most bids go wrong for the same reason: the requirement was never
testable in the first place, and nobody asked the question while it was still
cheap to ask.

## How it works

Four stages, each a separate model call passing structured JSON to the next:

| Stage | Job |
|---|---|
| **1. Extract** | Pull every discrete requirement out of the document, verbatim. No paraphrasing — the supplier is held to the buyer's words, not a tidied version of them. |
| **2. Classify** | Categorise each requirement and name the deliverable it implies. Priority is read from the language: *shall/must* is mandatory, *should/may* is desirable, anything else is flagged Unclear. |
| **3. Assess** | Decide whether each requirement could be turned into a pass/fail acceptance test from its wording alone. "Prompt", "appropriate" and "industry standard" fail this test. |
| **4. Gaps** | Re-read the original document against everything found, looking for what *isn't* there — missing acceptance criteria, undefined terms, contradictions, unstated dependencies. |

### Why four stages instead of one prompt

Each stage has one narrow job, so its output can be inspected on its own and a
failure stays local instead of corrupting the whole result. Stage 4 also depends on
stages 1–3 having finished: you can only reliably identify what a document is
missing once you have established what it contains. Finding absences is a different
task from summarising presences, and it needs its own pass.

Temperature is 0 throughout. A procurement tool that returns a different scope on
the second run is worse than no tool, because someone is pricing work off it.

## Configuration

Every stage talks to one OpenAI-compatible endpoint, so the model provider is
configuration rather than code:

| Setting | Purpose |
|---|---|
| `LLM_API_KEY` | your key, from whichever provider you pick |
| `LLM_BASE_URL` | the provider's OpenAI-compatible endpoint |
| `MODEL` | the model id |

Provider options, all with a free tier:

- **Google Gemini** *(default)* — best fit here, because tender documents are long
  and Gemini's free tier has by far the largest context window.
  `https://generativelanguage.googleapis.com/v1beta/openai/`
- **Groq** — fastest, open-weight models. `https://api.groq.com/openai/v1`
- **GitHub Models** — free with any GitHub account, but capped at 8K input tokens
  per request, which is too small for most real tenders without chunking first.
  `https://models.github.ai/inference`

Keeping the provider behind two settings means a pricing change or a deprecated
model is a config edit, not a rewrite.

## Running it

**Deployed:** Streamlit Community Cloud, from this repo. Add the key under
**Manage app → Settings → Secrets** in TOML form:

```toml
LLM_API_KEY = "your-key-here"
```

**Locally:**

```bash
pip install -r requirements.txt
export LLM_API_KEY=your-key-here
streamlit run streamlit_app.py
```

`MOCK=1` runs the interface with canned data and no API calls — useful for checking
layout changes without spending quota.

## Try it with a real tender

UK public procurement documents are published openly:

- [Find a Tender](https://www.find-tender.service.gov.uk)
- [Contracts Finder](https://www.contractsfinder.service.gov.uk)

## A note on data

Free API tiers generally reserve the right to train on what you send. That is fine
here because the intended input is published public procurement documents. Do not
put confidential or client material through a free tier.

## Limitations

- Long documents are sent whole rather than chunked, so very large tenders will hit
  the context limit. Chunking with per-section extraction is the obvious next step.
- The testability judgement is strict by design. It will occasionally flag a
  requirement that a domain expert would accept — that is the intended bias, since
  the cost of an unnoticed ambiguity is much higher than the cost of one extra
  question to the buyer.
- Scanned PDFs without a text layer will not extract. OCR is not included.
