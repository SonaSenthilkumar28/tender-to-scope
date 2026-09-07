"""
Tender-to-Scope Analyser
========================

Takes a procurement document (tender / RFP / statement of requirements) and turns
it into a structured scope, then flags what is ambiguous, missing or contradictory
BEFORE anyone commits to delivering it.

Why this is built as separate stages rather than one big prompt:

  1. EXTRACT  - pull out every discrete requirement, verbatim.
  2. ENRICH   - categorise each one, name its deliverable, and judge whether it
                is actually testable.
  3. GAPS     - find what ISN'T in the document but should be.

Each stage does one narrow job and hands structured JSON to the next, so a
failure stays local and any stage's output can be inspected on its own.

Stage 3 is the one that genuinely needs the others finished first: you can only
identify what a document is missing once you have established what it contains.
Finding absences is a different task from summarising presences.

Classification and testability were originally two stages. They were merged
because neither depends on the other's output - both enrich the same records
independently - and on a free tier with a fixed daily request quota, the second
call cost a run without buying any accuracy. Separation is worth paying for when
one stage depends on another; it is not worth paying for on principle.
"""

import json
import os
import random
import re
import time

import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Provider configuration.
#
# Every stage below talks to the model through one OpenAI-compatible endpoint,
# so the provider is a config value rather than a code change. Gemini, Groq,
# OpenAI and most others all expose this interface. Swapping provider means
# changing two settings, not rewriting the pipeline - which also means a
# provider changing its pricing or deprecating a model doesn't strand the tool.
#
# Defaults point at Google's free tier.
#   Gemini : https://generativelanguage.googleapis.com/v1beta/openai/
#   Groq   : https://api.groq.com/openai/v1
#   OpenAI : https://api.openai.com/v1
# ---------------------------------------------------------------------------


def setting(key: str, default: str = "") -> str:
    """Read config from Streamlit secrets first, then environment variables.
    Secrets is where the deployed app gets its key; env vars keep local runs
    working without a secrets file."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


BASE_URL = setting("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = setting("MODEL", "gemini-2.5-flash")
MOCK = setting("MOCK") == "1"          # lets the UI be tested with no API key


@st.cache_resource
def client():
    key = setting("LLM_API_KEY")
    if not key:
        st.error(
            "No LLM_API_KEY found. On Streamlit Community Cloud, add it under "
            "**Manage app → Settings → Secrets** as `LLM_API_KEY = \"your-key\"`."
        )
        st.stop()
    return OpenAI(api_key=key, base_url=BASE_URL)


# A long ITT produces a long requirements array. The first real document this
# was tested on blew through a 4,000-token output cap and returned JSON that
# stopped mid-object, which the parser rejected outright - the whole run lost
# because the last item was incomplete. Two changes came out of that: a much
# higher ceiling, and a parser that salvages the complete objects rather than
# discarding everything. A partial scope the user is warned about beats no scope.
MAX_OUTPUT_TOKENS = 8000


# Transient failures are the tax on a multi-stage pipeline. Each stage is another
# chance for the provider to return "high demand, try later", and on a free tier
# that is common. With four stages, a 5% per-call failure rate means roughly one
# run in five dies - and it dies after the earlier stages have already done their
# work. Retrying the one call that failed is far cheaper than redoing the run.
RETRY_ON = (429, 500, 502, 503, 504)
MAX_ATTEMPTS = 4


def ask(system_prompt: str, user_content: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    """One call to the model, retried with exponential backoff on transient errors.

    Temperature 0 so the same document gives the same scope twice - a procurement
    tool that changes its mind between runs is useless to the person relying on it."""
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client().chat.completions.create(
                model=MODEL,
                max_tokens=max_tokens,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            if getattr(resp.choices[0], "finish_reason", None) == "length":
                st.session_state["truncated"] = True
            return resp.choices[0].message.content

        except Exception as e:
            last_error = e
            status = getattr(e, "status_code", None)

            # A 429 is ambiguous: per-minute throttling is worth waiting out,
            # a per-day quota is not. Retrying a daily limit just burns three
            # more requests against the same exhausted allowance.
            if "PerDay" in str(e) or "RequestsPerDay" in str(e):
                st.error(
                    "**Daily quota exhausted.** Your provider's free tier allows a fixed "
                    "number of requests per day for this model, and this run used them up.\n\n"
                    "Either wait for the quota to reset, or set `MODEL` in Secrets to a model "
                    "with a higher daily allowance — check yours at "
                    "https://aistudio.google.com/rate-limit"
                )
                st.stop()

            # 400/401/404 will fail identically however many times we ask.
            if status is not None and status not in RETRY_ON:
                break

            if attempt < MAX_ATTEMPTS - 1:
                wait = 2 ** attempt + random.uniform(0, 1)   # 1s, 2s, 4s + jitter
                st.write(f"Provider busy, retrying in {wait:.0f}s…")
                time.sleep(wait)

    st.error(
        f"Model call failed after {MAX_ATTEMPTS} attempts "
        f"(`{MODEL}` at `{BASE_URL}`): {last_error}\n\n"
        "A 503 means the provider is under load - wait a minute and try again. "
        "A model-not-found error means `MODEL` in Secrets needs updating."
    )
    st.stop()


def parse_json(raw: str):
    """Models sometimes wrap JSON in markdown fences or add a sentence before it.
    Strip that off rather than letting the whole pipeline fail on formatting.

    If the response was truncated mid-array, keep the objects that did complete
    instead of throwing the run away. Losing the last requirement is recoverable;
    losing all forty is not."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = min([i for i in (text.find("["), text.find("{")) if i != -1], default=-1)
    if start > 0:
        text = text[start:]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Salvage: cut back to the last complete object and close the array.
    if text.lstrip().startswith("["):
        cut = text.rfind("}")
        if cut != -1:
            try:
                salvaged = json.loads(text[:cut + 1] + "]")
                st.session_state["truncated"] = True
                return salvaged
            except json.JSONDecodeError:
                pass

    st.error(
        "The model's response could not be read as JSON, and no complete records "
        "could be salvaged from it. Try a shorter document, or paste one section."
    )
    st.stop()


# A real ITT produced 83 requirements. Sending all 83 into one enrichment call
# and asking for 83 enriched objects back overruns any sane output limit - and
# the salvage path then silently drops the tail of the list. Batching keeps each
# response comfortably inside the ceiling, and turns one fragile call into
# several independent ones, so a failure costs a batch rather than the run.
#
# Batch size is a direct trade against the provider's daily request quota: the
# free tier allows a fixed number of calls per day, so every extra batch is one
# fewer run available. 40 is the largest batch whose output reliably fits.
BATCH_SIZE = 40


def run_batched(system_prompt, records):
    """Enrich a list of requirements in batches, preserving order."""
    out = []
    batches = [records[i:i + BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]
    for n, batch in enumerate(batches, 1):
        if len(batches) > 1:
            st.write(f"  batch {n} of {len(batches)}")
        out.extend(parse_json(ask(system_prompt, json.dumps(batch, indent=2))))
    return out


# ---------------------------------------------------------------- stage 1

EXTRACT_SYS = """You extract requirements from procurement documents.

Pull out every discrete requirement the supplier would have to satisfy. Quote the
document's own wording - do not paraphrase, tidy up or interpret. If one sentence
contains two requirements, split it into two.

Ignore background, boilerplate and company history. A requirement is something a
supplier could be held to.

Return ONLY a JSON array:
[{"id": "R1", "requirement": "<verbatim text>", "section": "<heading or 'unstated'>"}]"""


def stage_extract(doc_text):
    if MOCK:
        return [
            {"id": "R1", "requirement": "The supplier shall provide a case management system.", "section": "3.1 Scope"},
            {"id": "R2", "requirement": "The system must be available during core hours.", "section": "3.2 Service"},
            {"id": "R3", "requirement": "Training shall be provided to all staff.", "section": "4.1 Implementation"},
        ]
    return parse_json(ask(EXTRACT_SYS, f"<document>\n{doc_text}\n</document>"))


# ---------------------------------------------------------------- stage 2

ENRICH_SYS = """You structure extracted requirements into a delivery scope AND judge
whether each one is written well enough to deliver against.

For each requirement add:
  category    - one of: Functional, Technical, Service Level, Commercial, Compliance, Implementation
  deliverable - the concrete thing the supplier would hand over to satisfy it
  priority    - Mandatory if the document uses shall/must/required; Desirable if
                should/may/preferred; Unclear if the language does not say
  testable    - true only if you could write a pass/fail acceptance test from the
                wording ALONE, with no further questions
  issue       - if not testable, the single specific thing that is undefined
                (e.g. "'core hours' is never defined"). Empty string if testable.

Be strict about testable. "Fast", "robust", "user-friendly", "as required",
"appropriate", "industry standard" and undefined time windows are not testable.
A requirement naming a number, a standard or a named artefact usually is.

Keep id and requirement exactly as given. Return ONLY the JSON array."""


def stage_enrich(reqs):
    if MOCK:
        return [
            {**reqs[0], "category": "Functional", "deliverable": "Case management application",
             "priority": "Mandatory", "testable": False,
             "issue": "'case management system' is never specified in functional terms"},
            {**reqs[1], "category": "Service Level", "deliverable": "Availability SLA",
             "priority": "Mandatory", "testable": False, "issue": "'core hours' is never defined"},
            {**reqs[2], "category": "Implementation", "deliverable": "Training programme",
             "priority": "Mandatory", "testable": False,
             "issue": "'all staff' is not quantified and no training format is given"},
        ]
    return run_batched(ENRICH_SYS, reqs)


# ---------------------------------------------------------------- stage 3

# ---------------------------------------------------------------- stage 4

GAPS_SYS = """You review a procurement document against the requirements already
extracted from it, and identify what is MISSING or CONTRADICTORY.

Look for: absent acceptance criteria, undefined terms used as if defined,
requirements that conflict, unstated dependencies and assumptions, missing
timelines, unclear data ownership or exit arrangements, and obligations placed on
the buyer that are never confirmed.

Do not restate requirements that are present. Report only problems.

For each finding:
  type     - Missing, Contradiction, Ambiguity, Dependency, or Assumption
  finding  - what the problem is, in one sentence
  risk     - what it costs the supplier if it goes unresolved
  severity - High, Medium or Low
  question - the exact question to put to the buyer before bidding

Return ONLY a JSON array."""


def stage_gaps(doc_text, reqs):
    if MOCK:
        return [
            {"type": "Missing", "finding": "No acceptance criteria for any deliverable.",
             "risk": "Sign-off becomes a matter of opinion and payment can be withheld.",
             "severity": "High", "question": "What are the acceptance criteria for each deliverable?"},
            {"type": "Ambiguity", "finding": "'Core hours' is used as an SLA term but never defined.",
             "risk": "Pricing the support model is guesswork; 24/7 costs many times 9-5.",
             "severity": "High", "question": "What exact hours and days does 'core hours' cover?"},
            {"type": "Dependency", "finding": "Training assumes buyer-side availability that is never confirmed.",
             "risk": "Delivery slips against a fixed date for reasons outside supplier control.",
             "severity": "Medium", "question": "Who schedules trainees, and what happens if sessions are missed?"},
        ]
    payload = f"<document>\n{doc_text}\n</document>\n\n<requirements_found>\n{json.dumps(reqs, indent=2)}\n</requirements_found>"
    return parse_json(ask(GAPS_SYS, payload, max_tokens=4000))


# ---------------------------------------------------------------- plumbing

def read_input(uploaded_file, pasted_text):
    """Prefer an uploaded file; fall back to the textbox."""
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".pdf"):
            pages = [p.extract_text() or "" for p in PdfReader(uploaded_file).pages]
            return "\n\n".join(pages)
        return uploaded_file.read().decode("utf-8", errors="ignore")
    return (pasted_text or "").strip()


def analyse(doc_text):
    """The four stages, in order. Each hands structured data to the next."""
    st.session_state["truncated"] = False
    steps = st.status("Analysing tender…", expanded=True)

    with steps:
        st.write("Extracting requirements verbatim")
        reqs = stage_extract(doc_text)

        st.write(f"Structuring {len(reqs)} requirements and assessing testability")
        reqs = stage_enrich(reqs)

        st.write("Re-reading the document for what is missing")
        gaps = stage_gaps(doc_text, reqs)

    steps.update(label="Analysis complete", state="complete", expanded=False)

    if st.session_state.get("truncated"):
        st.warning(
            "**This document was too long to process whole.** The model hit its output "
            "limit, so the results below are complete but not exhaustive — later "
            "requirements may be missing. For a full analysis, paste one section at a time.",
            icon="⚠️",
        )

    return reqs, gaps


SAMPLE = """3.1 Scope of Requirement
The Supplier shall provide a digital case management system for use by the Authority's
regional teams. The system must handle the full case lifecycle from intake to closure.

3.2 Service Levels
The system must be available during core hours. Response to critical incidents shall be
prompt. The Supplier shall provide appropriate support arrangements.

3.3 Data
All data shall remain the property of the Authority. The Supplier shall ensure data is
handled securely and in line with industry standard practice.

4.1 Implementation
Training shall be provided to all staff prior to go-live. The Supplier shall migrate
existing case records from the legacy system. Go-live is required by the end of Q3.

4.2 Commercial
Pricing shall be submitted on a fixed price basis. The Authority reserves the right to
extend the contract for a further period on the same terms.
"""

# ---------------------------------------------------------------- interface

st.set_page_config(page_title="Tender-to-Scope Analyser", page_icon="📋", layout="wide")

st.title("Tender-to-Scope Analyser")
st.markdown(
    "Turns a procurement document into a structured delivery scope, then flags what is "
    "ambiguous, missing or contradictory **before** anyone commits to delivering it.\n\n"
    "*Four stages: extract requirements verbatim → structure them into deliverables → "
    "test whether each one is measurable → re-read the document for what is absent.*"
)

left, right = st.columns([1, 2], gap="large")

with left:
    uploaded = st.file_uploader("Upload a tender", type=["pdf", "txt", "md"])
    pasted = st.text_area("…or paste the text", value=SAMPLE, height=320)
    go = st.button("Analyse", type="primary", use_container_width=True)
    st.caption(
        "Public UK tenders: [Find a Tender](https://www.find-tender.service.gov.uk) · "
        "[Contracts Finder](https://www.contractsfinder.service.gov.uk)"
    )

with right:
    if go:
        doc = read_input(uploaded, pasted)
        if len(doc) < 200:
            st.warning("Paste a tender document, or upload one. Needs to be at least a couple of paragraphs.")
            st.stop()

        reqs, gaps = analyse(doc)

        total = len(reqs)
        untestable = sum(1 for r in reqs if not r.get("testable"))
        high = sum(1 for g in gaps if g.get("severity") == "High")

        a, b, c = st.columns(3)
        a.metric("Requirements found", total)
        b.metric("Not testable as written", f"{untestable} of {total}")
        c.metric("Gaps found", len(gaps), f"{high} high severity", delta_color="inverse")

        st.subheader("Structured scope")
        st.dataframe(
            [{
                "ID": r.get("id", ""),
                "Requirement": r.get("requirement", ""),
                "Category": r.get("category", ""),
                "Deliverable": r.get("deliverable", ""),
                "Priority": r.get("priority", ""),
                "Testable": "Yes" if r.get("testable") else "No",
                "Issue": r.get("issue", ""),
            } for r in reqs],
            use_container_width=True, hide_index=True,
        )

        st.subheader("Gaps, contradictions and questions to ask")
        st.caption("Every question below should be answered by the buyer before this is priced.")
        st.dataframe(
            [{
                "Severity": g.get("severity", ""),
                "Type": g.get("type", ""),
                "Finding": g.get("finding", ""),
                "Risk to supplier": g.get("risk", ""),
                "Question for the buyer": g.get("question", ""),
            } for g in gaps],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Load a tender on the left and press **Analyse**. The sample is a composite of the vague phrasing that shows up in real public-sector tenders.")# Every stage below talks to the model through one OpenAI-compatible endpoint,
# so the provider is a config value rather than a code change. Gemini, Groq,
# OpenAI and most others all expose this interface. Swapping provider means
# changing two settings, not rewriting the pipeline - which also means a
# provider changing its pricing or deprecating a model doesn't strand the tool.
#
# Defaults point at Google's free tier.
#   Gemini : https://generativelanguage.googleapis.com/v1beta/openai/
#   Groq   : https://api.groq.com/openai/v1
#   OpenAI : https://api.openai.com/v1
# ---------------------------------------------------------------------------


def setting(key: str, default: str = "") -> str:
    """Read config from Streamlit secrets first, then environment variables.
    Secrets is where the deployed app gets its key; env vars keep local runs
    working without a secrets file."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


BASE_URL = setting("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = setting("MODEL", "gemini-2.5-flash")
MOCK = setting("MOCK") == "1"          # lets the UI be tested with no API key


@st.cache_resource
def client():
    key = setting("LLM_API_KEY")
    if not key:
        st.error(
            "No LLM_API_KEY found. On Streamlit Community Cloud, add it under "
            "**Manage app → Settings → Secrets** as `LLM_API_KEY = \"your-key\"`."
        )
        st.stop()
    return OpenAI(api_key=key, base_url=BASE_URL)


# A long ITT produces a long requirements array. The first real document this
# was tested on blew through a 4,000-token output cap and returned JSON that
# stopped mid-object, which the parser rejected outright - the whole run lost
# because the last item was incomplete. Two changes came out of that: a much
# higher ceiling, and a parser that salvages the complete objects rather than
# discarding everything. A partial scope the user is warned about beats no scope.
MAX_OUTPUT_TOKENS = 8000


# Transient failures are the tax on a multi-stage pipeline. Each stage is another
# chance for the provider to return "high demand, try later", and on a free tier
# that is common. With four stages, a 5% per-call failure rate means roughly one
# run in five dies - and it dies after the earlier stages have already done their
# work. Retrying the one call that failed is far cheaper than redoing the run.
RETRY_ON = (429, 500, 502, 503, 504)
MAX_ATTEMPTS = 4


def ask(system_prompt: str, user_content: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    """One call to the model, retried with exponential backoff on transient errors.

    Temperature 0 so the same document gives the same scope twice - a procurement
    tool that changes its mind between runs is useless to the person relying on it."""
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = client().chat.completions.create(
                model=MODEL,
                max_tokens=max_tokens,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            if getattr(resp.choices[0], "finish_reason", None) == "length":
                st.session_state["truncated"] = True
            return resp.choices[0].message.content

        except Exception as e:
            last_error = e
            status = getattr(e, "status_code", None)

            # 400/401/404 will fail identically however many times we ask.
            if status is not None and status not in RETRY_ON:
                break

            if attempt < MAX_ATTEMPTS - 1:
                wait = 2 ** attempt + random.uniform(0, 1)   # 1s, 2s, 4s + jitter
                st.write(f"Provider busy, retrying in {wait:.0f}s…")
                time.sleep(wait)

    st.error(
        f"Model call failed after {MAX_ATTEMPTS} attempts "
        f"(`{MODEL}` at `{BASE_URL}`): {last_error}\n\n"
        "A 503 means the provider is under load - wait a minute and try again. "
        "A model-not-found error means `MODEL` in Secrets needs updating."
    )
    st.stop()


def parse_json(raw: str):
    """Models sometimes wrap JSON in markdown fences or add a sentence before it.
    Strip that off rather than letting the whole pipeline fail on formatting.

    If the response was truncated mid-array, keep the objects that did complete
    instead of throwing the run away. Losing the last requirement is recoverable;
    losing all forty is not."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = min([i for i in (text.find("["), text.find("{")) if i != -1], default=-1)
    if start > 0:
        text = text[start:]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Salvage: cut back to the last complete object and close the array.
    if text.lstrip().startswith("["):
        cut = text.rfind("}")
        if cut != -1:
            try:
                salvaged = json.loads(text[:cut + 1] + "]")
                st.session_state["truncated"] = True
                return salvaged
            except json.JSONDecodeError:
                pass

    st.error(
        "The model's response could not be read as JSON, and no complete records "
        "could be salvaged from it. Try a shorter document, or paste one section."
    )
    st.stop()


# A real ITT produced 83 requirements. Sending all 83 into one enrichment call
# and asking for 83 enriched objects back overruns any sane output limit - and
# the salvage path then silently drops the tail of the list. Batching keeps each
# response comfortably inside the ceiling, and turns one fragile call into
# several independent ones, so a failure costs a batch rather than the run.
BATCH_SIZE = 25


def run_batched(system_prompt, records):
    """Enrich a list of requirements in batches, preserving order."""
    out = []
    batches = [records[i:i + BATCH_SIZE] for i in range(0, len(records), BATCH_SIZE)]
    for n, batch in enumerate(batches, 1):
        if len(batches) > 1:
            st.write(f"  batch {n} of {len(batches)}")
        out.extend(parse_json(ask(system_prompt, json.dumps(batch, indent=2))))
    return out


# ---------------------------------------------------------------- stage 1

EXTRACT_SYS = """You extract requirements from procurement documents.

Pull out every discrete requirement the supplier would have to satisfy. Quote the
document's own wording - do not paraphrase, tidy up or interpret. If one sentence
contains two requirements, split it into two.

Ignore background, boilerplate and company history. A requirement is something a
supplier could be held to.

Return ONLY a JSON array:
[{"id": "R1", "requirement": "<verbatim text>", "section": "<heading or 'unstated'>"}]"""


def stage_extract(doc_text):
    if MOCK:
        return [
            {"id": "R1", "requirement": "The supplier shall provide a case management system.", "section": "3.1 Scope"},
            {"id": "R2", "requirement": "The system must be available during core hours.", "section": "3.2 Service"},
            {"id": "R3", "requirement": "Training shall be provided to all staff.", "section": "4.1 Implementation"},
        ]
    return parse_json(ask(EXTRACT_SYS, f"<document>\n{doc_text}\n</document>"))


# ---------------------------------------------------------------- stage 2

CLASSIFY_SYS = """You structure extracted requirements into a delivery scope.

For each requirement add:
  category    - one of: Functional, Technical, Service Level, Commercial, Compliance, Implementation
  deliverable - the concrete thing the supplier would hand over to satisfy it
  priority    - Mandatory if the document uses shall/must/required; Desirable if
                should/may/preferred; Unclear if the language does not say

Keep id and requirement exactly as given. Return ONLY the JSON array."""


def stage_classify(reqs):
    if MOCK:
        return [
            {**reqs[0], "category": "Functional", "deliverable": "Case management application", "priority": "Mandatory"},
            {**reqs[1], "category": "Service Level", "deliverable": "Availability SLA", "priority": "Mandatory"},
            {**reqs[2], "category": "Implementation", "deliverable": "Training programme", "priority": "Mandatory"},
        ]
    return run_batched(CLASSIFY_SYS, reqs)


# ---------------------------------------------------------------- stage 3

ASSESS_SYS = """You judge whether each requirement is written well enough to deliver
against and be measured on.

For each one add:
  testable - true only if you could write a pass/fail acceptance test from the
             wording ALONE, with no further questions
  issue    - if not testable, the single specific thing that is undefined
             (e.g. "'core hours' is never defined"). Empty string if testable.

Be strict. "Fast", "robust", "user-friendly", "as required", "appropriate",
"industry standard" and undefined time windows are not testable. A requirement
naming a number, a standard or a named artefact usually is.

Keep all existing fields. Return ONLY the JSON array."""


def stage_assess(reqs):
    if MOCK:
        return [
            {**reqs[0], "testable": False, "issue": "'case management system' is never specified in functional terms"},
            {**reqs[1], "testable": False, "issue": "'core hours' is never defined"},
            {**reqs[2], "testable": False, "issue": "'all staff' is not quantified and no training format is given"},
        ]
    return run_batched(ASSESS_SYS, reqs)


# ---------------------------------------------------------------- stage 4

GAPS_SYS = """You review a procurement document against the requirements already
extracted from it, and identify what is MISSING or CONTRADICTORY.

Look for: absent acceptance criteria, undefined terms used as if defined,
requirements that conflict, unstated dependencies and assumptions, missing
timelines, unclear data ownership or exit arrangements, and obligations placed on
the buyer that are never confirmed.

Do not restate requirements that are present. Report only problems.

For each finding:
  type     - Missing, Contradiction, Ambiguity, Dependency, or Assumption
  finding  - what the problem is, in one sentence
  risk     - what it costs the supplier if it goes unresolved
  severity - High, Medium or Low
  question - the exact question to put to the buyer before bidding

Return ONLY a JSON array."""


def stage_gaps(doc_text, reqs):
    if MOCK:
        return [
            {"type": "Missing", "finding": "No acceptance criteria for any deliverable.",
             "risk": "Sign-off becomes a matter of opinion and payment can be withheld.",
             "severity": "High", "question": "What are the acceptance criteria for each deliverable?"},
            {"type": "Ambiguity", "finding": "'Core hours' is used as an SLA term but never defined.",
             "risk": "Pricing the support model is guesswork; 24/7 costs many times 9-5.",
             "severity": "High", "question": "What exact hours and days does 'core hours' cover?"},
            {"type": "Dependency", "finding": "Training assumes buyer-side availability that is never confirmed.",
             "risk": "Delivery slips against a fixed date for reasons outside supplier control.",
             "severity": "Medium", "question": "Who schedules trainees, and what happens if sessions are missed?"},
        ]
    payload = f"<document>\n{doc_text}\n</document>\n\n<requirements_found>\n{json.dumps(reqs, indent=2)}\n</requirements_found>"
    return parse_json(ask(GAPS_SYS, payload, max_tokens=4000))


# ---------------------------------------------------------------- plumbing

def read_input(uploaded_file, pasted_text):
    """Prefer an uploaded file; fall back to the textbox."""
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(".pdf"):
            pages = [p.extract_text() or "" for p in PdfReader(uploaded_file).pages]
            return "\n\n".join(pages)
        return uploaded_file.read().decode("utf-8", errors="ignore")
    return (pasted_text or "").strip()


def analyse(doc_text):
    """The four stages, in order. Each hands structured data to the next."""
    st.session_state["truncated"] = False
    steps = st.status("Analysing tender…", expanded=True)

    with steps:
        st.write("Extracting requirements verbatim")
        reqs = stage_extract(doc_text)

        st.write(f"Structuring {len(reqs)} requirements into deliverables")
        reqs = stage_classify(reqs)

        st.write("Assessing whether each is testable")
        reqs = stage_assess(reqs)

        st.write("Re-reading the document for what is missing")
        gaps = stage_gaps(doc_text, reqs)

    steps.update(label="Analysis complete", state="complete", expanded=False)

    if st.session_state.get("truncated"):
        st.warning(
            "**This document was too long to process whole.** The model hit its output "
            "limit, so the results below are complete but not exhaustive — later "
            "requirements may be missing. For a full analysis, paste one section at a time.",
            icon="⚠️",
        )

    return reqs, gaps


SAMPLE = """3.1 Scope of Requirement
The Supplier shall provide a digital case management system for use by the Authority's
regional teams. The system must handle the full case lifecycle from intake to closure.

3.2 Service Levels
The system must be available during core hours. Response to critical incidents shall be
prompt. The Supplier shall provide appropriate support arrangements.

3.3 Data
All data shall remain the property of the Authority. The Supplier shall ensure data is
handled securely and in line with industry standard practice.

4.1 Implementation
Training shall be provided to all staff prior to go-live. The Supplier shall migrate
existing case records from the legacy system. Go-live is required by the end of Q3.

4.2 Commercial
Pricing shall be submitted on a fixed price basis. The Authority reserves the right to
extend the contract for a further period on the same terms.
"""

# ---------------------------------------------------------------- interface

st.set_page_config(page_title="Tender-to-Scope Analyser", page_icon="📋", layout="wide")

st.title("Tender-to-Scope Analyser")
st.markdown(
    "Turns a procurement document into a structured delivery scope, then flags what is "
    "ambiguous, missing or contradictory **before** anyone commits to delivering it.\n\n"
    "*Four stages: extract requirements verbatim → structure them into deliverables → "
    "test whether each one is measurable → re-read the document for what is absent.*"
)

left, right = st.columns([1, 2], gap="large")

with left:
    uploaded = st.file_uploader("Upload a tender", type=["pdf", "txt", "md"])
    pasted = st.text_area("…or paste the text", value=SAMPLE, height=320)
    go = st.button("Analyse", type="primary", use_container_width=True)
    st.caption(
        "Public UK tenders: [Find a Tender](https://www.find-tender.service.gov.uk) · "
        "[Contracts Finder](https://www.contractsfinder.service.gov.uk)"
    )

with right:
    if go:
        doc = read_input(uploaded, pasted)
        if len(doc) < 200:
            st.warning("Paste a tender document, or upload one. Needs to be at least a couple of paragraphs.")
            st.stop()

        reqs, gaps = analyse(doc)

        total = len(reqs)
        untestable = sum(1 for r in reqs if not r.get("testable"))
        high = sum(1 for g in gaps if g.get("severity") == "High")

        a, b, c = st.columns(3)
        a.metric("Requirements found", total)
        b.metric("Not testable as written", f"{untestable} of {total}")
        c.metric("Gaps found", len(gaps), f"{high} high severity", delta_color="inverse")

        st.subheader("Structured scope")
        st.dataframe(
            [{
                "ID": r.get("id", ""),
                "Requirement": r.get("requirement", ""),
                "Category": r.get("category", ""),
                "Deliverable": r.get("deliverable", ""),
                "Priority": r.get("priority", ""),
                "Testable": "Yes" if r.get("testable") else "No",
                "Issue": r.get("issue", ""),
            } for r in reqs],
            use_container_width=True, hide_index=True,
        )

        st.subheader("Gaps, contradictions and questions to ask")
        st.caption("Every question below should be answered by the buyer before this is priced.")
        st.dataframe(
            [{
                "Severity": g.get("severity", ""),
                "Type": g.get("type", ""),
                "Finding": g.get("finding", ""),
                "Risk to supplier": g.get("risk", ""),
                "Question for the buyer": g.get("question", ""),
            } for g in gaps],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Load a tender on the left and press **Analyse**. The sample is a composite of the vague phrasing that shows up in real public-sector tenders.")
