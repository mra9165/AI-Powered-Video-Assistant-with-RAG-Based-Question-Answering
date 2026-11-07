import re
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_utils import get_llm, split_text, safe_invoke, SINGLE_PASS_CHARS

NOTES_MAX_CHARS = 12000

CHUNK_SYSTEM = """You are an expert meeting analyst. From this PORTION of a meeting
transcript, produce notes with EXACTLY these headers (write NONE if nothing applies):

## SUMMARY
- concise bullet points (max 120 words total, keep names, numbers, decisions)
## ACTION ITEMS
1. Task description | Owner | Deadline or 'Not specified'
## KEY DECISIONS
1. one decision per item
## OPEN QUESTIONS
1. unresolved questions or topics needing follow-up

Do not add anything else."""

FINAL_SYSTEM = """You are an expert meeting analyst. The input is either a meeting
transcript or structured notes taken from different parts of one meeting. Merge
everything into one final result, removing duplicates and keeping all unique items.
Use EXACTLY these headers:

## TITLE
(a short professional meeting title, maximum 8 words, one line)
## SUMMARY
- professional bullet points covering the whole meeting
## ACTION ITEMS
1. Task description | Owner | Deadline or 'Not specified'
(if none, write exactly: No action items found.)
## KEY DECISIONS
1. one decision per item
(if none, write exactly: No key decisions found.)
## OPEN QUESTIONS
1. unresolved questions or topics needing follow-up
(if none, write exactly: No open questions found.)

Do not add anything else."""

_HEADER_RE = re.compile(
    r"^##\s*(TITLE|SUMMARY|ACTION ITEMS|KEY DECISIONS|OPEN QUESTIONS)\s*$",
    flags=re.M | re.I,
)

_DEFAULTS = {
    "TITLE": "Meeting Summary",
    "SUMMARY": "- No summary available.",
    "ACTION ITEMS": "No action items found.",
    "KEY DECISIONS": "No key decisions found.",
    "OPEN QUESTIONS": "No open questions found.",
}


def _parse_final(text: str) -> dict:
    out = dict(_DEFAULTS)
    parts = _HEADER_RE.split(text)
    # parts = [pre, header1, body1, header2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1].strip()
        if body:
            out[parts[i].upper()] = body
    return out


@lru_cache(maxsize=4)
def analyze(transcript: str) -> dict:
    """ONE pass over the transcript -> title, summary, actions, decisions, questions.
    Cached, so summarize() and the extract_* functions share the same result."""
    chunk_llm = get_llm(temperature=0.2, max_tokens=1000)
    final_llm = get_llm(temperature=0.2, max_tokens=1500)

    chunk_chain = (
        ChatPromptTemplate.from_messages([("system", CHUNK_SYSTEM), ("human", "{text}")])
        | chunk_llm | StrOutputParser()
    )
    final_chain = (
        ChatPromptTemplate.from_messages([("system", FINAL_SYSTEM), ("human", "{text}")])
        | final_llm | StrOutputParser()
    )

    # FAST PATH: short transcript -> a single LLM call
    if len(transcript) <= SINGLE_PASS_CHARS:
        return _parse_final(safe_invoke(final_chain, {"text": transcript},
                                        out_tokens=1100))

    # Map: one call per chunk
    notes = [safe_invoke(chunk_chain, {"text": c}, out_tokens=700)
             for c in split_text(transcript)]
    combined = "\n\n".join(notes)

    # Only for very long videos: shrink notes until they fit in one final request
    while len(combined) > NOTES_MAX_CHARS:
        groups = split_text(combined, chunk_chars=NOTES_MAX_CHARS // 2)
        notes = [safe_invoke(chunk_chain, {"text": g}, out_tokens=700) for g in groups]
        combined = "\n\n".join(notes)

    return _parse_final(safe_invoke(final_chain, {"text": combined}, out_tokens=1100))


def summarize(transcript: str) -> str:
    r = analyze(transcript)
    return f"Title: {r['TITLE']}\n\nSummary:\n{r['SUMMARY']}"