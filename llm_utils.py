import os
import re
import time
from collections import deque

from groq import RateLimitError, APIStatusError
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
# Free tier is ~8000 TPM for this model; stay just under it.
TPM_BUDGET = int(os.getenv("GROQ_TPM_BUDGET", "6800"))
OUTPUT_ALLOWANCE = 700           # default output reservation per call
PROMPT_OVERHEAD = 400            # system-prompt tokens not counted in inputs
CHUNK_CHARS = 10000              # ~2500 tokens per chunk (fewer calls = less overhead)
SINGLE_PASS_CHARS = 12000        # transcripts up to this size need only ONE call


def get_llm(temperature: float = 0.2, max_tokens: int = 1000):
    params = dict(
        model=MODEL,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=max_tokens,
    )

    if "gpt-oss" in MODEL:
        # Less hidden reasoning = fewer tokens and faster responses
        try:
            # Newer langchain-groq: explicit parameter
            return ChatGroq(**params, reasoning_effort="low")
        except Exception:
            # Older langchain-groq: must go through model_kwargs
            return ChatGroq(**params, model_kwargs={"reasoning_effort": "low"})

    return ChatGroq(**params)


def split_text(text: str, chunk_chars: int = CHUNK_CHARS) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_chars, chunk_overlap=200
    )
    return splitter.split_text(text)


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + OUTPUT_ALLOWANCE + PROMPT_OVERHEAD


class _Throttle:
    """Sliding 60s window so we never exceed the tokens-per-minute budget."""

    def __init__(self, budget):
        self.budget = budget
        self.events = deque()  # (timestamp, tokens)

    def wait_for(self, tokens):
        tokens = min(tokens, self.budget)
        while True:
            now = time.time()
            while self.events and now - self.events[0][0] >= 60:
                self.events.popleft()
            used = sum(t for _, t in self.events)
            if used + tokens <= self.budget:
                self.events.append((now, tokens))
                return
            sleep_for = 60 - (now - self.events[0][0]) + 0.5
            print(f"Throttling: waiting {sleep_for:.0f}s for token budget...")
            time.sleep(max(sleep_for, 1))


_throttle = _Throttle(TPM_BUDGET)


def _retry_after(err) -> float:
    """Parse 'Please try again in 7.5s' / '1m3.2s' from Groq's error message."""
    msg = str(err)
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
    if m:
        return int(m.group(1) or 0) * 60 + float(m.group(2)) + 1
    m = re.search(r"try again in ([\d.]+)ms", msg)
    if m:
        return float(m.group(1)) / 1000 + 1
    return 0


def safe_invoke(chain, inputs: dict, max_retries: int = 8,
                out_tokens: int = OUTPUT_ALLOWANCE) -> str:
    """Throttled invoke with smart retry on rate limits."""
    text_len = sum(len(str(v)) for v in inputs.values())
    cost = text_len // 4 + PROMPT_OVERHEAD + out_tokens
    for attempt in range(max_retries):
        _throttle.wait_for(cost)
        try:
            return chain.invoke(inputs)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = _retry_after(e) or min(10 * (attempt + 1), 60)
            print(f"Rate limit hit. Retrying in {wait:.0f}s "
                  f"(attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
        except APIStatusError as e:
            if getattr(e, "status_code", None) == 413:
                raise RuntimeError(
                    "Request too large for your Groq tier. "
                    "Lower CHUNK_CHARS / SINGLE_PASS_CHARS in llm_utils.py."
                ) from e
            raise