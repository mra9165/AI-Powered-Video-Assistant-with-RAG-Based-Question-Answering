from core.summarize import analyze

# Same function names as before, so your calling code doesn't change.
# No extra LLM calls: everything comes from the cached single pass in summarize.analyze().
def extract_action_items(transcript: str) -> str:
    return analyze(transcript)["ACTION ITEMS"]


def extract_key_decisions(transcript: str) -> str:
    return analyze(transcript)["KEY DECISIONS"]


def extract_questions(transcript: str) -> str:
    return analyze(transcript)["OPEN QUESTIONS"]