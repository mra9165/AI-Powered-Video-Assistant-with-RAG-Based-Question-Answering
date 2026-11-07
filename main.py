from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# Build the RAG index while the LLM analysis runs.
# Set to False if build_rag_chain() calls Groq (it would share the rate limit).
PARALLEL_RAG = True


def run_pipeline(source: str, language: str = "english") -> dict:
    print("Starting AI Video Assistant")
    chunks = process_input(source)
    transcript = transcribe_all(chunks, language=language)
    print(f"Raw transcription (first 300 characters): {transcript[:300]}")

    if PARALLEL_RAG:
        with ThreadPoolExecutor(max_workers=1) as pool:
            rag_future = pool.submit(build_rag_chain, transcript)

            summary = summarize(transcript)                   # the one LLM pass (cached)
            action_items = extract_action_items(transcript)   # instant, from cache
            decisions = extract_key_decisions(transcript)     # instant, from cache
            questions = extract_questions(transcript)         # instant, from cache

            rag_chain = rag_future.result()
    else:
        summary = summarize(transcript)
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)
        rag_chain = build_rag_chain(transcript)

    return {
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2: chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")