import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from faster_whisper import WhisperModel
from pydub import AudioSegment

try:
    from faster_whisper import BatchedInferencePipeline   # faster-whisper >= 1.1
except ImportError:
    BatchedInferencePipeline = None

SARVAM_PIECE_SECONDS = 25
SARVAM_WORKERS = int(os.getenv("SARVAM_WORKERS", "6"))   # parallel Sarvam requests

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

_base_model = None
_batched_model = None


def load_model():
    """Load the plain Whisper model once (GPU if possible, else CPU)."""
    global _base_model, _batched_model

    if _base_model is None:
        print(f"Loading faster-whisper model: {WHISPER_MODEL}")
        try:
            _base_model = WhisperModel(
                WHISPER_MODEL, device="cuda", compute_type="float16"
            )
        except Exception as e:
            print(f"CUDA load failed ({e}). Falling back to CPU.")
            _base_model = WhisperModel(
                WHISPER_MODEL, device="cpu", compute_type="int8"
            )

        if BatchedInferencePipeline is not None:
            try:
                _batched_model = BatchedInferencePipeline(model=_base_model)
            except Exception as e:
                print(f"Batched pipeline unavailable ({e}). Using standard model.")
                _batched_model = None

        print("Whisper model loaded successfully")

    return _base_model


def _join(segments) -> str:
    return " ".join(s.text.strip() for s in segments).strip()


def transcribe_chunk_whisper(chunk_path: str) -> str:
    base = load_model()

    # Fast path: batched decoding (several times faster on GPU)
    if _batched_model is not None:
        try:
            segments, _ = _batched_model.transcribe(
                chunk_path,
                language="en",        # skip language detection
                beam_size=1,          # greedy decoding
                batch_size=16,
            )
            return _join(segments)
        except Exception as e:
            print(f"Batched transcription failed ({e}). Retrying with standard model.")

    # Fallback: standard model
    segments, _ = base.transcribe(
        chunk_path,
        language="en",
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return _join(segments)


def _send_to_sarvam(piece_path: str) -> str:
    """Send audio to Sarvam with retries for temporary API failures."""

    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    headers = {"api-subscription-key": SARVAM_API_KEY}
    data = {
        "model": SARVAM_MODEL,
        "with_diarization": "false",
    }

    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        try:
            with open(piece_path, "rb") as audio_file:
                files = {
                    "file": (os.path.basename(piece_path), audio_file, "audio/wav")
                }
                response = requests.post(
                    SARVAM_STT_TRANSLATE_URL,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=120,
                )

            if response.status_code in (429, 500, 502, 503, 504):
                if attempt == max_attempts:
                    print("Sarvam failed after all retry attempts.")
                    response.raise_for_status()

                wait_seconds = min(2 ** attempt, 30)
                print(f"Sarvam returned HTTP {response.status_code}. "
                      f"Retrying in {wait_seconds}s (attempt {attempt}/{max_attempts})...")
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response.json().get("transcript", "").strip()

        except requests.exceptions.Timeout:
            if attempt == max_attempts:
                raise
            wait_seconds = min(2 ** attempt, 30)
            print(f"Sarvam timed out. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

        except requests.exceptions.ConnectionError:
            if attempt == max_attempts:
                raise
            wait_seconds = min(2 ** attempt, 30)
            print(f"Connection error. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    raise RuntimeError("Sarvam transcription failed after all retries.")


def _send_piece_and_cleanup(piece_path: str) -> str:
    try:
        return _send_to_sarvam(piece_path)
    finally:
        if os.path.exists(piece_path):
            os.remove(piece_path)


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts <=30s audio. We split this chunk into
    25-second pieces and send them IN PARALLEL, keeping the original order.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    piece_paths = []
    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")
        piece_paths.append(piece_path)

    print(f"  → Sending {len(piece_paths)} Sarvam pieces "
          f"({SARVAM_WORKERS} in parallel)...")

    # pool.map preserves input order
    with ThreadPoolExecutor(max_workers=SARVAM_WORKERS) as pool:
        texts = list(pool.map(_send_piece_and_cleanup, piece_paths))

    return " ".join(t for t in texts if t).strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to whisper or sarvam depending on language choice.
    - english  -> whisper (local model)
    - hinglish -> sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    language = language.strip().lower()
    full_transcript = []

    engine = "Sarvam AI" if language == "hinglish" else "Whisper"

    print(f"Selected language: {language}")
    print(f"Selected engine: {engine}")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)} using {engine}")

        text = transcribe_chunk(chunk, language=language)

        if text:
            full_transcript.append(text)

    print("Transcription completed")
    return " ".join(full_transcript).strip()