"""
stt.py -- speech to text, fully offline.

The interface
-------------
`SpeechToText` is a *streaming* interface: you push audio blocks in and get
transcripts back as they become available.

    stt.start()
    for block in audio:
        result = stt.feed(block)      # None, or a partial/final transcript
        ...
    final = stt.finish()             # flush whatever is left

Why streaming rather than "hand me a recording, give me a string"? Because
that is how a real-time assistant has to work. You cannot make a user wait
until they stop talking, and a partial transcript lets the UI show progress.
It is also exactly the shape a locally trained model will have: an encoder that
consumes chunks of audio and a decoder that emits tokens incrementally. So the
interface you write against now is the interface you will still want later.

Vosk specifics
--------------
`KaldiRecognizer` has two ways to hand back text:
  * PartialResult() -- the current best guess, changes as more audio arrives
  * Result()        -- a finalized phrase, only after the recognizer decides
                       the speaker paused
  * FinalResult()   -- flush at end of stream

We call PartialResult on every block (cheap) and Result only when
AcceptWaveform reports end-of-utterance.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import to_pcm16

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Word:
    """One recognised word with its timing and confidence.

    Timings are a free by-product of Kaldi and are genuinely useful: they let a
    future version tell "turn on" (verb) from "the TV" (object) by position, and
    they let you slice out the exact audio for a phrase you misheard.
    """

    text: str
    start_s: float
    end_s: float
    confidence: float


@dataclass
class Transcription:
    """What the recogniser currently believes was said."""

    text: str
    is_final: bool
    confidence: float = 0.0
    words: list[Word] = field(default_factory=list)

    def __bool__(self) -> bool:
        """So you can write `if result:` and mean 'there was actual text'."""
        return bool(self.text.strip())


class SpeechToText(ABC):
    """Interface every STT backend must implement."""

    name: str = "abstract"

    @abstractmethod
    def start(self) -> None:
        """Begin a new utterance. Any previous state is discarded."""

    @abstractmethod
    def feed(self, block: np.ndarray) -> Transcription | None:
        """Consume one block of audio.

        Return a Transcription when there is something to report (a partial
        hypothesis or a finalized phrase), or None if the block produced
        nothing worth showing yet. Never block.
        """

    @abstractmethod
    def finish(self) -> Transcription | None:
        """Flush the stream and return any remaining final text."""

    def close(self) -> None:
        """Release models and memory."""


class VoskSpeechToText(SpeechToText):
    """Offline speech-to-text using a Vosk Kaldi model.

    Uses the same model directory as the wake word. The difference is purely
    in how the recognizer is constructed: the wake word passes a grammar
    (closed vocabulary), this one does not (open vocabulary, any words).
    One model download, two configurations.
    """

    name = "vosk"

    def __init__(self, model_path: str | Path, sample_rate: int, show_partials: bool = True) -> None:
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.show_partials = show_partials
        self._model = None
        self._recognizer = None

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"speech model not found: {self.model_path}")

        try:
            import vosk
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("the 'vosk' package is missing: pip install vosk") from exc

        log.info("loading speech model from %s", self.model_path)
        self._model = vosk.Model(str(self.model_path))
        log.info("speech-to-text ready (%s)", self.model_path.name)

    def start(self) -> None:
        import vosk

        self._recognizer = vosk.KaldiRecognizer(self._model, self.sample_rate)
        # Ask Kaldi for word-level timings and confidences. This costs a little
        # extra compute and buys us the Word objects above.
        self._recognizer.SetWords(True)

    def feed(self, block: np.ndarray) -> Transcription | None:
        if self._recognizer is None:
            raise RuntimeError("feed() called before start()")

        # to_pcm16, not block.tobytes(): Kaldi wants 16-bit PCM, PortAudio gave
        # us float32. See audio.to_pcm16 for why this is not optional.
        if self._recognizer.AcceptWaveform(to_pcm16(block)):
            # End of an utterance segment.
            return self._parse(json.loads(self._recognizer.Result()), is_final=True)

        if self.show_partials:
            text = json.loads(self._recognizer.PartialResult()).get("partial", "")
            if text.strip():
                return Transcription(text=text, is_final=False, confidence=0.0)
        return None

    def finish(self) -> Transcription | None:
        if self._recognizer is None:
            return None
        # FinalResult() drains whatever the recognizer is still holding. This
        # is why we call it on timeout: without it the last few words of a
        # command can be silently lost.
        return self._parse(json.loads(self._recognizer.FinalResult()), is_final=True)

    def _parse(self, payload: dict, is_final: bool) -> Transcription | None:
        """Turn Vosk's JSON dict into our Transcription dataclass."""
        text = (payload.get("text") or "").strip()
        if not text:
            return None

        words: list[Word] = []
        for item in payload.get("result", []) or []:
            words.append(
                Word(
                    text=item.get("word", ""),
                    start_s=float(item.get("start", 0.0)),
                    end_s=float(item.get("end", 0.0)),
                    # Kaldi reports this as 0-100; normalise to 0-1 so it is
                    # directly comparable to the wake word's confidence.
                    confidence=float(item.get("conf", 0.0)) / 100.0,
                )
            )

        # Average word confidence is a decent utterance-level proxy. With no
        # word timings (a partial), confidence stays 0.0 and the caller should
        # not trust it.
        confidence = (
            sum(w.confidence for w in words) / len(words) if words else 0.0
        )

        return Transcription(text=text, is_final=is_final, confidence=round(confidence, 3), words=words)

    def close(self) -> None:
        self._recognizer = None
        self._model = None
        log.debug("speech-to-text closed")


class ScriptedSpeechToText(SpeechToText):
    """A fake STT for tests: returns a fixed phrase after N blocks.

    Pairs with ScriptedWakeWordDetector to exercise the full state machine
    without a microphone, a model, or a network.
    """

    name = "scripted"

    def __init__(self, text: str = "turn on the TV", emit_after_blocks: int = 2) -> None:
        self.text = text
        self.emit_after_blocks = emit_after_blocks
        self.blocks_seen = 0
        self.started = False
        # How many times start() was called. Tests use this to confirm that
        # Diana re-initialised the recogniser on the way back to idle, which is
        # what stops one utterance bleeding into the next.
        self.starts = 0

    def start(self) -> None:
        self.blocks_seen = 0
        self.started = True
        self.starts += 1

    def feed(self, block: np.ndarray) -> Transcription | None:
        if not self.started:
            raise RuntimeError("feed() called before start()")
        self.blocks_seen += 1
        if self.blocks_seen == self.emit_after_blocks:
            return Transcription(
                text=self.text,
                is_final=True,
                confidence=1.0,
                words=[Word(text=self.text, start_s=0.0, end_s=1.0, confidence=1.0)],
            )
        return None

    def finish(self) -> Transcription | None:
        return None
