"""
wakeword.py -- detecting "Diana" / "Hey Diana", fully offline.

The technique: a closed-vocabulary grammar
------------------------------------------
V0 does not use a dedicated neural wake-word model (like Porcupine or
openWakeWord). Instead it reuses the Vosk speech model in *grammar mode*.

Normally a speech model is free-form: any word can come out. In grammar mode
you hand it a JSON list of the only phrases that are legal, and it constrains
its decoder to that list. If the audio is speech but not one of those phrases,
it emits the special token `[unk]` instead of inventing a word.

    grammar = ["diana", "hey diana", "[unk]"]

This is a good fit for V0 because:
  * it is 100% offline and needs no API key or account
  * it needs no extra model download (we already have the STT model)
  * it needs no training -- "Diana" is a perfectly ordinary English word
  * constraining the vocabulary makes it *faster* and *more accurate* at
    spotting its own name than free-form recognition would be

The trade-off, and the plan for later: a grammar can only ever detect phrases
you listed, and it re-runs a full acoustic model on every block, which is
heavier than a purpose-built tiny CNN. When you want a low-CPU, always-on
detector that runs on a battery-powered node, that is when you swap in a real
wake-word model. `WakeWordDetector` below is the interface for exactly that
swap -- you write a new subclass, change one line in main.py, and nothing
else in the project moves.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import to_pcm16

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WakeEvent:
    """A confirmed wake word."""

    phrase: str
    confidence: float
    at: float
    latency_s: float
    """Seconds from the start of the triggering speech to this event. Lower is
    more responsive. Logged so you can see whether Diana feels sluggish."""


class WakeWordDetector(ABC):
    """Interface every wake-word backend must implement.

    The contract is deliberately tiny: push one audio block in, get an event
    out or None. Everything else -- cooldown, energy gating, what to do after
    a wake -- lives in the state machine, not in the detector. That separation
    is what lets you swap backends without touching the loop.
    """

    name: str = "abstract"

    @abstractmethod
    def detect(self, block: np.ndarray) -> WakeEvent | None:
        """Feed one block of audio. Return a WakeEvent if the wake word fired."""

    def reset(self) -> None:
        """Discard any accumulated state.

        Called after a wake and whenever the assistant goes idle, so the tail
        of one utterance cannot be mistaken for the start of the next.
        """

    def close(self) -> None:
        """Release models and memory."""


class VoskWakeWordDetector(WakeWordDetector):
    """Wake word detection by constraining Vosk to a phrase list.

    How the matching works: Vosk grammar mode returns whatever it heard. If the
    user said "diana turn on the lights" with no pause, we get "diana [unk]"
    rather than a clean "diana". So instead of requiring the result to equal a
    configured phrase, we check whether any configured phrase appears as a
    *contiguous run of words* inside the result. That handles both the clean
    "Diana" case and the run-on case without extra tuning.

    The false-positive problem, and why `require_at_start` is on by default
    -----------------------------------------------------------------------
    A closed grammar has a dirty secret: the decoder is *not allowed* to
    output anything else. When it hears speech that is not one of your
    phrases, it does not return nothing -- it returns whichever legal phrase
    is phonetically closest, padded with `[unk]`.

    Measured on a real 8-second speech recording whose actual transcript was
    "one zero zero zero one / nah no to i know / zero one eight zero three",
    the grammar returned:

        [unk] diana [unk]

    Diana would have woken up in the middle of a phone number. The word
    "diana" was not in the audio at all; the decoder simply had nowhere else
    to put those phonemes.

    So a raw "does the phrase appear in the result" test is not good enough.
    The fix is `require_at_start`: the phrase must be the *first* thing said.
    That rejects the example above (diana is preceded by `[unk]`) while still
    accepting every real use, because a wake word by definition opens the
    conversation -- nobody says "turn on the lights, Diana".

    The asymmetry is deliberate. A false positive is far worse than a false
    negative: one makes Diana interrupt you at random, the other just means
    you say her name again. When in doubt, reject.
    """

    name = "vosk-grammar"

    def __init__(
        self,
        model_path: str | Path,
        sample_rate: int,
        phrases: list[str],
        fire_on_partial: bool = False,
        require_at_start: bool = True,
    ) -> None:
        if not phrases:
            raise ValueError("at least one wake phrase is required")

        self.sample_rate = sample_rate
        self.phrases = [self._normalize(p) for p in phrases]
        self.fire_on_partial = fire_on_partial
        self.require_at_start = require_at_start

        # Precompute each phrase as a word list once, not on every block.
        self._phrase_words = [p.split() for p in self.phrases]

        # `[unk]` is Vosk's catch-all: any speech that is not one of our
        # phrases comes back as this, and we ignore it. It is the entire
        # mechanism that stops Diana waking up on "the kettle just boiled".
        grammar = json.dumps([*self.phrases, "[unk]"])
        self._grammar = grammar

        self._recognizer = None
        self._model = None
        self._speech_started_at: float | None = None

        log.info("loading wake-word model from %s", model_path)
        self._model = self._load_model(Path(model_path))
        self._recognizer = self._make_recognizer()
        log.info("wake word ready; grammar = %s", grammar)

    # -- construction helpers ---------------------------------------------
    @staticmethod
    def _load_model(path: Path):
        try:
            import vosk
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "the 'vosk' package is missing. Install it with: pip install vosk"
            ) from exc
        if not path.is_dir():
            raise FileNotFoundError(f"wake-word model not found: {path}")
        # Model(...) loads the acoustic model; it is memory-hungry, so we load
        # it exactly once and share it via self._model.
        return vosk.Model(str(path))

    def _make_recognizer(self):
        import vosk

        # The third constructor argument is the grammar JSON. This is the line
        # that turns a general speech recognizer into a wake-word detector.
        return vosk.KaldiRecognizer(self._model, self.sample_rate, self._grammar)

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and collapse whitespace, so 'Hey  Diana' == 'hey diana'."""
        return " ".join(text.lower().split())

    # -- WakeWordDetector interface ---------------------------------------
    def detect(self, block: np.ndarray) -> WakeEvent | None:
        if self._recognizer is None:
            raise RuntimeError("detector used before load(); call reset() or construct first")

        # to_pcm16, not block.tobytes(): Kaldi wants 16-bit PCM, PortAudio gave
        # us float32. See audio.to_pcm16 for why this is not optional.
        #
        # feed_input is the streaming entry point. We hand over the raw bytes
        # rather than a file path -- that is what makes continuous recognition
        # possible.
        if self._recognizer.AcceptWaveform(to_pcm16(block)):
            # End of an utterance: the recognizer has committed to a result.
            result = json.loads(self._recognizer.Result())
            return self._maybe_wake(result.get("text", ""))
        # Mid-utterance: only the current best guess so far.
        partial = json.loads(self._recognizer.PartialResult()).get("partial", "")
        if self.fire_on_partial and partial:
            return self._maybe_wake(partial, is_partial=True)
        return None

    def _maybe_wake(self, text: str, is_partial: bool = False) -> WakeEvent | None:
        text = self._normalize(text)
        if not text:
            return None

        words = text.split()
        match = self._find_phrase(words)

        if match is None:
            # `[unk]` and unrelated speech. Note that a *partial* result
            # matching nothing is the common case, so we log at DEBUG only.
            log.debug("wake: no match in %r", text)
            if words and self._speech_started_at is None:
                self._speech_started_at = time.monotonic()
            return None

        now = time.monotonic()
        latency = (now - self._speech_started_at) if self._speech_started_at else 0.0

        event = WakeEvent(
            phrase=match,
            # Vosk's grammar mode does not report a usable per-utterance
            # confidence. Rather than invent a number, we use a fixed value and
            # document it. A real detector will return a real score here, and
            # you can then threshold on it in the state machine.
            confidence=0.9,
            at=now,
            latency_s=round(latency, 3),
        )
        log.info(
            "wake word %r detected (%s) in %.2fs",
            match,
            "partial" if is_partial else "final",
            latency,
        )
        self._speech_started_at = None
        return event

    def _find_phrase(self, words: list[str]) -> str | None:
        """Return the first configured phrase found as a contiguous run.

        Scanning in order of the configured list means the longer, more
        specific phrase ("hey diana") wins over the bare name when both are
        present in the same utterance.

        With `require_at_start` (the default) the run must begin at index 0.
        See the class docstring for why that single rule removes essentially
        all false wake-ups.
        """
        # With require_at_start we only ever look at index 0, because the
        # first word of the utterance decides whether this is a wake at all.
        # Without it we scan every position, which is more permissive and,
        # as the class docstring explains, far more prone to false wake-ups.
        first_position = 0
        last_position = 0 if self.require_at_start else None

        for phrase, phrase_words in zip(self.phrases, self._phrase_words):
            span = len(phrase_words)
            stop_at = len(words) - span + 1
            if last_position is not None:
                stop_at = min(stop_at, last_position + 1)
            for i in range(first_position, max(stop_at, 0)):
                if words[i : i + span] == phrase_words:
                    return phrase
        return None

    def reset(self) -> None:
        """Rebuild the recognizer, throwing away all partial state.

        This is the important one. If we kept the recognizer alive across a
        wake word, the audio that produced "Diana" would still be inside it and
        it would immediately fire again. A fresh recognizer is the clean reset.
        """
        if self._model is not None:
            self._recognizer = self._make_recognizer()
        self._speech_started_at = None

    def close(self) -> None:
        self._recognizer = None
        self._model = None
        log.debug("wake-word detector closed")


class ScriptedWakeWordDetector(WakeWordDetector):
    """A fake detector for tests: fires at chosen block numbers.

    This exists to prove the interface is real. selftest.py uses it to drive
    the whole state machine with no microphone and no model, which is also the
    fastest way to check your own logic when you start changing things.

    `fire_on` takes a list of block indices so a single run can contain
    several complete interactions, which is how we test that Diana really
    does return to idle and wake up again.
    """

    name = "scripted"

    def __init__(self, fire_on: list[int] | int = 1, phrase: str = "diana") -> None:
        self.fire_on = [fire_on] if isinstance(fire_on, int) else list(fire_on)
        self.phrase = phrase
        self.blocks_seen = 0
        self.fired_count = 0
        # Which blocks were offered to us. The IDLE energy gate can skip
        # blocks, so this count is not the same as blocks elapsed.
        self.blocks_offered = 0

    def detect(self, block: np.ndarray) -> WakeEvent | None:
        self.blocks_offered += 1
        if self.blocks_offered in self.fire_on:
            self.fired_count += 1
            return WakeEvent(phrase=self.phrase, confidence=1.0, at=time.monotonic(), latency_s=0.0)
        return None

    def reset(self) -> None:
        # A real detector discards its audio state here. The scripted one has
        # none, but it still counts calls so the tests can assert that reset
        # is being called where it should be.
        self.resets = getattr(self, "resets", 0) + 1
