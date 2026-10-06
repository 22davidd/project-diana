"""
speaker.py -- how Diana talks back.

In V0 she prints to the terminal; with a voice model she speaks. The main loop
only ever calls `speaker.speak(...)` and `speaker.wait()`, so swapping the two
is one function in main.py and nothing else moves.

Why this is separated from command.py: a command handler decides *what* to say
("The TV is already on"), the speaker decides *how* to say it. Keeping them
apart means you can test your parser's dialogue logic with no audio output at
all, which is far easier when you are iterating.

The one method that matters is `wait()`
-------------------------------------
`wait()` is not a nicety. The main loop calls it before it lets the microphone
go back to the wake word, and it is the only thing standing between you and this:

    Diana hears her own voice through the speakers, believes you said her
    name, says "Yes?", hears "Yes?" through the speakers, says "Yes?"...

With the console speaker there is nothing to wait for, which is exactly why V0
could not have this bug. With a real voice it is the first thing you hit, and
`PiperSpeaker` below is written around the two things it needs to get right:
`speak()` must not block, and `wait()` must not return early.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class SpeakerError(RuntimeError):
    """Raised when the voice cannot be loaded or the output device opened."""


class Speaker(ABC):
    """Interface for anything that can vocalise text."""

    name: str = "abstract"

    @abstractmethod
    def speak(self, text: str) -> None:
        """Say something. Should not block for long -- see note below."""

    def wait(self) -> None:
        """Block until any speech has finished playing.

        The main loop calls this before switching the microphone back to the
        wake word. This is the single most important detail in a voice
        assistant: if you go back to listening while Diana is still talking,
        her own voice through the speakers re-triggers the wake word and you
        get an infinite conversation with yourself. With a console speaker
        there is nothing to wait for, which is why V0 has no such bug.
        """

    def close(self) -> None:
        """Release the audio device."""


class ConsoleSpeaker(Speaker):
    """Prints to stdout with a "Diana:" prefix. The V0 speaker."""

    name = "console"

    def __init__(self, stream=sys.stdout) -> None:
        self._stream = stream

    def speak(self, text: str) -> None:
        if not text:
            return
        print(f"Diana: {text}", file=self._stream, flush=True)


class PiperSpeaker(Speaker):
    """Offline neural TTS, played on a background thread.

    Why piper
    ---------
    It is a single ONNX file per voice (about 60 MB) that runs on CPU, needs no
    account, no API key and no network at runtime -- the same properties the
    rest of the project is built on. On this laptop it synthesises roughly
    *ten times faster than real time* (measured: a 2.3 s reply generated in
    0.20 s, i.e. RTF 0.09), which is the whole reason a neural voice is viable
    inside a voice loop at all.

    Why a worker thread
    -------------------
    Since synthesis is ~10x faster than playback, calling it inline would
    "work" -- and would also add the full generation time to the latency of
    every single reply, then dump all of that audio on the sound card at once.
    Piper's `synthesize()` is a generator of audio chunks, so the natural thing
    is to feed the queue as chunks appear and let the first one play
    immediately. `speak()` then returns in about a millisecond, as the
    interface promises, and `wait()` is the only call that blocks.

    Two details that are easy to get wrong
    --------------------------------------
    * Headroom. Piper normalises every utterance so its loudest sample is
      exactly 1.0, which is precisely where a sound card starts to clip. Every
      reply is therefore scaled to `peak` (0.85 by default).

    * Frame accounting. `wait()` cannot just join the worker thread: the worker
      being finished only means the audio has been *generated*, not that it has
      been *played*. So every chunk is counted on its way into the queue and
      every sample is counted on its way out of the PortAudio callback, and
      wait() returns when the two agree. Getting this wrong means `wait()`
      returns while she is still talking, and the assistant listens to itself.

    Barge-in: calling `speak()` while she is already talking abandons the
    current line and says the new one instead. A single pending slot is
    enough for that -- there is never more than one thing worth saying.
    """

    name = "piper"

    def __init__(
        self,
        voice_path: str | Path,
        output_device: int | None = None,
        peak: float = 0.85,
        length_scale: float = 1.0,
        echo: bool = True,
    ) -> None:
        if not 0.0 < peak <= 1.0:
            raise ValueError("peak must be in (0, 1]")

        self.voice_path = Path(voice_path)
        self.peak = peak
        self.length_scale = length_scale
        self.echo = echo
        self.output_device = output_device
        self.name = f"piper[{self.voice_path.stem}]"

        self._voice = self._load_voice()
        self.sample_rate = int(self._voice.config.sample_rate)
        self._syn_config = self._make_syn_config()

        # Playback state. `_queue` and the counters are the hand-off between the
        # synthesis thread and PortAudio's real-time callback; `_partial` is
        # owned exclusively by the callback and never by the worker.
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._partial: np.ndarray | None = None
        self._frames_queued = 0
        self._frames_played = 0
        self._pending: str | None = None
        self._generation = 0
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closing = False
        self._closed = False

        self._stream = self._open_output()

        log.info(
            "voice ready: %s at %d Hz -> %s",
            self.voice_path.name, self.sample_rate,
            "default output" if output_device is None else f"device {output_device}",
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _load_voice(self):
        if not self.voice_path.is_file():
            raise SpeakerError(
                f"Piper voice not found: {self.voice_path}\n"
                f"Fetch it with:  python download_models.py --voices"
            )
        try:
            from piper import PiperVoice
        except ImportError as exc:
            raise SpeakerError(
                "the 'piper-tts' package is missing. Install it with:\n"
                "    pip install piper-tts\n"
                "or set tts_engine to \"console\" to go back to printing."
            ) from exc
        # Loading costs ~2 s and hundreds of MB, so it happens here, once, at
        # startup -- never inside speak(), which runs on the audio timeline.
        return PiperVoice.load(self.voice_path)

    def _make_syn_config(self):
        from piper import SynthesisConfig

        # length_scale: Piper's name for speaking rate, where larger is slower.
        # 1.0 is the voice's natural pace. Everything else stays at the
        # defaults, which were tuned against the original voice training.
        return SynthesisConfig(length_scale=self.length_scale)

    def _open_output(self):
        import sounddevice as sd

        try:
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                device=self.output_device,
                channels=1,
                dtype="float32",
                callback=self._on_audio,
            )
            stream.start()
        except Exception as exc:
            hint = ""
            if self.output_device is not None:
                try:
                    native = int(sd.query_devices(self.output_device)["default_samplerate"])
                    hint = (
                        f"\nThat device runs natively at {native} Hz and will not be resampled. "
                        f'Use the system default ("default", "pulse" or "pipewire"), or leave '
                        f"tts_output_device unset."
                    )
                except Exception:
                    pass
            raise SpeakerError(
                f"could not open audio output for {self.voice_path.name} "
                f"(device={self.output_device!r}): {exc}{hint}"
            ) from exc
        return stream

    # ------------------------------------------------------------------
    # Speaker interface
    # ------------------------------------------------------------------
    def speak(self, text: str) -> None:
        """Queue a line for playback and return immediately."""
        text = (text or "").strip()
        if not text or self._closed:
            return

        with self._lock:
            self._generation += 1
            self._pending = text
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._run, name="diana-tts", daemon=True
                )
                self._worker.start()

        if self.echo:
            # The transcript is printed by the state machine, but the reply is
            # not, and without this you lose the "Diana: ..." line from the
            # terminal and the log -- which is the thing you actually want to
            # read when the speakers are muted or the room is quiet.
            print(f"Diana: {text}", flush=True)
        log.info("speaking: %r", text)

    def wait(self) -> None:
        """Return only once the last queued sample has been played."""
        # Two conditions, and both are needed: the worker being finished only
        # means generation is done, and an empty queue only means the PortAudio
        # callback has caught up with it. The loop re-checks after every join so
        # that a speak() arriving mid-wait is still waited for.
        while True:
            worker = self._worker
            if worker is not None and worker.is_alive():
                worker.join(timeout=0.02)
                continue
            if self._frames_queued > self._frames_played:
                time.sleep(0.01)
                continue
            return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._closing = True

        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

        # Drop anything still queued. abort() rather than stop() because we want
        # the sound card silenced immediately, not after the buffer plays out.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._frames_queued = self._frames_played

        try:
            self._stream.abort()
            self._stream.close()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            log.warning("error closing audio output: %s", exc)
        log.debug("voice output closed")

    # ------------------------------------------------------------------
    # Synthesis thread
    # ------------------------------------------------------------------
    def _run(self) -> None:
        """Generate and queue audio until there is nothing left to say."""
        while not self._closing:
            with self._lock:
                text, self._pending = self._pending, None
                generation = self._generation
            if not text:
                return

            started = time.monotonic()
            try:
                for chunk in self._voice.synthesize(text, self._syn_config):
                    if self._closing or generation != self._generation:
                        # A newer line replaced this one mid-sentence. Stop
                        # generating now; whatever is already queued still plays.
                        break
                    self._push(self._as_playable(chunk))
            except Exception:
                # A failed voice must not take the assistant down with it, and
                # the user is owed the words even if the audio did not work.
                log.exception("text-to-speech failed; falling back to the console")
                ConsoleSpeaker().speak(text)
                continue
            log.debug("generated %r in %.2fs", text[:40], time.monotonic() - started)

    def _as_playable(self, chunk) -> np.ndarray:
        """One synthesis chunk as a contiguous float32 array with headroom."""
        samples = np.ascontiguousarray(chunk.audio_float_array, dtype=np.float32)
        if samples.size:
            samples = samples * np.float32(self.peak)
        return samples

    def _push(self, samples: np.ndarray) -> None:
        # The _closing check is what makes close() safe: without it, a
        # synthesis that is still running when we shut down could push a chunk
        # after the queue was drained, and a later wait() would wait for audio
        # that no device is going to play.
        if samples.size == 0 or self._closing:
            return
        with self._lock:
            self._frames_queued += samples.size
        self._queue.put(samples)

    # ------------------------------------------------------------------
    # PortAudio callback
    # ------------------------------------------------------------------
    def _on_audio(self, outdata, frames, _time_info, status) -> None:
        """Pull the next samples for the sound card. Runs on a real-time thread.

        The rules for this function: never block, never log at INFO, and never
        allocate more than you must. Everything it needs is preallocated.
        """
        if status:  # underflow/overflow flags from PortAudio itself
            log.debug("output status: %s", status)

        if self._partial is None or self._partial.size == 0:
            try:
                self._partial = self._queue.get_nowait()
            except queue.Empty:
                outdata[:] = 0.0  # nothing queued: output silence, do not stop
                return

        take = min(frames, self._partial.size)
        outdata[:take, 0] = self._partial[:take]
        if take < frames:
            outdata[take:, 0] = 0.0
        self._partial = self._partial[take:] if take < self._partial.size else None
        self._frames_played += take
