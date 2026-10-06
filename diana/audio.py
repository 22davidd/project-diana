"""
audio.py -- the microphone layer.

Why this file exists
--------------------
Two components need to hear the microphone: the wake-word detector and the
speech-to-text engine. If each of them opened PortAudio independently you would
get two competing input streams, doubled CPU, and on Linux often a hard failure
because PulseAudio/PipeWire will not hand the same device to two processes
twice.

So `AudioSource` is the *single owner* of the microphone. It hands out
fixed-size blocks of audio on demand, and the state machine in main.py decides
who gets to look at each block. One mic, many consumers, no conflicts.

The `AudioSource` interface is also the seam that lets you test the whole
system with no hardware at all (see selftest.py), or later swap the mic for a
network stream coming from a Raspberry Pi node in another room.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger(__name__)


class AudioError(RuntimeError):
    """Raised when the microphone cannot be opened or read."""


# --------------------------------------------------------------------------
# Signal helpers -- tiny maths that the rest of the codebase reuses.
# --------------------------------------------------------------------------
def rms(block: np.ndarray) -> float:
    """Root-mean-square amplitude of a block, in the range 0.0 - 1.0.

    RMS is our loudness meter. We use it instead of peak amplitude because peak
    is dominated by single-sample spikes (clicks, pops, keyboard noise) while
    RMS reflects sustained energy, which is what "someone is speaking" means.

    0.0 is digital silence, 1.0 is a full-scale square wave. Typical speech at
    a normal mic gain sits around 0.01 - 0.1.
    """
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))


def rms_to_db(level: float) -> float:
    """Convert an RMS amplitude to decibels, for human-readable logging.

    Log scale matters: the difference between 0.001 and 0.002 is obvious in dB
    but invisible on a linear scale, which is exactly the range where you tune
    a silence threshold.
    """
    if level <= 1e-10:
        return -120.0
    return 20.0 * float(np.log10(level))


def to_pcm16(block: np.ndarray) -> bytes:
    """Convert a float32 block in [-1.0, 1.0] into 16-bit PCM bytes.

    This function is the reason the whole pipeline works, and getting it wrong
    fails in the most confusing way possible.

    PortAudio hands us float32: one 4-byte IEEE float per sample, normalised
    to [-1.0, 1.0]. That is a lovely format for computing RMS, applying gain
    and feeding a neural network that expects tensors.

    Kaldi -- and therefore Vosk, and therefore every model you are likely to
    train yourself on -- expects signed 16-bit PCM: one 2-byte integer per
    sample. The formats have the same sample rate and the same duration but
    completely different bytes.

    So you must NOT do `block.tobytes()`. That hands Kaldi 4 bytes per sample
    and it will happily decode them as 2 bytes per sample: you get twice the
    audio, wrong amplitude, and nonsense words. The recogniser does not raise,
    does not warn, and returns confident-looking garbage or nothing at all.
    That is a genuinely painful bug to track down, which is why it is
    isolated in one small function with this explanation attached.
    """
    # Clip first: values above 1.0 wrap around on conversion to int16, and a
    # loud clap would otherwise turn into a burst of noise.
    clipped = np.clip(block, -1.0, 1.0)
    # 32767 rather than 32768, so a full-scale positive sample does not become
    # -32768 (the int16 minimum) and click.
    return (clipped * 32767.0).astype(np.int16).tobytes()


class AudioSource(ABC):
    """A source of fixed-size, mono, float32 audio blocks.

    Subclasses must guarantee:
      * every block returned by `read()` has exactly `block_frames` samples
      * sample rate is `sample_rate`
      * samples are normalized floats in [-1.0, 1.0]
    """

    def __init__(self, sample_rate: int, block_frames: int) -> None:
        self.sample_rate = sample_rate
        self.block_frames = block_frames

    @abstractmethod
    def start(self) -> None:
        """Open the underlying device. Must be safe to call once."""

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the next block of audio, or None if the source is finished.

        Returning None (rather than raising) is how a synthetic or finite
        source signals "no more audio", which lets the main loop exit cleanly
        instead of treating it as an error.
        """

    @abstractmethod
    def stop(self) -> None:
        """Release the device. Must be safe to call more than once."""

    def close(self) -> None:
        """Alias for stop().

        main.py's shutdown path calls close() on every collaborator uniformly.
        Giving AudioSource the same method as the other components means the
        cleanup loop does not need to know which of them is the microphone.
        """
        self.stop()

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier, used in log lines."""

    def __enter__(self) -> "AudioSource":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def describe(self) -> str:
        return (
            f"{self.name} @ {self.sample_rate} Hz, "
            f"{self.block_frames} samples/block ({self.block_frames / self.sample_rate * 1000:.0f} ms)"
        )


class MicrophoneSource(AudioSource):
    """The real thing: a PortAudio input stream via the `sounddevice` package.

    We deliberately ask PortAudio for exactly one block at a time
    (blocksize=block_frames) and call read() in a loop, rather than opening a
    callback-driven stream. A callback runs on a real-time audio thread where
    you must never allocate, log, or take a lock; doing recognition on the main
    thread keeps all the failure modes ordinary and debuggable. At a 30 ms
    block size the latency cost is irrelevant for a voice assistant.
    """

    def __init__(self, sample_rate: int, block_frames: int, device: int | None = None) -> None:
        super().__init__(sample_rate, block_frames)
        self.device = device
        self._stream = None
        self._name = f"microphone[{device if device is not None else 'default'}]"
        self.blocks_read = 0
        self.overflow_count = 0

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - depends on install
            raise AudioError(
                "the 'sounddevice' package is missing. Install it with:\n"
                "    pip install sounddevice\n"
                "On Linux you may also need the PortAudio system library:\n"
                "    sudo apt install libportaudio2"
            ) from exc

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_frames,
                device=self.device,
                channels=1,
                dtype="float32",  # matches what numpy and Vosk both want
                latency="low",  # trade a little robustness for responsiveness
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioError(
                f"could not open the microphone (device={self.device!r}): {exc}\n"
                f"Run 'python -m diana.main --list-devices' to see the available inputs."
            ) from exc

        actual = self._stream.samplerate
        if actual != self.sample_rate:
            # The driver silently resampling would make every recognition
            # result quietly worse. Fail instead.
            self.stop()
            raise AudioError(
                f"device is running at {actual} Hz but config.sample_rate is {self.sample_rate}. "
                f"Set sample_rate to match, or choose a different input device."
            )

        self.blocks_read = 0
        self.overflow_count = 0
        log.info("microphone open: %s", self.describe())

    def read(self) -> np.ndarray | None:
        if self._stream is None:
            raise AudioError("read() called before start()")
        try:
            data, overflowed = self._stream.read(self.block_frames)
        except Exception as exc:
            raise AudioError(f"microphone read failed: {exc}") from exc

        if overflowed:
            # Overflow means the audio thread fell behind and dropped samples.
            # Worth counting: if this climbs, the block size or the CPU is
            # wrong, and that is a real cause of "Diana stopped hearing me".
            self.overflow_count += 1
            if self.overflow_count % 50 == 1:
                log.warning("microphone overflow x%d (dropped audio)", self.overflow_count)

        self.blocks_read += 1
        # sounddevice returns shape (block_frames, channels) float32. Flatten
        # to 1-D and guarantee C-contiguity: Vosk's tobytes() needs a plain
        # contiguous buffer, and a non-contiguous view would copy silently.
        return np.ascontiguousarray(data.reshape(-1), dtype=np.float32)

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            log.warning("error closing microphone: %s", exc)
        finally:
            self._stream = None
            log.info("microphone closed after %d blocks", self.blocks_read)


def list_input_devices() -> None:
    """Print every input device PortAudio can see. Wired to --list-devices."""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is not installed: pip install sounddevice")
        return

    print("PortAudio version:", sd.get_portaudio_version()[1])
    print()
    print("Index  Name                                  Channels  Default")
    print("-----  ------------------------------------  --------  -------")
    found = False
    for idx, dev in enumerate(sd.query_devices()):
        # max_input_channels > 0 means the device can record.
        if dev["max_input_channels"] < 1:
            continue
        found = True
        is_default = "*" if dev.get("default_samplerate") and sd.default.device[0] == idx else " "
        print(
            f"{idx:>5}  {dev['name'][:36]:<36}  {dev['max_input_channels']:>8}  {is_default:>7}"
        )
    if not found:
        print("\nNo input devices found. Check that a microphone is connected.")
    print("\nSet the device in config.json:  \"input_device\": <index>")


def list_output_devices() -> None:
    """Print every output device, for choosing `tts_output_device`.

    The reason this needs its own listing is the sample rate. A Piper voice
    outputs at 22.05 kHz, and a device opened at a rate it does not support
    simply fails to open -- so the rate column below is the thing to read, not
    the channel count. The system default is the safe choice: on Linux,
    PulseAudio/PipeWire resample for you.
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is not installed: pip install sounddevice")
        return

    print()
    print("Output devices")
    print("Index  Name                                  Channels  Rate     Default")
    print("-----  ------------------------------------  --------  -------  -------")
    found = False
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] < 1:
            continue
        found = True
        is_default = "*" if sd.default.device[1] == idx else " "
        rate = int(dev["default_samplerate"])
        print(
            f"{idx:>5}  {dev['name'][:36]:<36}  {dev['max_output_channels']:>8}  "
            f"{rate:>6} Hz  {is_default:>7}"
        )
    if not found:
        print("\nNo output devices found. Check that speakers are connected.")
    print("\nSet the device in config.json:  \"tts_output_device\": <index>")
    print("Leave it null for the system default, which is what you usually want.")
