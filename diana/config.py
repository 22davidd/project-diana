"""
config.py -- all tunable settings for Project Diana live here.

Design notes
------------
Every setting has a sensible default, so Diana runs with zero configuration.
Three layers can override the defaults, in increasing priority:

    1. dataclass defaults        (the values written below)
    2. a JSON file               (./config.json, or --config <path>)
    3. environment variables     (DIANA_<FIELD_NAME>, e.g. DIANA_LOG_LEVEL=DEBUG)

The JSON file may group settings into sections for readability -- see
config.example.json -- and may use "_comment" keys to annotate itself. Both
are flattened before use; the sections are cosmetic. A key set in two places
is an error rather than a silent last-one-wins.

Why a dataclass and not a YAML/TOML file?
  A dataclass gives us type checking, IDE autocomplete and validation for
  free, and it means an unknown key in config.json is a loud error instead of
  a silently ignored typo.

Everything downstream reads settings from a single `Config` instance. No other
module is allowed to hard-code a number like 16000 or 1.5 -- if you want to
tune it, it goes in this file. That rule is what makes the rest of the code
replaceable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Well-known paths. Computed from this file's location so that Diana works no
# matter which directory you launch her from (important once she runs as a
# daemon, since daemons usually start in /).
# --------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DEFAULT_CONFIG_FILE = PROJECT_DIR / "config.json"
DEFAULT_MODELS_DIR = PROJECT_DIR / "models"
DEFAULT_STATE_DIR = PROJECT_DIR / "var"


@dataclass
class Config:
    """The complete runtime configuration for Diana V0."""

    # -- Audio / microphone ------------------------------------------------
    # Vosk's small English models are trained at 16 kHz mono. If you later
    # swap in your own model, change this AND the model you load together.
    sample_rate: int = 16000

    # How much audio we ask the microphone for at once, in milliseconds.
    # Smaller = more responsive but more overhead. 30 ms is a good balance and
    # is the value most streaming ASR systems use.
    block_ms: int = 30

    # `None` means "use the system default input device". Set to an integer
    # index to pin a specific mic (see `python -m diana.main --list-devices`).
    input_device: int | None = None

    # RMS level below which a block is considered silence, while we are IDLE.
    # This is a cheap CPU saver: pure silence is never a wake word, so we skip
    # the recognizer entirely instead of feeding it zeros 33 times a second.
    # 0.008 ~= -42 dBFS, a reasonable "someone is talking" floor for a normal
    # mic in a quiet room. Raise it if the room is noisy and Diana wakes on
    # the fridge hum; lower it if she ignores you when you speak quietly.
    idle_silence_rms: float = 0.008

    # -- Wake word ---------------------------------------------------------
    # These phrases are compiled into a *grammar* (see wakeword.py for why
    # that matters). Keep this list short: every extra phrase makes the
    # recognizer's search space larger and false wake-ups more likely.
    # default_factory rather than a plain list, because dataclasses reject a
    # mutable default -- one shared list would be a bug waiting to happen.
    wake_phrases: list[str] = field(
        default_factory=lambda: ["diana", "hey diana", "ok diana"]
    )

    # After a wake word fires, ignore the wake detector for this long. Without
    # it, the tail of the wake word itself can re-trigger Diana, putting you
    # in a loop of "Yes? ... Yes?".
    wake_cooldown_s: float = 1.2

    # Fire on the recognizer's *partial* hypothesis instead of waiting for the
    # final one. Partial is lower latency but riskier (it can match a word that
    # then changes). Keep it False for reliability in V0.
    wake_on_partial: bool = False

    # -- Command capture (post-wake) ---------------------------------------
    # Discard a little audio right after the wake word. The wake recognizer
    # confirms "Diana" a few hundred ms after you say it, by which point your
    # mouth is already forming the first syllable of the command. Throwing
    # that fragment away is cheaper than trying to repair it in software.
    post_wake_discard_ms: int = 180

    # Give up on a command after this long, even if the user is still talking.
    command_max_s: float = 4.7

    # End the command after this much *continuous* silence. Speech pauses
    # mid-sentence ("turn on... the TV"), so this must be generous enough to
    # survive a normal hesitation.
    command_silence_s: float = 1.6

    # RMS above which the command is considered "the user is talking", used to
    # measure silence for command_silence_s. Lower than idle_silence_rms
    # because after the wake word we want to catch quiet speech.
    command_silence_rms: float = 0.006

    # Print the in-progress transcript as you speak. Great for learning how
    # streaming recognition actually behaves.
    show_partials: bool = True

    # What Diana says when she hears her name. Swap for TTS later via
    # speaker.py -- nothing else needs to change.
    wake_response: str = "Yes, David?"

    # What Diana says after handling a command. This is a debugging control
    # first and a personality second:
    #
    #   "intent" say what actually happened ("lights are now on") (the default)
    #   "echo"  repeat the command verbatim
    #   "ack"   a short fixed confirmation, varied so it does not loop
    #   "none"  say nothing and only print
    #
    # This used to default to "echo" and the reasoning was sound: "echo" is
    # the only mode that can prove the recogniser heard you correctly. If she
    # says "Lice." you have found the bug; if she says "lights are now on" you
    # have learned nothing, because a confident sentence built on a
    # misrecognition looks exactly like a working assistant.
    #
    # It is no longer the default, because there is now a second console line
    # that shows the transcript -- `COMMAND:` -- and the intent reply no longer
    # hides anything. The bug that echo was protecting against is now visible
    # on its own line, and the reply is allowed to be useful. Flip back to
    # "echo" any time the recogniser is in question.
    command_reply: str = "intent"

    # Which command handler to build.
    #
    #   "intents" the rule-based parser: understands commands, prints an
    #            event code, updates device state, says what happened
    #   "print"   the original V0 handler, which repeats what it heard and
    #            does nothing at all
    #
    # "print" is kept because "is she hearing me correctly?" has to stay cheap
    # to ask: with it, the only thing between the microphone and the terminal
    # is the recogniser, so anything wrong is wrong in recognition.
    command_handler: str = "intents"

    # Commands Diana claims to support. Anything that does not match gets
    # command_unsupported_response and status UNSUPPORTED. null (the default)
    # means "everything is supported", which is the honest V0 setting: nothing
    # in this codebase knows what the device layer can do yet, so any list
    # written here would be invented. Set it when a real handler exists.
    command_known: list[str] | None = None

    # Said when a command is outside command_known.
    command_unsupported_response: str = "I don't know how to do that yet."

    # Placeholder intents -- weather, timers -- are understood but have no
    # real source behind them. Said instead of pretending to know.
    command_placeholder_response: str = "I don't have a real source for that yet."

    # Print the `EVENT: <CODE>` line for every handled command.
    command_show_events: bool = True

    # -- Fallback chat (a local LLM) ----------------------------------------
    # When nothing in the grammar matched, the utterance is probably not a
    # command at all -- it is somebody talking to her. Rather than answering
    # "I don't know how to do that yet" to "how do I make lasagne", those go to
    # a small local model instead.
    #
    # "off" is a real setting and not a formality: the rest of Diana works
    # with no network and no server of any kind, and that property is worth
    # keeping testable. Turning this off must leave a fully working assistant.
    chat_enabled: bool = True

    # Ollama model tag. qwen2.5:0.5b is ~400 MB and answers in well under a
    # second on CPU, which is the only speed that works here: the reply is
    # spoken, so a slow answer is a response that arrives after you have
    # already given up. The 1.5b sibling is noticeably better at conversation
    # and noticeably worse at waiting.
    chat_model: str = "qwen2.5:0.5b"

    # Where Ollama is listening. Localhost only -- if this is ever pointed
    # somewhere else, it is sending your microphone audio to a third party.
    chat_host: str = "http://localhost:11434"

    # Give up on the model after this long. A voice assistant that waits ten
    # seconds for a reply has failed more completely than one that says
    # nothing, so this is deliberately short.
    chat_timeout_s: float = 20.0

    # Hard cap on the reply length, in characters, on top of the model's own
    # token limit. Piper reads at roughly three words a second and a person
    # stops listening a good deal sooner than that, so a long answer is not a
    # thorough answer -- it is a reply that gets cut off mid-word by the next
    # wake word. Anything over this is truncated at a sentence boundary.
    chat_max_reply_chars: int = 200

    # Said when a chat reply could not be obtained. Distinct from
    # command_unsupported_response on purpose: that one means "I understood
    # that and could not do it", this one means "the model did not answer",
    # and pretending the two are the same is how you end up debugging a
    # recogniser problem for an afternoon when the fault is in ollama.
    chat_unavailable_response: str = "I didn't catch an answer to that."

    # -- Voice output (text-to-speech) -------------------------------------
    # "console" prints, "piper" speaks with a local neural voice, and "auto"
    # (the default) speaks if a voice is installed and prints otherwise -- so a
    # fresh checkout still runs with no configuration at all, and the moment
    # you fetch a voice she starts talking without you editing anything.
    tts_engine: str = "auto"

    # Piper voice name. The model lives at models/<tts_voice>.onnx; fetch it
    # with `python download_models.py --voices`. The "-medium" voices are about
    # 63 MB and sound like a person; "-low" is smaller and faster but flatter.
    tts_voice: str = "en_US-amy-medium"

    # Output device index, or null for the system default. Null is right almost
    # always, and not a stylistic choice: on Linux the system default
    # (PulseAudio/PipeWire) resamples the voice's 22 kHz output to whatever
    # your sound card runs at, while a raw ALSA device pinned by index almost
    # certainly will not, and simply refuses to open.
    tts_output_device: int | None = None

    # Peak amplitude each utterance is scaled to. Piper normalises every reply
    # so its loudest sample is exactly 1.0, which is exactly where a sound
    # card starts to clip, so this is headroom rather than loudness control.
    tts_peak: float = 0.85

    # Speaking rate, as Piper's length_scale: 1.0 is the voice's natural pace,
    # 1.2 is noticeably faster, 0.8 slower. A slightly slow assistant reads as
    # calm; a fast one reads as frantic.
    tts_length_scale: float = 1.0

    # Also print what she says. The state machine prints your side of the
    # conversation, so without this you lose the "Diana: ..." line from the
    # terminal and from the log.
    tts_echo: bool = True

    # Microphone audio to throw away *after* she finishes talking, in addition
    # to post_wake_discard_ms. The microphone keeps recording while the state
    # machine is blocked in speaker.wait(), so the first blocks after
    # playback are her own voice on its way back in from the speakers. Feeding
    # those to the recogniser is how "Yes?" ends up inside your command.
    post_speak_discard_ms: int = 300

    # -- Models ------------------------------------------------------------
    models_dir: str = str(DEFAULT_MODELS_DIR)

    # Vosk model used for BOTH the wake word (grammar-restricted) and the
    # speech-to-text (free-form). One download, two jobs.
    stt_model: str = "vosk-model-small-en-us-0.15"

    # Logging: DEBUG shows per-block energy and state transitions. INFO is
    # what you want day to day.
    log_level: str = "INFO"

    # -- Daemon / process management ---------------------------------------
    # Where the pid file lives. Its presence is how --stop and --status work.
    pid_file: str = str(DEFAULT_STATE_DIR / "diana.pid")
    log_file: str = str(DEFAULT_STATE_DIR / "diana.log")

    # Refuse to start if another instance is already running. Without this you
    # can end up with two Dianas fighting over the microphone.
    single_instance: bool = True

    # -- Derived values ----------------------------------------------------
    # Never set these directly; they are computed from the fields above.
    @property
    def block_frames(self) -> int:
        """Microphone buffer size in samples (not milliseconds)."""
        return int(self.sample_rate * self.block_ms / 1000)

    @property
    def block_seconds(self) -> float:
        return self.block_ms / 1000.0

    @property
    def stt_model_path(self) -> Path:
        return Path(self.models_dir) / self.stt_model

    @property
    def tts_voice_path(self) -> Path:
        return Path(self.models_dir) / f"{self.tts_voice}.onnx"

    # -- Loading and validation -------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None, require_model: bool = True) -> "Config":
        """Build a Config from defaults, then a JSON file, then the env.

        `require_model=False` skips the model-exists check, which is what lets
        --selftest run before you have downloaded anything.
        """
        cfg = cls()

        # Layer 2: JSON file. An explicitly passed path must exist; the
        # default one is allowed to be absent (that is the normal case).
        explicit = path is not None
        config_path = Path(path) if path is not None else DEFAULT_CONFIG_FILE
        if config_path.exists():
            raw = flatten(json.loads(config_path.read_text()))
            cfg = cfg._merged(raw, source=str(config_path))
        elif explicit:
            raise FileNotFoundError(f"config file not found: {config_path}")

        # Layer 3: environment variables. These arrive as strings, so they
        # need type coercion -- see _coerce.
        cfg = cfg._merged(_env_overrides(), source="environment", coerce=True)

        cfg.validate(require_model=require_model)
        return cfg

    def _merged(self, raw: dict[str, Any], source: str, coerce: bool = False) -> "Config":
        """Return a copy with `raw` applied, rejecting unknown keys loudly."""
        known = {f.name for f in fields(self)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown setting(s) in {source}: {', '.join(sorted(unknown))}. "
                f"Valid names: {', '.join(sorted(known))}"
            )
        for key, value in raw.items():
            if coerce and isinstance(value, str):
                value = self._coerce(key, value)
            setattr(self, key, value)
        return self

    def _coerce(self, name: str, text: str) -> Any:
        """Turn an env-var string into the field's real type.

        Tries JSON first so that DIANA_WAKE_ON_PARTIAL=false becomes a real
        bool, not the non-empty string "false" (which is truthy!).
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def validate(self, require_model: bool = True) -> None:
        """Fail loudly and early on settings that cannot possibly work."""
        if self.sample_rate < 8000:
            raise ValueError("sample_rate must be >= 8000")
        if self.block_ms < 10:
            # Below ~10 ms PortAudio starts dropping buffers under load and
            # the CPU cost per block goes up sharply.
            raise ValueError("block_ms must be >= 10")
        if not self.wake_phrases:
            raise ValueError("wake_phrases must not be empty")
        if self.command_silence_s <= 0 or self.command_max_s <= 0:
            raise ValueError("command_silence_s and command_max_s must be > 0")
        if self.command_max_s <= self.command_silence_s:
            raise ValueError("command_max_s must be greater than command_silence_s")
        if self.command_reply not in ("intent", "echo", "ack", "none"):
            raise ValueError("command_reply must be one of: intent, echo, ack, none")
        if self.command_handler not in ("intents", "print"):
            raise ValueError("command_handler must be one of: intents, print")
        if self.command_known is not None and not isinstance(self.command_known, list):
            raise ValueError("command_known must be a list of strings or null")
        if self.chat_timeout_s <= 0:
            raise ValueError("chat_timeout_s must be > 0")
        if self.chat_max_reply_chars < 40:
            # Below roughly 40 characters there is no room for a sentence, so
            # every reply would be a fragment and the cap would be
            # indistinguishable from the model being broken.
            raise ValueError("chat_max_reply_chars must be >= 40")
        if not self.chat_host.startswith(("http://", "https://")):
            raise ValueError("chat_host must be an http:// or https:// URL")
        if self.tts_engine not in ("auto", "console", "piper"):
            raise ValueError("tts_engine must be one of: auto, console, piper")
        if not 0.0 < self.tts_peak <= 1.0:
            raise ValueError("tts_peak must be in (0, 1]")
        if self.tts_length_scale <= 0:
            raise ValueError("tts_length_scale must be > 0")
        if self.post_speak_discard_ms < 0:
            raise ValueError("post_speak_discard_ms must be >= 0")
        if self.post_speak_discard_ms < self.block_ms:
            # The discard is counted in whole audio blocks, so anything under
            # one block rounds down to zero and the setting silently does
            # nothing. Better to say so.
            print(
                f"[config] warning: post_speak_discard_ms ({self.post_speak_discard_ms}) is "
                f"less than one block ({self.block_ms} ms) and will be rounded down to zero."
            )
        if require_model and not self.stt_model_path.is_dir():
            raise FileNotFoundError(
                f"speech model not found at {self.stt_model_path}\n"
                f"Run:  python download_models.py"
            )
        # "auto" treats a missing voice as a normal situation and falls back to
        # the console speaker, so only an explicit "piper" is a hard error
        # here. build_speaker() announces which one it ended up with.
        if require_model and self.tts_engine == "piper" and not self.tts_voice_path.is_file():
            raise FileNotFoundError(
                f"tts_engine is 'piper' but no voice at {self.tts_voice_path}\n"
                f"Run:  python download_models.py --voices {self.tts_voice}"
            )

    def describe(self) -> str:
        """Multi-line summary printed at startup, so you always know what
        Diana is actually running with."""
        return "\n".join(
            [
                "  sample rate      : {0} Hz  (block {1} ms = {2} samples)".format(
                    self.sample_rate, self.block_ms, self.block_frames
                ),
                "  input device     : {0}".format(
                    "system default" if self.input_device is None else self.input_device
                ),
                "  wake phrases     : {0}".format(", ".join(f'"{p}"' for p in self.wake_phrases)),
                "  wake cooldown    : {0} s".format(self.wake_cooldown_s),
                "  command window   : max {0} s, silence ends after {1} s".format(
                    self.command_max_s, self.command_silence_s
                ),
                "  model            : {0}".format(self.stt_model_path.name),
            ]
        )


def flatten(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested config dict into the flat form Config expects.

    Diana's settings are a flat dataclass, but a hand-written config file is
    much easier to read when it is grouped:

        { "audio": { "sample_rate": 16000 },
          "wake":  { "wake_phrases": ["diana"] } }

    So sections are treated as pure grouping and their contents are lifted to
    the top level. Section names themselves are never settings.

    Two conventions make the file pleasant to write and safe to keep:

    * Any key starting with "_" is ignored, at any depth. That is how
      config.example.json documents itself with "_comment" keys instead of you
      having to keep the prose somewhere else.
    * A name appearing twice -- in two sections, or at the top level *and* in
      a section -- is an error. Silently letting one win would be a bug you
      would only notice as "my setting did nothing".
    """
    out: dict[str, Any] = {}
    seen_in: dict[str, str] = {}

    def walk(node: dict[str, Any], path: str) -> None:
        for key, value in node.items():
            if key.startswith("_"):
                continue  # documentation key, not a setting
            where = f"{path}.{key}" if path else key
            if isinstance(value, dict):
                walk(value, where)
                continue
            if key in seen_in:
                raise ValueError(
                    f"setting {key!r} is set twice: in {seen_in[key]} and in {where}. "
                    f"Settings are global -- pick one section for it."
                )
            seen_in[key] = where
            out[key] = value

    walk(raw, "")
    return out


def _env_overrides() -> dict[str, Any]:
    """Collect DIANA_* environment variables into a plain dict.

    DIANA_SAMPLE_RATE=8000 -> {"sample_rate": "8000"} (coerced later).
    """
    out: dict[str, Any] = {}
    # Compare in the same case on both sides. Getting this wrong is easy and
    # silent: DIANA_COMMAND_MAX_S would simply be ignored, and you would spend
    # an afternoon wondering why your override did nothing.
    valid = {f.name.lower() for f in fields(Config)}
    for key, value in os.environ.items():
        if not key.startswith("DIANA_"):
            continue
        name = key[len("DIANA_") :].lower()
        if name in valid:
            out[name] = value
        else:
            # A typo'd DIANA_ variable is worth mentioning -- otherwise you
            # spend an afternoon wondering why your setting did nothing.
            print(f"[config] ignoring unknown environment variable {key}")
    return out
