"""
main.py -- Diana's event loop.

The idea
--------
A voice assistant is a state machine wrapped around a microphone. Everything
else in this project exists to feed that machine. The machine has exactly five
states and the transitions between them are the entire behaviour of V0:

              ┌──────────────────────────────────────────┐
              │                                          │
              v                                          │
    ┌──────────────┐   wake word    ┌──────────┐   "Yes?"   ┌───────────┐
    │     IDLE     │───────────────>│ GREETING │───────────>│ LISTENING │
    │ listen for   │                │ say      │            │ speech to │
    │ "Diana"      │                │ response │            │ text      │
    └──────────────┘                └──────────┘            └───────────┘
          ^                                                      │
          │                                        final text /   │
          │                                        timeout        v
          │        done            silence        ┌────────────┐
          └───────────────────────/  \─────────────│ PROCESSING │
                  back to                timeout   │ run the    │
                  waiting                     /    │ handler    │
                                              /     └────────────┘

Writing it as an explicit enum rather than a chain of booleans is the single
most useful decision in the project. It is impossible to be in two states at
once, every timeout has one obvious owner, and when something misbehaves the
log line `IDLE -> LISTENING (reason=...)` tells you exactly what happened.

Try to picture the bug this prevents: a version with an `is_listening` flag and
a `wake_cooldown` timer. The timer expires during the command, the wake word
fires on the tail of the utterance, and now two things are running at once with
no way to tell from the code which. The enum makes that unrepresentable.

Usage
-----
    python -m diana.main                 # run in the foreground
    python -m diana.main --daemon        # detach and run in the background
    python -m diana.main --stop          # stop a running daemon
    python -m diana.main --status        # is it running?
    python -m diana.main --selftest      # test the loop with no microphone
    python -m diana.main --list-devices  # show microphones
"""

from __future__ import annotations

import argparse
import atexit
import importlib.util
import logging
import os
import signal
import sys
import time
from enum import Enum, auto
from pathlib import Path

import numpy as np

from .audio import AudioError, AudioSource, MicrophoneSource, rms, rms_to_db
from .command import (
    CommandHandler,
    CommandStatus,
    IntentCommandHandler,
    PrintCommandHandler,
    command_from_transcription,
)
from .config import Config
from .chat import build_chat
from .speaker import ConsoleSpeaker, PiperSpeaker, Speaker
from .stt import SpeechToText, Transcription, VoskSpeechToText
from .wakeword import VoskWakeWordDetector, WakeEvent, WakeWordDetector

log = logging.getLogger("diana")


class DianaState(Enum):
    IDLE = auto()
    """Listening for the wake word. The only state that runs forever."""

    GREETING = auto()
    """Wake word heard; acknowledging."""

    LISTENING = auto()
    """Capturing the command and running speech-to-text."""

    PROCESSING = auto()
    """Handing the finished transcript to the command handler."""

    STOPPED = auto()
    """Terminal state. The loop exits when it sees this."""


# Why the state changed, for the log line. A state machine without reasons in
# its logs is miserable to debug.
REASON_WAKE = "wake word"
REASON_FINAL = "final transcript"
REASON_SILENCE = "silence timeout"
REASON_MAX_TIME = "max duration reached"
REASON_NO_SPEECH = "no speech detected"
REASON_ERROR = "error"


class Diana:
    """The assistant: owns the state machine and drives every component.

    Note what this class does *not* do: it contains no recognition logic, no
    audio processing and no command parsing. It only decides *when* to ask each
    collaborator to do its job. That is why the collaborators are replaceable.
    """

    def __init__(
        self,
        config: Config,
        audio: AudioSource,
        wake: WakeWordDetector,
        stt: SpeechToText,
        handler: CommandHandler,
        speaker: Speaker,
        max_commands: int = 0,
    ) -> None:
        self.config = config
        self.audio = audio
        self.wake = wake
        self.stt = stt
        self.handler = handler
        self.speaker = speaker
        self.max_commands = max_commands
        """Stop after this many commands. 0 means run forever.

        This lives here rather than being bolted on in main() because the
        command counter is incremented in _finish_command, and a wrapper
        around the handler would have to guess at that to avoid counting each
        command twice."""

        self.state = DianaState.IDLE
        self._state_since = time.monotonic()
        # Until when the wake detector is not to be trusted. Two separate
        # things set it: the cooldown after a real wake word, and the moment
        # she stops talking. The second one is the whole ball game with a real
        # voice -- see _return_to_idle.
        self._wake_blocked_until: float = 0.0
        self._stop_requested = False
        self._blocks = 0
        self._commands_handled = 0
        self._started_at = time.monotonic()

        # Command-capture bookkeeping, all reset on entering LISTENING.
        self._command_started_at = 0.0
        self._speech_seen = False
        self._last_voice_at = 0.0
        self._final_text = ""
        self._final_words = []
        self._discard_blocks = 0
        self._partial_shown = False
        # Tracks whether the last IDLE block was below the energy threshold, so
        # the gate can log the moment it opens rather than 33 times a second.
        self._gated_last = True

    # ------------------------------------------------------------------
    # State machine plumbing
    # ------------------------------------------------------------------
    def _to(self, new_state: DianaState, reason: str = "") -> None:
        """Transition to a new state and log it."""
        if new_state is self.state:
            return
        log.info("%s -> %s%s", self.state.name, new_state.name, f"  ({reason})" if reason else "")
        self.state = new_state
        self._state_since = time.monotonic()

    def _elapsed_in_state(self) -> float:
        return time.monotonic() - self._state_since

    def request_stop(self) -> None:
        """Ask the loop to finish. Safe to call from a signal handler."""
        self._stop_requested = True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> int:
        """Run until stopped. Returns a process exit code."""
        self._print_banner()
        self._install_signal_handlers()

        try:
            self.audio.start()
        except AudioError as exc:
            log.error("%s", exc)
            return 1

        try:
            while not self._stop_requested:
                block = self.audio.read()
                if block is None:
                    # A finite source ran out -- only happens in tests.
                    log.info("audio source finished")
                    break
                self._blocks += 1
                self._dispatch(block)
                if self.state is DianaState.STOPPED:
                    break
        except KeyboardInterrupt:
            log.info("interrupted")
        except AudioError as exc:
            log.error("audio error: %s", exc)
            return 1
        except Exception:
            # Never let the daemon die silently in the background: log the
            # traceback, because nobody is watching the terminal at 2am.
            log.exception("unhandled error in main loop")
            return 1
        finally:
            self.shutdown()

        return 0

    def _dispatch(self, block: np.ndarray) -> None:
        """Route one block of audio to the handler for the current state."""
        if self.state is DianaState.IDLE:
            self._handle_idle(block)
        elif self.state is DianaState.GREETING:
            self._handle_greeting(block)
        elif self.state is DianaState.LISTENING:
            self._handle_listening(block)
        elif self.state is DianaState.PROCESSING:
            # PROCESSING never blocks; it is resolved in the same tick as the
            # transition (see _finish_command), so there is no audio to route.
            pass

    # ------------------------------------------------------------------
    # IDLE: wait for the wake word
    # ------------------------------------------------------------------
    def _handle_idle(self, block: np.ndarray) -> None:
        # Cooldown first. Without it, the audio that just contained "Diana"
        # would still be sitting in the detector and could fire again.
        if time.monotonic() < self._wake_blocked_until:
            return

        # Energy gate. Pure silence cannot be a wake word, so skip the
        # recogniser entirely. On a typical laptop this removes 80-95% of the
        # work, which is what lets Diana sit idle for days without the fan
        # spinning up.
        level = rms(block)
        threshold = self.config.idle_silence_rms
        if level < threshold:
            self._gated_last = True
            return

        # Log only on the *transition* from gated to open, not on every block.
        # At 33 blocks a second a per-block log would be unreadable, but the
        # transition is exactly what you want when tuning the threshold: it
        # tells you the level at which Diana decided you were talking, and
        # the threshold she was comparing it against.
        if self._gated_last:
            self._gated_last = False
            log.debug(
                "voice detected: %.4f (%.1f dBFS) above threshold %.4f (%.1f dBFS)",
                level, rms_to_db(level), threshold, rms_to_db(threshold),
            )

        event = self.wake.detect(block)
        if event is not None:
            self._on_wake(event)

    def _on_wake(self, event: WakeEvent) -> None:
        # Clear the detector *before* anything else, so no audio block can be
        # matched against the "Diana" that is still inside it.
        self.wake.reset()
        self._wake_blocked_until = time.monotonic() + self.config.wake_cooldown_s
        self._to(DianaState.GREETING, f'heard "{event.phrase}"')

    # ------------------------------------------------------------------
    # GREETING: acknowledge, then start listening
    # ------------------------------------------------------------------
    def _handle_greeting(self, block: np.ndarray) -> None:
        # We are in GREETING because the previous block contained the wake
        # word. Discarding that block is deliberate: it is the tail of
        # "Diana", not part of the command.
        self.speaker.speak(self.config.wake_response)

        # Block until playback finishes before listening again. This is the
        # step that stops Diana hearing herself: with a real TTS engine, her
        # voice comes out of the speakers and straight back into the mic, and
        # an assistant that re-triggers on its own output will talk to itself
        # forever. ConsoleSpeaker.wait() is a no-op, so V0 cannot hit this.
        self.speaker.wait()

        self._to(DianaState.LISTENING, REASON_WAKE)
        self._begin_command_capture()

    def _begin_command_capture(self) -> None:
        """Reset all per-utterance state and arm the post-wake discard."""
        self._command_started_at = time.monotonic()
        self._speech_seen = False
        self._last_voice_at = self._command_started_at
        self._final_text = ""
        self._final_words = []
        self._partial_shown = False
        # Two different tails to throw away, and they are the same length of
        # audio for the same reason -- she is the one making both noises:
        #   * the wake word, because you have not finished saying "Diana" when
        #     the detector confirms it
        #   * her own "Yes?", which the microphone picked up out of the
        #     speakers while this state machine was blocked in wait()
        self._discard_blocks = int(
            (self.config.post_wake_discard_ms + self.config.post_speak_discard_ms)
            / self.config.block_ms
        )
        self.stt.start()
        log.debug("command capture started (discarding %d blocks)", self._discard_blocks)

    # ------------------------------------------------------------------
    # LISTENING: capture the command, recognise it as it arrives
    # ------------------------------------------------------------------
    def _handle_listening(self, block: np.ndarray) -> None:
        now = time.monotonic()

        # 1. Hard timeout, checked first so a wedged recognizer cannot hang us.
        if now - self._command_started_at > self.config.command_max_s:
            self._finish_command(REASON_MAX_TIME)
            return

        # 2. Post-wake discard: drop the tail of the wake word.
        if self._discard_blocks > 0:
            self._discard_blocks -= 1
            return

        # 3. Energy tracking, to measure the silence that ends an utterance.
        #    Note we feed the recognizer *every* block, including silence. A
        #    recognizer needs to hear the gaps between words to segment them.
        level = rms(block)
        if level >= self.config.command_silence_rms:
            self._speech_seen = True
            self._last_voice_at = now

        # 4. Silence timeout -- but only once we have heard actual speech, or
        #    a room with a constant hum would start a command that never ends.
        if (
            self._speech_seen
            and now - self._last_voice_at > self.config.command_silence_s
        ):
            self._finish_command(REASON_SILENCE)
            return

        # 5. Hand the audio to the recogniser.
        result = self.stt.feed(block)
        if result is None:
            return
        if result.is_final:
            self._final_text = result.text
            self._final_words = result.words
            self._partial_shown = False
        elif result and not self._partial_shown:
            # Print the live hypothesis in place, so you can watch recognition
            # converge. `\r` rewrites the line instead of scrolling.
            self._show_partial(result)

    def _show_partial(self, result: Transcription) -> None:
        if not self.config.show_partials or not sys.stdout.isatty():
            return
        sys.stdout.write(f"\rYou: {result.text}...   ")
        sys.stdout.flush()
        self._partial_shown = True

    def _clear_partial(self) -> None:
        if self._partial_shown and sys.stdout.isatty():
            sys.stdout.write("\r" + " " * 70 + "\r")
            sys.stdout.flush()
            self._partial_shown = False

    def _finish_command(self, reason: str) -> None:
        """Leave LISTENING. Flush the recognizer, then decide what we heard."""
        # Always flush: the recognizer may be holding the last few words, and
        # FinalResult() is the only way to get them out.
        final = self.stt.finish()
        if final and final.text.strip() and not self._final_text:
            self._final_text = final.text
            self._final_words = final.words

        self._to(DianaState.PROCESSING, reason)
        utterance_s = time.monotonic() - self._command_started_at
        self._clear_partial()

        if not self._final_text.strip():
            # Distinguish "you said nothing" from "we timed out mid-sentence";
            # they deserve different messages and different tuning.
            log.info("nothing recognised (%s)", reason)
            self.speaker.speak("I didn't hear anything.")
            self._return_to_idle()
            return

        # The user-visible line, exactly as in the V0 spec.
        print(f"You: {self._final_text}", flush=True)

        transcript = Transcription(
            text=self._final_text,
            is_final=True,
            confidence=0.0,  # filled in from word timings below
            words=self._final_words,
        )
        if transcript.words:
            transcript.confidence = round(
                sum(w.confidence for w in transcript.words) / len(transcript.words), 3
            )

        command = command_from_transcription(transcript, utterance_s=utterance_s)
        try:
            result = self.handler.handle(command)
        except Exception:
            # A failing handler must not kill the loop. Diana should apologise
            # and go back to listening.
            log.exception("command handler failed")
            self.speaker.speak("Sorry, something went wrong.")
            self._return_to_idle()
            return

        self._commands_handled += 1
        if result.status is not CommandStatus.OK:
            log.info("command status: %s", result.status.value)
        if result.response:
            self.speaker.speak(result.response)

        # Honour --max-commands. Checked here, right after the increment, so
        # the count and the limit can never drift apart.
        if self.max_commands and self._commands_handled >= self.max_commands:
            log.info("handled %d commands, reaching --max-commands; stopping",
                     self._commands_handled)
            self.request_stop()

        self._return_to_idle()

    def _return_to_idle(self) -> None:
        """Back to waiting for the wake word, with clean detector state."""
        # Hold the microphone shut until she has finished talking. This is the
        # single most important line in the file once there is a real voice.
        #
        # The microphone never stops recording while the state machine is
        # blocked here, so the audio sitting in the input buffer when we wake
        # up is the *tail of Diana's own reply* on its way back in from the
        # speakers. Feed that to the wake detector and she answers herself; the
        # classic failure. wait() is a no-op for the console speaker, so this
        # costs nothing until it means something.
        self.speaker.wait()
        self.stt.start()  # discard any leftover recogniser state
        self.wake.reset()
        # And a short block on the wake detector afterwards, which covers the
        # few hundred milliseconds of room noise and speaker tail that the
        # recogniser reset cannot see.
        self._wake_blocked_until = time.monotonic() + self.config.wake_cooldown_s
        self._to(DianaState.IDLE, "ready")

    # ------------------------------------------------------------------
    # Shutdown and reporting
    # ------------------------------------------------------------------
    def _install_signal_handlers(self) -> None:
        """Make SIGINT/SIGTERM a clean exit rather than a traceback.

        This is what makes `--daemon` pleasant: `kill` runs the same shutdown
        path as Ctrl-C, so the pid file gets removed and the microphone is
        released properly.
        """
        def handler(signum, _frame):
            log.info("received %s, shutting down", signal.Signals(signum).name)
            self.request_stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                # signal.signal only works in the main thread. If Diana ever
                # runs her loop inside a worker thread, this fails harmlessly.
                pass

    def _print_banner(self) -> None:
        print("=" * 62, flush=True)
        print("  Project Diana  --  V0 voice loop", flush=True)
        print("=" * 62, flush=True)
        print(self.config.describe(), flush=True)
        print(f"  audio          : {self.audio.describe()}", flush=True)
        print(f"  wake detector  : {self.wake.name}", flush=True)
        print(f"  speech-to-text : {self.stt.name}", flush=True)
        print(f"  command handler: {self.handler.name}", flush=True)
        chat = getattr(self.handler, "chat", None)
        if chat is not None:
            # Worth its own line. "command handler: intents" tells you a rule
            # table is running; it does not tell you whether the thing that
            # answers everything the rule table rejected is alive, and the
            # difference between "(not running)" and a model name is the
            # difference between a working assistant and one that apologises
            # to every question.
            print(f"  fallback chat  : {chat.describe()}", flush=True)
        print(f"  speaker         : {self.speaker.name}", flush=True)
        print("-" * 62, flush=True)
        print(f'  Listening for: {", ".join(self.config.wake_phrases)}', flush=True)
        print("  Ctrl-C to stop.", flush=True)
        print("=" * 62, flush=True)
        print(flush=True)

    def shutdown(self) -> None:
        """Release everything, in reverse order of construction."""
        print(flush=True)
        log.info(
            "shutting down: %d blocks read, %d commands handled, uptime %.1fs",
            self._blocks,
            self._commands_handled,
            time.monotonic() - self._started_at,
        )
        for component in (self.handler, self.speaker, self.stt, self.wake, self.audio):
            try:
                component.close()
            except Exception as exc:  # pragma: no cover - best effort
                log.warning("error closing %s: %s", type(component).__name__, exc)
        self._to(DianaState.STOPPED)


# ======================================================================
# Daemon plumbing
# ======================================================================
def _pid_path(cfg: Config) -> Path:
    return Path(cfg.pid_file)


def _running_pid(cfg: Config) -> int | None:
    """Return the pid of a live instance, or None if there isn't one."""
    path = _pid_path(cfg)
    try:
        pid = int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None
    return pid if _pid_alive(pid) else None


def _pid_alive(pid: int) -> bool:
    """Check for a live process without depending on `ps`."""
    try:
        os.kill(pid, 0)  # signal 0 tests existence and permission only
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else


def _check_no_other_instance(cfg: Config) -> None:
    """Refuse to start if an instance is already running.

    Two Dianas would both try to open the microphone. On Linux the second one
    usually fails outright, but on some systems both succeed and you get doubled
    audio and doubled wake-ups. Cheap to prevent.

    This only *checks*. The pid file is written separately, by
    _claim_instance, and that ordering matters when daemonising -- see
    daemonize().
    """
    if not cfg.single_instance:
        return
    existing = _running_pid(cfg)
    if existing is not None:
        raise SystemExit(
            f"Diana is already running as pid {existing}.\n"
            f"Stop it with:  python -m diana.main --stop"
        )


def _claim_instance(cfg: Config) -> None:
    """Record this process as the running instance.

    Must be called *after* daemonize(), because the whole point is to record
    the pid of the process that will actually be doing the work. Writing it
    before the fork would store the pid of the short-lived parent, and then
    --status and --stop would report "not running" while Diana is very much
    awake and listening.
    """
    if not cfg.single_instance:
        return
    path = _pid_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{os.getpid()}\n")
    # Remove the pid file even on an unhandled exception or sys.exit.
    atexit.register(_remove_lock, cfg)


def _remove_lock(cfg: Config) -> None:
    _pid_path(cfg).unlink(missing_ok=True)


def daemonize(cfg: Config) -> None:
    """Detach from the terminal using the classic double fork.

    Step 1 forks; the parent exits immediately so your shell continues.
    Step 2's child calls setsid() to become a session leader with no
    controlling terminal -- this is what stops Diana being killed by Ctrl-C
    in the shell that launched her. Then it forks again: forking *after*
    setsid guarantees she can never reacquire a terminal.

    Then stdout/stderr are redirected to the log file, because anything
    printed to a closed terminal raises IOError.
    """
    if os.fork() > 0:
        os._exit(0)  # original process: leave immediately

    os.setsid()
    if os.fork() > 0:
        os._exit(0)  # session leader: also leave, leaving a true orphan

    log_path = Path(cfg.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Line-buffered, so log lines appear promptly even though nothing is a tty.
    out = open(log_path, "ab", buffering=0)
    devnull = open(os.devnull, "rb")
    os.dup2(devnull.fileno(), sys.stdin.fileno())
    os.dup2(out.fileno(), sys.stdout.fileno())
    os.dup2(out.fileno(), sys.stderr.fileno())
    # Keep the working directory so relative paths in config keep resolving.
    os.chdir(os.getcwd())


def stop_daemon(cfg: Config) -> int:
    """Send SIGTERM to a running instance."""
    pid = _running_pid(cfg)
    if pid is None:
        _remove_lock(cfg)  # clean up a stale file
        print("Diana is not running.")
        return 1
    print(f"Stopping Diana (pid {pid})...")
    os.kill(pid, signal.SIGTERM)
    # Give it a moment to shut down cleanly before reporting.
    for _ in range(30):
        if not _pid_alive(pid):
            print("Stopped.")
            return 0
        time.sleep(0.1)
    print(f"Process {pid} ignored SIGTERM. Try: kill -9 {pid}")
    return 1


def daemon_status(cfg: Config) -> int:
    """Report whether the daemon is running."""
    pid = _running_pid(cfg)
    if pid is None:
        _remove_lock(cfg)
        print("Diana is not running.")
        return 1
    print(f"Diana is running (pid {pid}).")
    log_file = Path(cfg.log_file)
    if log_file.exists():
        print(f"Log: {log_file}")
    return 0


# ======================================================================
# Wiring: this is the one place that knows about concrete classes
# ======================================================================
def build_speaker(cfg: Config) -> Speaker:
    """Choose a voice, degrading to the console rather than to a failure.

    Sitting next to build_diana() because this is the same kind of thing: the
    one place that knows which concrete class is in use. Everything downstream
    only sees `Speaker`.
    """
    if cfg.tts_engine == "console":
        return ConsoleSpeaker()

    if cfg.tts_engine == "auto" and not _tts_ready(cfg):
        return ConsoleSpeaker()

    # Any failure past this point is a real one -- with "piper" the user asked
    # for a voice, and with "auto" everything that could be checked already was
    # -- so it propagates to main(), which reports "startup failed" and exits
    # rather than running silently mute.
    return PiperSpeaker(
        voice_path=cfg.tts_voice_path,
        output_device=cfg.tts_output_device,
        peak=cfg.tts_peak,
        length_scale=cfg.tts_length_scale,
        echo=cfg.tts_echo,
    )


def _tts_ready(cfg: Config) -> bool:
    """Can we speak at all? Logs exactly what is missing if we cannot.

    Everything checkable without opening an audio device is checked here, so
    that tts_engine="auto" genuinely never fails: a missing voice or a missing
    package is a normal state for a checkout that has not run `make voices`,
    not an error worth refusing to start over.
    """
    if not cfg.tts_voice_path.is_file():
        log.info(
            "no voice at %s, so Diana will print instead of speak. "
            "Fetch one with:  python download_models.py --voices",
            cfg.tts_voice_path,
        )
        return False
    if importlib.util.find_spec("piper") is None:
        log.info(
            "the 'piper-tts' package is not installed, so Diana will print "
            "instead of speak. Install it with:  pip install piper-tts"
        )
        return False
    return True


def _speak_once(cfg: Config, text: str) -> int:
    """Say one line and exit, without touching the microphone.

    The one-line smoke test for the whole voice path: voice model, sound card,
    sample rate, and wait(). If this works and `make run` does not, the problem
    is the voice loop rather than the voice.
    """
    if text.strip() == "-":
        text = sys.stdin.read().strip()
    if not text:
        print("nothing to say", file=sys.stderr)
        return 1

    try:
        speaker = build_speaker(cfg)
    except Exception as exc:
        log.error("voice unavailable: %s", exc)
        return 1

    try:
        speaker.speak(text)
        speaker.wait()
    finally:
        speaker.close()
    return 0


def _text_once(cfg: Config, text: str) -> int:
    """Run sentences through the real command stack. No microphone.

    This is the tool for tuning the grammar. Without it, adding a phrasing
    means saying it out loud to a running assistant and reading a log file to
    find out which rule fired; with it, the same question is a shell command
    that returns in milliseconds and prints what was decided.

    It builds the same handler `build_diana` does, so what it reports is what
    the microphone would have produced. Replies are printed rather than spoken,
    so this stays usable with no voice installed and over ssh.

    A bare "-" reads stdin and treats *each line as a separate utterance*,
    because checking a list of phrasings is the whole point:

        printf 'turn on the lights\\nplay music\\n' | make text TEXT=-
    """
    from .chat import build_chat
    from .intents import CommandParser

    chat = build_chat(cfg)
    if cfg.command_handler == "print":
        handler: CommandHandler = PrintCommandHandler(
            reply="echo" if cfg.command_reply == "intent" else cfg.command_reply,
            known_commands=cfg.command_known,
            unsupported_response=cfg.command_unsupported_response,
        )
    else:
        handler = IntentCommandHandler(
            CommandParser(),
            reply=cfg.command_reply,
            echo=cfg.command_show_events,
            show_events=cfg.command_show_events,
            known_commands=cfg.command_known,
            unsupported_response=cfg.command_unsupported_response,
            placeholder_response=cfg.command_placeholder_response,
            chat=chat,
            chat_unavailable_response=cfg.chat_unavailable_response,
        )

    if text.strip() == "-":
        lines = [ln for ln in sys.stdin.read().splitlines() if ln.strip()]
        if not lines:
            print("nothing to say", file=sys.stderr)
            return 1
    else:
        lines = [text.strip()]

    for line in lines:
        result = handler.handle(command_from_transcription(_as_transcription(line)))
        if result.response:
            print(f"Diana: {result.response}", flush=True)
        print(f"status: {result.status.value}\n", flush=True)
    handler.close()
    return 0


def _as_transcription(text: str) -> Transcription:
    """Wrap typed text as if the recogniser had produced it.

    Confidence is 0.0 on purpose. Nothing in the intent path reads it, and
    typing should not invent a confidence figure the microphone never
    reported -- a `--text` run that printed 0.98 would be a lie you then go
    and trust in a bug report.
    """
    return Transcription(text=text, is_final=True, confidence=0.0, words=[])


def build_diana(cfg: Config) -> Diana:
    """Assemble the concrete components.

    Read this function as the project's dependency wiring diagram. Every
    collaborator arrives through an interface, so replacing the AI later means
    editing only the three lines marked below and nothing else in the codebase.
    """
    audio = MicrophoneSource(
        sample_rate=cfg.sample_rate,
        block_frames=cfg.block_frames,
        device=cfg.input_device,
    )

    # -- the three lines you will change when you add real models --------
    wake = VoskWakeWordDetector(
        model_path=cfg.stt_model_path,
        sample_rate=cfg.sample_rate,
        phrases=cfg.wake_phrases,
        fire_on_partial=cfg.wake_on_partial,
    )
    stt = VoskSpeechToText(
        model_path=cfg.stt_model_path,
        sample_rate=cfg.sample_rate,
        show_partials=cfg.show_partials,
    )
    # The grammar decides commands; the local model only answers the ones that
    # turn out not to be commands. Order matters and is enforced here rather
    # than in the handler: chat is built unconditionally and handed over as a
    # null object when it is switched off, so the "no rule matched" path is
    # the same code either way.
    chat = build_chat(cfg)
    handler: CommandHandler
    if cfg.command_handler == "print":
        handler = PrintCommandHandler(
            reply="echo" if cfg.command_reply == "intent" else cfg.command_reply,
            known_commands=cfg.command_known,
            unsupported_response=cfg.command_unsupported_response,
        )
    else:
        handler = IntentCommandHandler(
            reply=cfg.command_reply,
            echo=True,
            show_events=cfg.command_show_events,
            known_commands=cfg.command_known,
            unsupported_response=cfg.command_unsupported_response,
            placeholder_response=cfg.command_placeholder_response,
            chat=chat,
            chat_unavailable_response=cfg.chat_unavailable_response,
        )
    speaker = build_speaker(cfg)
    # -------------------------------------------------------------------

    return Diana(
        config=cfg,
        audio=audio,
        wake=wake,
        stt=stt,
        handler=handler,
        speaker=speaker,
    )


def setup_logging(cfg: Config) -> None:
    """Configure logging once, sensibly."""
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)-12s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
        force=True,
    )
    # These are chatty at DEBUG and drown out everything else.
    logging.getLogger("sounddevice").setLevel(logging.WARNING)

    # Vosk (and the Kaldi C++ library inside it) writes directly to stderr,
    # bypassing Python's logging entirely. On model load it prints dozens of
    # lines about decoding parameters, orphan nodes and i-vectors -- which
    # buries the one line you actually want to read, the "Yes?".
    #
    # This is done here, once, rather than inside vosk.py's classes, because
    # it is purely a presentation concern and the classes should not care how
    # loud a native library is.
    if not log.isEnabledFor(logging.DEBUG):
        try:
            import vosk

            vosk.SetLogLevel(-1)  # -1 == silent
        except ImportError:
            pass  # setup_logging runs before --selftest needs anything


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="diana",
        description="Project Diana V0 -- offline wake word voice loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="path to a JSON config file")
    parser.add_argument("--daemon", action="store_true", help="run in the background")
    parser.add_argument("--stop", action="store_true", help="stop a running daemon")
    parser.add_argument("--status", action="store_true", help="report daemon status")
    parser.add_argument("--selftest", action="store_true", help="test the loop with no microphone")
    parser.add_argument("--list-devices", action="store_true", help="list audio input and output devices")
    parser.add_argument(
        "--speak", metavar="TEXT",
        help="say TEXT and exit -- test the voice without a microphone (use - for stdin)",
    )
    parser.add_argument(
        "--text", metavar="TEXT",
        help="run TEXT through the command stack and print the result, without a "
             "microphone or any speech (use - for stdin)",
    )
    parser.add_argument(
        "--max-commands", type=int, default=0,
        help="exit after handling N commands (0 = run forever; handy for testing)",
    )
    args = parser.parse_args(argv)

    if args.list_devices:
        from .audio import list_input_devices, list_output_devices

        list_input_devices()
        list_output_devices()
        return 0

    cfg = Config.load(args.config, require_model=not args.selftest)
    setup_logging(cfg)

    if args.selftest:
        from .selftest import run_selftest

        return run_selftest(cfg)

    if args.speak is not None:
        # Deliberately before the single-instance check and the daemon fork:
        # this is a one-shot that touches no microphone and no pid file, so it
        # works even while a Diana is already running. It is the fastest way to
        # answer "is the voice installed and does it sound right?" without
        # having to say her name first.
        return _speak_once(cfg, args.speak)

    if args.text is not None:
        return _text_once(cfg, args.text)

    if args.status:
        return daemon_status(cfg)

    if args.stop:
        return stop_daemon(cfg)

    # Order matters here. Check for another instance *before* forking, so the
    # "already running" message reaches your terminal instead of the log file.
    # Then fork. Then record the pid -- after the fork -- so the pid file
    # names the process that is actually listening. See _claim_instance.
    _check_no_other_instance(cfg)

    if args.daemon:
        daemonize(cfg)
        # logging was pointed at stderr before the fork; rebind it now that
        # stderr has been redirected to the log file.
        setup_logging(cfg)

    _claim_instance(cfg)

    try:
        diana = build_diana(cfg)
    except Exception as exc:
        log.error("startup failed: %s", exc)
        return 1

    diana.max_commands = args.max_commands
    return diana.run()


if __name__ == "__main__":
    sys.exit(main())
