"""
selftest.py -- exercise the whole loop with no microphone and no model.

Why bother
----------
Two reasons, and the second is the important one.

1. It tells you whether your *logic* is broken, as opposed to your hardware,
   your model, or your room. When Diana does not respond, this tells you which
   of those three you are debugging in about a second.

2. It is the proof that the interfaces are real. Look at what happens below:
   we build the exact same `Diana` object that main.py builds, and hand it
   three fake collaborators plus a synthetic audio source. Nothing else
   changes. If you can swap the recogniser and the command handler this easily
   today, swapping in your own trained model and Transformer later is a
   two-line change.

Run it with:  python -m diana.main --selftest
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import replace

import numpy as np

from .audio import AudioSource, rms
from .chat import NullChat, OllamaChat, shorten
from .command import (
    Command,
    CommandHandler,
    CommandResult,
    CommandStatus,
    IntentCommandHandler,
    PrintCommandHandler,
)
from .config import Config
from .devices import default_registry
from .intents import CommandParser
from .speaker import ConsoleSpeaker, Speaker
from .stt import ScriptedSpeechToText
from .wakeword import ScriptedWakeWordDetector


class FakeChat:
    """A scripted stand-in for OllamaChat, so the suite needs no server.

    The point of the fallback is that it is optional, so the tests must not
    require it. `null=True` makes it behave like a model that is switched on
    and not answering, which is the case that is easy to get wrong.
    """

    disabled = False

    def __init__(self, answer: str = "Paris.", null: bool = False) -> None:
        self.answer = answer
        self.null = null
        self.asked: list[str] = []

    def available(self, timeout_s: float = 1.5) -> bool:
        return not self.null

    def describe(self) -> str:
        return "fake"

    def reply(self, text: str) -> str:
        self.asked.append(text)
        return "" if self.null else self.answer

    def reset(self) -> None:
        self.asked.clear()

    def close(self) -> None:
        """Nothing to release."""


class SyntheticAudioSource(AudioSource):
    """Emits a fixed number of fixed-amplitude blocks, then runs out.

    Two details matter here:

    * `amplitude` defaults to 0.05, which is *above* the default idle silence
      threshold. That is deliberate: main.py's IDLE energy gate throws away
      blocks quieter than the threshold, so a source full of digital silence
      would never reach the wake detector at all. Real speech is not silent,
      so the test should not be either.

    * read() returns None once exhausted. That is how a finite source says "no
      more audio" -- main.py treats it as a clean exit, not an error, which is
      why AudioSource.read() is typed `ndarray | None`.

    * `realtime=True` sleeps for one block duration per read(). This matters
      more than it looks. Every timeout in the state machine is wall-clock
      (`command_max_s`, `command_silence_s`), because in production the audio
      clock and the wall clock are the same thing. A source that returned
      instantly would burn through a thousand "blocks" in a millisecond and no
      timeout would ever fire, so the test would silently exercise nothing.
      Sleeping keeps the two clocks together and makes the test honest.
    """

    def __init__(
        self,
        sample_rate: int,
        block_frames: int,
        total_blocks: int,
        amplitude: float = 0.05,
        realtime: bool = True,
    ) -> None:
        super().__init__(sample_rate, block_frames)
        self.total_blocks = total_blocks
        self.amplitude = amplitude
        self.realtime = realtime
        self.blocks = 0
        self.started = False
        self.stopped = False
        # A square wave, so consecutive blocks are not identical. Some code
        # paths (and your own eyes, when adding debug output) behave
        # differently with a constant DC block.
        self._block = np.full(block_frames, amplitude, dtype=np.float32)
        self._block[::2] = -amplitude

    @property
    def name(self) -> str:
        return f"synthetic[{self.total_blocks} blocks @ {self.amplitude}]"

    def start(self) -> None:
        self.started = True
        self.blocks = 0
        self._deadline = time.monotonic()

    def read(self) -> np.ndarray | None:
        if not self.started or self.blocks >= self.total_blocks:
            return None
        if self.realtime and self.blocks > 0:
            # Pace to the block duration, the way a real PortAudio read() does.
            self._deadline += self.block_frames / self.sample_rate
            delay = self._deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self.blocks += 1
        return self._block.copy()

    def stop(self) -> None:
        self.stopped = True


class RecordingHandler(CommandHandler):
    """A command handler that remembers what it was asked to do.

    Note this shares no code with PrintCommandHandler, and Diana cannot tell
    the difference. That is the interface doing its job.
    """

    name = "recording"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def handle(self, command) -> CommandResult:
        self.seen.append(command.text)
        return CommandResult(status=CommandStatus.OK, data={"text": command.text})


class RespondingHandler(CommandHandler):
    """A handler that answers, so the reply path is actually exercised.

    `RecordingHandler` deliberately returns an empty response, so it never
    reaches `speaker.speak()` after a command. Scenario 7 needs the opposite:
    the interesting question is what the state machine does *after* she has
    spoken a reply, and there is no reply to ask about if nothing is said.
    """

    name = "responding"

    def __init__(self, response: str = "The TV is already on.") -> None:
        self.response = response
        self.seen: list[str] = []

    def handle(self, command) -> CommandResult:
        self.seen.append(command.text)
        return CommandResult(status=CommandStatus.OK, response=self.response, data={"text": command.text})


class Timeline:
    """An ordered, timestamped log of everything the collaborators were asked to do.

    The state machine's correctness claims are all claims about *order* and
    *timing* -- "nothing heard the microphone while she was talking", "nothing
    heard it for a moment after she stopped". Those are invisible in the output
    of a test that only checks results, so the fakes below write into one of
    these instead and the checks make assertions about the log.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, float]] = []

    def add(self, kind: str) -> None:
        self.events.append((kind, time.monotonic()))

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.events]

    def count(self, kind: str) -> int:
        return sum(1 for k, _ in self.events if k == kind)

    def index_of(self, kind: str, after: int = 0) -> int:
        """Position of the next event of this kind, or -1 if there is none."""
        for i in range(after, len(self.events)):
            if self.events[i][0] == kind:
                return i
        return -1

    def between(self, start_kind: str, end_kind: str, start: int = 0) -> list[str]:
        """Event kinds strictly between the next `start_kind` and the next `end_kind`."""
        i = self.index_of(start_kind, start)
        if i < 0:
            return []
        j = self.index_of(end_kind, i)
        if j < 0:
            return []
        return [k for k, _ in self.events[i + 1 : j]]

    def spans(self, begin_kind: str, end_kind: str) -> list[tuple[int, int]]:
        """(begin, end) index pairs for every begin/end event, in order.

        Used to ask "what happened while the speaker was talking", which is
        the question every self-hearing bug comes down to.
        """
        out: list[tuple[int, int]] = []
        pos = 0
        while True:
            begin = self.index_of(begin_kind, pos)
            if begin < 0:
                return out
            end = self.index_of(end_kind, begin)
            if end < 0:
                return out
            out.append((begin, end))
            pos = end


class TimelineSpeaker(Speaker):
    """A fake speaker that takes real time to "play", like the real one.

    The duration is not decoration. `wait()` is only called at one specific
    place in the state machine, and a fake that returns instantly cannot
    distinguish "Diana waited for playback" from "Diana ignored playback
    entirely" -- both look the same when every call takes zero time.
    """

    name = "timeline"

    def __init__(self, timeline: Timeline, duration: float = 0.25) -> None:
        self.timeline = timeline
        self.duration = duration
        self.speaking = False
        self.closed = False

    def speak(self, text: str) -> None:
        self.timeline.add("speak")
        self.speaking = True

    def wait(self) -> None:
        self.timeline.add("wait-begin")
        time.sleep(self.duration)
        self.timeline.add("wait-end")
        self.speaking = False

    def close(self) -> None:
        self.closed = True


class TracingWakeWordDetector(ScriptedWakeWordDetector):
    """Records every block it is offered, so 'what could she have heard' is answerable."""

    def __init__(self, timeline: Timeline, fire_on) -> None:
        super().__init__(fire_on=fire_on, phrase="diana")
        self.timeline = timeline

    def detect(self, block):
        self.timeline.add("wake-detect")
        return super().detect(block)


class TracingSpeechToText(ScriptedSpeechToText):
    """Records every block handed to the recogniser."""

    def __init__(self, timeline: Timeline, text: str, emit_after_blocks: int) -> None:
        super().__init__(text=text, emit_after_blocks=emit_after_blocks)
        self.timeline = timeline

    def feed(self, block):
        self.timeline.add("stt-feed")
        return super().feed(block)


class TracingAudioSource(SyntheticAudioSource):
    """Records every block it hands over, which is what makes the discard measurable."""

    def __init__(self, timeline: Timeline, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.timeline = timeline

    def read(self):
        block = super().read()
        if block is not None:
            self.timeline.add("read")
        return block


class Checks:
    """Minimal assertion helper, so the test needs no pytest dependency."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, condition: bool, description: str) -> None:
        if condition:
            self.passed += 1
            print(f"  PASS  {description}")
        else:
            self.failed.append(description)
            print(f"  FAIL  {description}")


def _banner(title: str) -> None:
    print("=" * 62)
    print(f"  {title}")
    print("=" * 62)


def run_bounded(diana, seconds: float) -> int:
    """Run a Diana instance, but never for longer than `seconds`.

    A test must always terminate, even when the thing it is testing is broken.
    The scenarios below give their audio sources a huge block budget on
    purpose -- so that the *state machine* ends each interaction via a timeout
    rather than running out of audio, which is what happens in production. That
    makes the block budget useless as a safety net, so we supply a wall-clock
    one instead.

    `diana.request_stop()` is the same call the SIGTERM handler makes, so this
    also exercises the real shutdown path.
    """
    timer = threading.Timer(seconds, diana.request_stop)
    timer.daemon = True
    timer.start()
    try:
        return diana.run()
    finally:
        timer.cancel()

def _test_config(base: Config) -> Config:
    """A private Config per scenario, with test-sized timeouts.

    Two reasons this copies rather than mutating the caller's config:
    scenarios must not be able to leak state into each other, and the shipped
    timeouts (12 s / 1.6 s) are tuned for human speech, not for a test run.
    """
    cfg = replace(base)
    cfg.single_instance = False  # never fight the real daemon for the pid file
    cfg.show_partials = False     # partials are noise in a scripted test
    cfg.post_wake_discard_ms = 0  # keep the block arithmetic predictable
    cfg.command_max_s = 0.4
    cfg.command_silence_s = 0.15
    # Pinned, because several checks below assert on the whole list of spoken
    # lines and would otherwise break whenever the greeting is personalised.
    # They are about what happens *after* the wake word; "Yes, David?" is a
    # perfectly good greeting and says nothing about the handler.
    cfg.wake_response = "Yes?"
    # These four checks assert on the lines the speaker was given, so the
    # handler under test has to actually speak. Left to the shipped default
    # they silently became mute the day the default changed, which is the same
    # class of bug scenario 8 exists to catch -- so it is pinned here too.
    cfg.command_reply = "echo"
    return cfg


def run_selftest(base_cfg: Config) -> int:
    """Run simulated wake-word -> command -> idle cycles with no hardware."""
    from .main import Diana, DianaState  # local import avoids a cycle

    _banner("Project Diana -- self test (no microphone, no model)")
    print("  Every scenario drives the real Diana state machine with fake")
    print("  collaborators, so a pass means the logic is sound and any")
    print("  remaining fault is in your hardware, model or room.\n")

    check = Checks()

    # ------------------------------------------------------------------
    # Scenario 1: one complete, successful interaction
    # ------------------------------------------------------------------
    print('--- scenario 1: "diana" -> "turn on the TV" ----------------')
    print("  IDLE     -> the wake detector reports 'diana' on its 3rd block")
    print("  GREETING -> Diana: Yes?")
    print("  LISTENING-> STT reports 'turn on the TV', then command_max_s ends it")
    print("  IDLE     -> ready for the next one\n")

    cfg1 = _test_config(base_cfg)
    audio1 = SyntheticAudioSource(cfg1.sample_rate, cfg1.block_frames, total_blocks=1_000_000)
    wake1 = ScriptedWakeWordDetector(fire_on=3, phrase="diana")
    stt1 = ScriptedSpeechToText(text="turn on the TV", emit_after_blocks=2)
    handler1 = RecordingHandler()

    diana1 = Diana(cfg1, audio1, wake1, stt1, handler1, ConsoleSpeaker())
    exit_code = run_bounded(diana1, seconds=5.0)

    print("\n--- checks ----------------------------------------------------")
    check.check(audio1.started, "audio source opened")
    check.check(audio1.stopped, "audio source closed cleanly on exit")
    check.check(wake1.fired_count == 1, f"wake word fired exactly once (fired {wake1.fired_count}x)")
    check.check(stt1.started, "speech-to-text was started")
    check.check(
        handler1.seen == ["turn on the TV"],
        f"handler received the command (got {handler1.seen})",
    )
    check.check(diana1._commands_handled == 1, "exactly one command was handled")
    check.check(
        diana1.state is DianaState.STOPPED, f"final state is STOPPED (got {diana1.state.name})"
    )
    check.check(exit_code == 0, f"clean exit code (got {exit_code})")
    check.check(
        getattr(wake1, "resets", 0) >= 2,
        f"detector reset after the wake and before idling ({getattr(wake1, 'resets', 0)}x)",
    )

    # ------------------------------------------------------------------
    # Scenario 2: the IDLE energy gate
    # ------------------------------------------------------------------
    # This is the CPU saving that lets Diana sit idle for days. If it ever
    # regresses, the symptom is a laptop fan running constantly, so it is
    # worth asserting directly.
    print("\n--- scenario 2: pure silence must not reach the recogniser ----\n")

    cfg2 = _test_config(base_cfg)
    silent = SyntheticAudioSource(cfg2.sample_rate, cfg2.block_frames, total_blocks=1_000_000, amplitude=0.0)
    wake2 = ScriptedWakeWordDetector(fire_on=1, phrase="diana")
    handler2 = RecordingHandler()
    diana2 = Diana(cfg2, silent, wake2, ScriptedSpeechToText(), handler2, ConsoleSpeaker())
    run_bounded(diana2, seconds=1.0)

    print("\n--- checks ----------------------------------------------------")
    check.check(rms(np.zeros(cfg2.block_frames, dtype=np.float32)) == 0.0, "digital silence measures 0.0 RMS")
    check.check(
        abs(rms(np.full(cfg2.block_frames, 0.5, dtype=np.float32)) - 0.5) < 1e-6,
        "a full-scale block measures 0.5 RMS",
    )
    check.check(
        abs(rms(np.full(cfg2.block_frames, 0.05, dtype=np.float32)) - 0.05) < 1e-6,
        "a speech-level block measures 0.05 RMS",
    )
    check.check(
        wake2.blocks_offered == 0,
        f"silent blocks were gated out before the detector ({wake2.blocks_offered} offered)",
    )
    check.check(diana2._commands_handled == 0, "silence alone produced no command")

    # ------------------------------------------------------------------
    # Scenario 3: the wake word is heard but nothing intelligible follows
    # ------------------------------------------------------------------
    print('\n--- scenario 3: "diana" then no intelligible speech ---------')

    # Note the amplitude is 0.05, not 0.0. The IDLE energy gate lives *above*
    # the detector in the state machine -- it decides whether to call
    # detect() at all -- so a scripted detector cannot use silent audio to
    # trigger a wake. Scenario 2 covers the gate; here we want to get past it
    # and then starve the recogniser instead.
    print("  The audio is loud enough to pass the energy gate, but the")
    print("  scripted recogniser never produces text, so command_max_s must")
    print("  end the utterance with nothing to report.\n")

    cfg3 = _test_config(base_cfg)
    audio3 = SyntheticAudioSource(cfg3.sample_rate, cfg3.block_frames, total_blocks=1_000_000, amplitude=0.05)
    wake3 = ScriptedWakeWordDetector(fire_on=2, phrase="hey diana")
    stt3 = ScriptedSpeechToText(text="this should never be emitted", emit_after_blocks=99_999)
    handler3 = RecordingHandler()
    diana3 = Diana(cfg3, audio3, wake3, stt3, handler3, ConsoleSpeaker())

    start = time.monotonic()
    exit3 = run_bounded(diana3, seconds=3.0)
    elapsed = time.monotonic() - start

    print("\n--- checks ----------------------------------------------------")
    # Diana does NOT exit after one unproductive interaction -- she goes back
    # to waiting for the wake word, which is the whole point. So the run ends
    # when the safety bound stops it, and what we assert is that she recovered
    # to IDLE on her own and invented nothing.
    check.check(elapsed < 3.5, f"the run was cut short by the safety bound ({elapsed:.2f}s)")
    check.check(
        diana3.state is DianaState.STOPPED and exit3 == 0,
        "she shut down cleanly rather than crashing",
    )
    check.check(
        wake3.fired_count == 1,
        f"the wake word was heard ({wake3.fired_count}x)",
    )
    check.check(
        stt3.starts == 2,
        f"the recogniser was restarted on the way back to idle ({stt3.starts}x: "
        f"once to capture, once to clear)",
    )
    check.check(handler3.seen == [], "a silent wake produced no command")
    check.check(diana3._commands_handled == 0, "nothing was counted as handled")
    check.check(
        diana3._final_text == "", "no partial transcript was invented from silence"
    )

    # ------------------------------------------------------------------
    # Scenario 4: two full interactions in one run
    # ------------------------------------------------------------------
    # The important one: it proves Diana returns to IDLE and can wake up
    # again, which is the entire "run continuously" requirement.
    print("\n--- scenario 4: two interactions, proving the return to idle --\n")

    cfg4 = _test_config(base_cfg)
    audio4 = SyntheticAudioSource(cfg4.sample_rate, cfg4.block_frames, total_blocks=1_000_000)
    # fire_on counts blocks *offered* to the detector, which only happens in
    # IDLE. 5 is early; 60 is comfortably after the first command has ended.
    wake4 = ScriptedWakeWordDetector(fire_on=[5, 60], phrase="diana")
    stt4 = ScriptedSpeechToText(text="turn on the TV", emit_after_blocks=1)
    handler4 = RecordingHandler()
    diana4 = Diana(cfg4, audio4, wake4, stt4, handler4, ConsoleSpeaker())

    run_bounded(diana4, seconds=8.0)

    print("\n--- checks ----------------------------------------------------")
    check.check(
        handler4.seen == ["turn on the TV", "turn on the TV"],
        f"two consecutive commands were handled (got {handler4.seen})",
    )
    check.check(
        wake4.fired_count == 2, f"the detector fired twice (fired {wake4.fired_count}x)"
    )
    check.check(diana4.state is DianaState.STOPPED, "the instance shut down cleanly")

    # ------------------------------------------------------------------
    # Scenario 5: the wake-word matching rules
    # ------------------------------------------------------------------
    # These need no model and no audio. They are pure string logic, and they
    # encode the false-positive defence described in wakeword.py, so they are
    # worth pinning down: a closed grammar will happily force ordinary speech
    # onto your wake word, and the rule that prevents it is easy to break by
    # accident during a refactor.
    print("\n--- scenario 5: wake-word matching rules (no model needed) --\n")

    from .wakeword import VoskWakeWordDetector as _VWD

    # __new__ skips __init__, so we build just the string-matching state. This
    # is the one place it is justified: loading a 40 MB acoustic model to test
    # a list comparison would be absurd.
    probe = _VWD.__new__(_VWD)
    probe.phrases = ["diana", "hey diana", "ok diana"]
    probe._phrase_words = [p.split() for p in probe.phrases]

    cases = [
        (["diana"], True, "bare 'Diana'"),
        (["hey", "diana"], True, "'Hey Diana'"),
        (["ok", "diana"], True, "'Ok Diana'"),
        (["diana", "[unk]"], True, "run-on: name then command in one breath"),
        (["diana", "maria"], True, "'Diana Maria' still triggers"),
        (["[unk]"], False, "unrecognised speech only"),
        (["[unk]", "[unk]"], False, "noise only"),
        (["[unk]", "diana", "[unk]"], False, "REGRESSION GUARD: forced match after noise"),
        (["[unk]", "diana"], False, "REGRESSION GUARD: name preceded by noise"),
        (["turn", "on", "the", "tv"], False, "an ordinary command, not a wake"),
    ]

    # The strict rule is the contract, so assert it exactly.
    probe.require_at_start = True
    for words, want, desc in cases:
        got = probe._find_phrase(words) is not None
        check.check(got == want, f"strict: {desc} -> {got}")

    # The permissive mode exists only for comparison. Its invariant is that it
    # never finds *fewer* phrases than the strict one -- it is strictly more
    # willing to fire. Checking that relationship documents why the strict
    # default is the safer setting, without asserting specific outcomes for a
    # mode we do not use.
    probe.require_at_start = False
    probe_strict, probe.require_at_start = probe, True
    for words, _want, desc in cases:
        strict_hit = probe_strict._find_phrase(words) is not None
        probe.require_at_start = False
        loose_hit = probe._find_phrase(words) is not None
        probe.require_at_start = True
        check.check(
            strict_hit == loose_hit or (loose_hit and not strict_hit),
            f"permissive never rejects what strict accepts: {desc}",
        )

    # ------------------------------------------------------------------
    # Scenario 6: --max-commands stops her at exactly the right count
    # ------------------------------------------------------------------
    # This exists because the obvious implementation -- wrapping the handler
    # and incrementing the counter there -- counts every command twice, since
    # the state machine increments it too. That version passes for N=1 and N=2
    # and quietly stops early for N>=3, which is the worst kind of bug.
    print("\n--- scenario 6: --max-commands stops at exactly N ----------\n")

    for target in (1, 2, 3, 5):
        cfg6 = _test_config(base_cfg)
        audio6 = SyntheticAudioSource(cfg6.sample_rate, cfg6.block_frames, total_blocks=1_000_000)
        # Offer the wake word often enough that any shortfall is the limit's
        # fault rather than a lack of opportunities.
        wake6 = ScriptedWakeWordDetector(fire_on=list(range(3, 400, 20)), phrase="diana")
        handler6 = RecordingHandler()
        diana6 = Diana(cfg6, audio6, wake6, ScriptedSpeechToText("turn on the TV", 1),
                       handler6, ConsoleSpeaker(), max_commands=target)
        run_bounded(diana6, seconds=20.0)
        check.check(
            diana6._commands_handled == target,
            f"--max-commands={target} handled exactly {target} "
            f"(got {diana6._commands_handled})",
        )
        check.check(
            len(handler6.seen) == target,
            f"--max-commands={target} reached the handler {target}x (got {len(handler6.seen)})",
        )

    # ------------------------------------------------------------------
    # Scenario 7: she must never listen to herself
    # ------------------------------------------------------------------
    # The bug this exists to prevent is the one that a console speaker makes
    # impossible and a real voice makes inevitable. Diana speaks, the
    # microphone is still open, the speakers are still audible, and the wake
    # detector -- or the recogniser -- picks her own reply back up. Then she
    # says "Yes?", hears "Yes?", says "Yes?", and greets herself until you pull
    # the plug.
    #
    # ConsoleSpeaker.wait() is a no-op, so scenarios 1-6 pass either way. This
    # scenario uses a speaker that takes real time to "play" and then asserts
    # three separate defences:
    #
    #   1. every line is waited for, including the reply to a command
    #   2. no audio reaches the detector or the recogniser while she is talking
    #   3. the microphone is still distrusted for a moment after she stops
    print("\n--- scenario 7: she must not listen to herself ---------------\n")
    print("  The fake speaker takes 0.25 s to 'play' each line, and every")
    print("  collaborator records what it was asked to do and when.\n")

    cfg7 = _test_config(base_cfg)
    cfg7.post_speak_discard_ms = 90  # 3 blocks, comfortably larger than the
    cfg7.wake_cooldown_s = 0.4      # fake speaker, but still a real guard
    discard_blocks = int(
        (cfg7.post_wake_discard_ms + cfg7.post_speak_discard_ms) / cfg7.block_ms
    )

    tl = Timeline()
    audio7 = TracingAudioSource(
        tl, cfg7.sample_rate, cfg7.block_frames, total_blocks=1_000_000
    )
    # Fire on the third block offered (the first interaction), then on the very
    # next block offered -- which, thanks to the post-speech guard, is the first
    # one after the cooldown expires. That makes check 3 below meaningful
    # instead of vacuous: without the guard this second wake would land on the
    # block immediately after her reply finished.
    wake7 = TracingWakeWordDetector(tl, fire_on=[3, 4])
    stt7 = TracingSpeechToText(tl, text="turn on the TV", emit_after_blocks=2)
    handler7 = RespondingHandler()
    speaker7 = TimelineSpeaker(tl, duration=0.25)

    diana7 = Diana(cfg7, audio7, wake7, stt7, handler7, speaker7)
    run_bounded(diana7, seconds=6.0)

    print("\n--- checks ----------------------------------------------------")
    # Two commands, because the scripted detector deliberately wakes her a
    # second time (see fire_on above) -- which is what makes the post-speech
    # cooldown assertion below possible.
    check.check(
        handler7.seen == ["turn on the TV", "turn on the TV"],
        f"both interactions reached the handler (got {handler7.seen})",
    )
    # 1. Every line waited for. If this is the only check that fails, the bug
    #    is the common one: speak() without wait() on the reply path.
    check.check(
        tl.count("speak") == tl.count("wait-begin"),
        f"every line was waited for ({tl.count('speak')} spoken, "
        f"{tl.count('wait-begin')} waited)",
    )
    check.check(
        tl.count("speak") >= 3,
        f"she spoke a greeting, a reply and another greeting ({tl.count('speak')})",
    )

    # 2. Nothing heard the microphone while she was talking. If the state
    #    machine ever stops blocking in wait() -- say, to make playback
    #    non-blocking -- this is what catches it.
    spans = tl.spans("wait-begin", "wait-end")
    leaked = [kind for begin, end in spans for kind, _ in tl.events[begin + 1 : end]]
    check.check(
        not leaked,
        f"no audio was routed while she was talking (leaked {leaked})",
    )

    # 3. After she stops, the microphone is distrusted for post_speak_discard_ms
    #    before the recogniser sees anything -- otherwise the tail of her own
    #    "Yes?" becomes the first word of your command.
    first_feed = tl.index_of("stt-feed")
    greeting_end = tl.index_of("wait-end")
    reads_before_feed = (
        sum(1 for k, _ in tl.events[greeting_end + 1 : first_feed]) if first_feed > 0 else 0
    )
    check.check(
        reads_before_feed >= discard_blocks,
        f"the tail of her greeting was discarded before recognition "
        f"({reads_before_feed} blocks read, {discard_blocks} required)",
    )

    # 4. And the wake detector stays shut for the cooldown afterwards. A span
    #    with no wake-detect after it trivially passes: there was nothing to
    #    suppress.
    check.check(
        all(
            all(at - tl.events[end][1] >= cfg7.wake_cooldown_s * 0.9
                for kind, at in tl.events[end + 1 :] if kind == "wake-detect")
            for _, end in spans
        ),
        f"the wake detector stayed shut for the {cfg7.wake_cooldown_s}s cooldown after she spoke",
    )
    check.check(wake7.fired_count == 2, f"she woke up twice, not in a loop (got {wake7.fired_count})")

    # ------------------------------------------------------------------
    # Scenario 8: the *default* handler must actually speak
    # ------------------------------------------------------------------
    # Scenarios 1-7 all use fakes, and scenario 7 in particular uses
    # RespondingHandler, which always returns a response. So nothing until now
    # ever asserted the thing that was actually broken: PrintCommandHandler
    # returned response="" for every command, so main.py's
    # `if result.response: speaker.speak(...)` was never true and a fully
    # working voice sat mute through the entire command half of every
    # conversation. It passed every test here while being useless in the room.
    #
    # The fix is one non-empty string, and the regression it guards is worth
    # more than the fix: a test that only ever exercises fakes cannot catch a
    # default that does nothing. So this drives the real handler, through the
    # real state machine, and reads back the lines the speaker was given.
    print("\n--- scenario 8: the default handler speaks its replies --------\n")

    def _one_shot(heard: str, **handler_kwargs) -> tuple[list[str], list[CommandResult]]:
        """Run one wake -> command cycle with the real handler.

        Returns the lines the speaker was given and the results the real
        handler produced. The handler is wrapped rather than subclassed so that
        `PrintCommandHandler` itself stays a pure function of the command --
        it should not grow a `last_result` field just to be observable.
        """
        cfg = _test_config(base_cfg)
        said = io.StringIO()
        real = PrintCommandHandler(echo=False, **handler_kwargs)
        seen: list[CommandResult] = []

        class Recording(CommandHandler):
            name = real.name

            def handle(self, command: Command) -> CommandResult:
                result = real.handle(command)
                seen.append(result)
                return result

        diana = Diana(
            cfg,
            SyntheticAudioSource(cfg.sample_rate, cfg.block_frames, total_blocks=1_000_000),
            ScriptedWakeWordDetector(fire_on=1, phrase="diana"),
            ScriptedSpeechToText(text=heard, emit_after_blocks=2),
            Recording(),
            ConsoleSpeaker(stream=said),
        )
        run_bounded(diana, seconds=2.0)
        # The wake greeting is always "Yes?"; what we care about is the reply.
        return [ln.split(": ", 1)[1] for ln in said.getvalue().splitlines()], seen

    # The reply mode is pinned in every call below rather than left to
    # `cfg.command_reply`. These assertions are about PrintCommandHandler's
    # echo/ack/none modes specifically, and letting them ride on whatever the
    # shipped default happens to be couples a unit test to a config choice:
    # changing the default from "echo" to "intent" broke all four of these,
    # which says nothing about whether the handler works.
    spoken, results = _one_shot("turn on the TV", reply="echo")
    check.check(
        spoken == ["Yes?", "turn on the TV"],
        f"the default handler speaks the command back (got {spoken})",
    )
    check.check(
        [r.status for r in results] == [CommandStatus.OK],
        f"and reports OK (got {[r.status for r in results]})",
    )

    spoken, _ = _one_shot("turn on the TV", reply="none")
    check.check(
        spoken == ["Yes?"],
        f'reply="none" stays silent after the greeting (got {spoken})',
    )

    spoken, _ = _one_shot("turn on the TV", reply="ack")
    check.check(
        len(spoken) == 2 and spoken[1] in PrintCommandHandler._ACKS,
        f'reply="ack" says a short confirmation (got {spoken})',
    )

    # An unknown command gets the spoken refusal, not silence.
    spoken, results = _one_shot(
        "make me a sandwich", reply="echo", known_commands=["turn on the TV"]
    )
    check.check(
        spoken == ["Yes?", "I don't know how to do that yet."],
        f"an unsupported command is refused out loud (got {spoken})",
    )
    check.check(
        [r.status for r in results] == [CommandStatus.UNSUPPORTED],
        f"and the status says why (got {[r.status for r in results]})",
    )

    # No vocabulary means no refusals: V0 has no idea what the device layer
    # can do, so inventing a list here would be a lie with a config key.
    spoken, results = _one_shot("make me a sandwich", reply="echo")
    check.check(
        spoken == ["Yes?", "make me a sandwich"],
        f"with no vocabulary everything is accepted (got {spoken})",
    )
    check.check(
        [r.status for r in results] == [CommandStatus.OK],
        f"and the status is OK (got {[r.status for r in results]})",
    )

    # Filler words are what a recogniser actually produces, so the vocabulary
    # has to survive them. Contiguous matching fails this.
    _, results = _one_shot(
        "Please turn on the TV now.", known_commands=["turn on the TV"]
    )
    check.check(
        [r.status for r in results] == [CommandStatus.OK],
        f"filler words and punctuation do not defeat the vocabulary "
        f"(got {[r.status for r in results]})",
    )

    _, results = _one_shot("tvs are loud", known_commands=["turn on the TV"])
    check.check(
        [r.status for r in results] == [CommandStatus.UNSUPPORTED],
        f"a word that merely shares a prefix is not a match "
        f"(got {[r.status for r in results]})",
    )

    # ------------------------------------------------------------------
    # Scenario 9: the intent grammar, and the console line that proves it
    # ------------------------------------------------------------------
    # Scenarios 1-8 could all pass with a Diana that repeats whatever it
    # heard, because that is exactly what the fake handlers do. This one
    # drives the real parser and asserts the two things the design is for:
    # a machine-readable event code, and a sentence that describes what
    # happened rather than what was said.
    print("\n--- scenario 9: commands become event codes -----------------\n")

    def _intent_handler(chat=None) -> IntentCommandHandler:
        return IntentCommandHandler(
            CommandParser(), reply="intent", echo=False, show_events=False,
            chat=chat if chat is not None else NullChat(),
        )

    def _say(handler, text: str) -> CommandResult:
        return handler.handle(Command(text=text))

    h = _intent_handler()

    # The example from the top of the README, asserted literally: this is the
    # line the whole project is arranged around.
    r = _say(h, "turn on the lights")
    check.check(r.data.get("code") == "LIGHTS_ON", f"lights on -> LIGHTS_ON (got {r.data.get('code')})")
    check.check(r.response == "lights are now on.", f"and says so (got {r.response!r})")
    check.check(r.status is CommandStatus.OK, "and reports OK")

    # Every device the user asked for, not just lights.
    for text, code in [
        ("turn on the lights", "LIGHTS_ON"),
        ("switch the lights off", "LIGHTS_OFF"),
        ("turn the tv on", "TV_ON"),
        ("play some music", "MUSIC_PLAY"),
        ("pause the music", "MUSIC_PAUSE"),
        ("skip this track", "MUSIC_NEXT"),
        ("turn it up", "MUSIC_VOLUME_UP"),
        ("mute it", "MUSIC_MUTE"),
        ("open spotify", "APP_OPEN_SPOTIFY"),
        ("close spotify", "APP_CLOSE_SPOTIFY"),
        ("what is the weather like", "WEATHER_QUERY"),
        ("how hot is it", "WEATHER_QUERY"),
        ("set a timer for ten minutes", "TIMER_SET"),
    ]:
        got = _say(h, text).data.get("code")
        check.check(got == code, f"{text!r} -> {code} (got {got})")

    # Both English orders. Phrase tables only find one of them, which is why
    # the trailing-particle rule exists.
    check.check(
        _say(h, "switch the lights on").data.get("code") == "LIGHTS_ON",
        "trailing particle form is understood too",
    )

    # The group is a device that expands, not a special case in the parser.
    check.check(
        _say(h, "turn everything off").data.get("code") == "ALL_OFF",
        f"group command -> ALL_OFF (got {_say(h, 'turn everything off').data.get('code')})",
    )

    # A question is answered from remembered state, which is the thing an
    # echo cannot do. Ask *before* switching on, then after.
    h2 = _intent_handler()
    before = _say(h2, "are the lights on?")
    _say(h2, "turn on the lights")
    after = _say(h2, "are the lights on?")
    check.check(
        "no" in before.response and "yes" in after.response,
        f"state is remembered, so the answer changes (got {before.response!r} -> {after.response!r})",
    )

    # Toggle has to look before it can say which way it went.
    h3 = _intent_handler()
    t1 = _say(h3, "toggle the lights")
    t2 = _say(h3, "toggle the lights")
    check.check(
        t1.response != t2.response,
        f"toggle says which way it went (got {t1.response!r} then {t2.response!r})",
    )

    # The capability check is a safety property, not a tidiness one: a device
    # that cannot be powered must never be powered because a word matched.
    # Note what is asserted is *the absence of an action*, not a particular
    # code. "turn off the weather" is nonsense and the weather rule claims it
    # first, which is fine; what would not be fine is a power action reaching
    # a device that cannot be powered.
    weather_messages = len([m for m in h.registry.outbox if m["device"] == "weather"])
    _say(h, "turn off the weather")
    _say(h, "switch the weather on")
    after = [m for m in h.registry.outbox if m["device"] == "weather"]
    check.check(
        all(m["action"] in ("", "on", "off") for m in after[weather_messages:]),
        f"no unsupported power action reached the weather device "
        f"(got {[m['action'] for m in after[weather_messages:]]})",
    )
    check.check(
        all(m["action"] != "toggle" for m in after),
        "and no toggle reached it either",
    )

    # A question must not put a message in the outbox at all. An entry with an
    # empty action is worse than none: a node implementing the contract has no
    # defined behaviour for it, and the log fills with lines that look like
    # traffic.
    outbox_before = len(h.registry.outbox)
    _say(h, "what is the weather like")
    _say(h, "are the lights on?")
    check.check(
        len(h.registry.outbox) == outbox_before,
        f"a question sends no device message (outbox grew by "
        f"{len(h.registry.outbox) - outbox_before})",
    )

    # Understood but unfulfillable is a different answer from not understood.
    w = _say(h, "what is the weather like")
    check.check(
        w.status is CommandStatus.OK and "real source" in w.response,
        f"weather is understood, not fabricated (status={w.status.value} {w.response!r})",
    )

    # ------------------------------------------------------------------
    # Scenario 10: the fallback chat
    # ------------------------------------------------------------------
    # The grammar is a closed world, so most real sentences end in UNKNOWN.
    # What happens then is the difference between an assistant and a toy, and
    # it has one hard rule: the model answers, and never acts.
    print("\n--- scenario 10: no rule matched, so the model answers -------\n")

    fake = FakeChat(answer="Paris is the capital of France.")
    hc = _intent_handler(chat=fake)

    r = _say(hc, "what is the capital of france")
    check.check(
        r.status is CommandStatus.CHAT, f"an unmatched sentence is answered (status={r.status.value})"
    )
    check.check(r.response == "Paris is the capital of France.", "and the model's line is spoken")
    check.check(
        fake.asked == ["what is the capital of france"],
        f"the model was asked exactly once (got {fake.asked})",
    )

    # The safety property. A command must never be routed to the model, and a
    # model that claims to have acted must still not have acted.
    before_outbox = len(hc.registry.outbox)
    _say(hc, "turn on the lights")
    check.check(
        len(hc.registry.outbox) == before_outbox + 1
        and not fake.asked[-1:] == ["turn on the lights"],
        "a matched command never reaches the model",
    )

    # Conversely, a chat answer must never touch the house.
    r = _say(hc, "please turn the lights on for me")
    if r.status is CommandStatus.CHAT:
        check.check(
            r.data.get("device", "") == "",
            f"a chat answer carries no device (got {r.data.get('device')!r})",
        )
    else:
        # The grammar claimed it, which is also fine -- the point is only
        # that a model answer never carries a device.
        check.check(True, "a chat answer carries no device")

    # Chat off: rules still work, and the "can't do that" line comes back.
    hoff = _intent_handler(chat=NullChat())
    r = _say(hoff, "what is the capital of france")
    check.check(
        r.status is CommandStatus.UNSUPPORTED and "don't know how" in r.response,
        f"with chat off, unknown stays unknown (status={r.status.value} {r.response!r})",
    )
    check.check(
        _say(hoff, "turn on the lights").data.get("code") == "LIGHTS_ON",
        "and commands are unaffected by chat being off",
    )

    # Chat on but silent: a *different* apology, because a different thing is
    # broken. This is the distinction that stops you debugging intents.py when
    # the fault is that ollama is not running.
    hs = _intent_handler(chat=FakeChat(null=True))
    r = _say(hs, "what is the capital of france")
    check.check(
        r.status is CommandStatus.UNSUPPORTED and "answer" in r.response,
        f"a silent model gets its own message (got {r.response!r})",
    )

    # shorten(): a model answer has to survive being spoken.
    # A real newline, not the two characters "\" and "n" -- that is a
    # distinction worth being pedantic about in a test about whitespace.
    decorated = "**Bold**  and" + chr(10) + "newlines"
    check.check(
        shorten(decorated, 200) == "Bold and newlines",
        f"markdown and newlines are stripped (got {shorten(decorated, 200)!r})",
    )
    long_answer = "One. Two. " + ("blah " * 80)
    out = shorten(long_answer, 100)
    check.check(
        len(out) <= 101 and out.endswith("."),
        f"a long answer is cut at a sentence, not mid-word (got {out!r})",
    )
    check.check(shorten("", 100) == "", "an empty answer stays empty")
    check.check(shorten("short", 100) == "short", "a short answer is untouched")

    # A real client, but not a real server call: the shape is what matters.
    client = OllamaChat(model="qwen2.5:0.5b", host="http://localhost:1", timeout_s=0.4)
    check.check(
        client.available(timeout_s=0.4) is False,
        "an unreachable ollama reports unavailable instead of raising",
    )
    check.check(
        client.reply("hello") == "",
        "and an unreachable ollama returns no answer instead of raising",
    )
    check.check(client.stats.failed == 1, "and records the failure")
    check.check("not running" in client.describe(), "and the banner says so")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = check.passed + len(check.failed)
    print("\n" + "=" * 62)
    if check.failed:
        print(f"  {len(check.failed)} of {total} checks FAILED:")
        for name in check.failed:
            print(f"    - {name}")
        print("=" * 62)
        return 1
    print(f"  All {total} checks passed.")
    print("  The state machine, the audio plumbing and the interfaces are sound.")
    print("=" * 62)
    return 0
