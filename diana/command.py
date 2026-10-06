"""
command.py -- what happens to the text after it is recognised.

Two handlers live here, and the difference between them is the whole point of
the module boundary.

`PrintCommandHandler` repeats what it heard. It can prove the microphone, the
wake word, the state machine and the recogniser all worked, and it can do
nothing else. It stays in the codebase because it is the fastest way to answer
"is she hearing me correctly?", and that question has to stay cheap to ask.

`IntentCommandHandler` is the one Diana actually runs. It parses the transcript
into a structured action, prints the event code, applies it to the device
registry and says what happened. It is a rule table, not a model -- but it
returns the same `CommandResult` shape a Transformer will, so replacing it is a
one-line change in main.py:

    # today
    handler = IntentCommandHandler(parser, registry, ...)
    # later
    handler = MyTransformerHandler(checkpoint="models/parser.pt", device="cuda")

Nothing in main.py, audio.py, stt.py or wakeword.py changes either way. That is
the entire reason the project is split into modules.
"""

from __future__ import annotations

import itertools
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .devices import DeviceRegistry, default_registry
from .intents import (
    QUERY_INTENTS,
    CommandParser,
    Match,
    message_for,
    speak_for,
)
from .chat import NullChat
from .stt import Transcription, Word

log = logging.getLogger(__name__)


class CommandStatus(Enum):
    """Outcome of handling a command.

    Having an explicit status, rather than just returning a string, is what
    lets the main loop decide whether to go back to idle or to apologise. A
    neural parser will want these same distinctions (e.g. UNSUPPORTED for
    "make me a sandwich").
    """

    OK = "ok"
    EMPTY = "empty"
    """Nothing was recognised, or only noise."""

    UNSUPPORTED = "unsupported"
    """Understood, but no handler for it."""

    CHAT = "chat"
    """No rule matched and the small local model answered instead.

    Separate from UNSUPPORTED because the two mean opposite things to the
    person talking. UNSUPPORTED is Diana saying no; CHAT is Diana having
    answered a question that was never a command. Collapsing them would make
    "how do I make lasagne" and "turn off the weather" indistinguishable in
    the log, which is the one thing a log is for.
    """

    ERROR = "error"
    """The handler itself failed. Should be logged loudly."""


@dataclass(frozen=True)
class Command:
    """One recognised request, ready to be acted on."""

    text: str
    at: float = field(default_factory=time.time)
    confidence: float = 0.0
    words: list[Word] = field(default_factory=list)
    utterance_s: float = 0.0
    """Wall-clock length of the utterance, from end-of-wake to end-of-speech."""


@dataclass(frozen=True)
class CommandResult:
    """What the handler decided."""

    status: CommandStatus
    response: str = ""
    """Text for Diana to say back. Empty means say nothing."""

    data: dict = field(default_factory=dict)
    """Structured payload for programmatic use.

    A print handler fills this with {'text': ...}. A real parser would fill it
    with {'intent': 'device.control', 'device': 'tv', 'action': 'on'} and the
    device layer would consume that dict instead of parsing English.
    """


class CommandHandler(ABC):
    """Interface for anything that can act on a recognised command."""

    name: str = "abstract"

    @abstractmethod
    def handle(self, command: Command) -> CommandResult:
        """Interpret one command. Must not block for long.

        Keep this fast. It runs on the audio thread's timeline -- if it takes
        two seconds, the user's next utterance is already gone. For a neural
        model that means loading weights at start-up, not inside handle().
        """

    def close(self) -> None:
        """Release resources (model weights, GPU memory)."""


class PrintCommandHandler(CommandHandler):
    """V0 handler: print the command, confirm it out loud, and do nothing else.

    This is the only "intelligence" in V0, on purpose. It exists so the
    pipeline can be verified end to end: if you see
    `COMMAND: turn on the TV` in the terminal, then the microphone, the wake
    word, the state machine and the recogniser all worked.

    It also speaks, because a loop you cannot hear is only half a loop. See
    `reply` below for why the default repeats the command rather than
    confirming it -- that choice is a debugging decision, not a personality.
    """

    name = "print"

    _ACKS = ("Okay.", "Got it.", "Sure.")

    def __init__(
        self,
        echo: bool = True,
        reply: str = "echo",
        known_commands: Iterable[str] | None = None,
        unsupported_response: str = "I don't know how to do that yet.",
    ) -> None:
        self.echo = echo
        self.reply = reply
        self.unsupported_response = unsupported_response
        # Rotates the ack so that repeating one command does not produce the
        # same reply every time, which is what makes a loop sound broken. A
        # hash of the text will not do: the same text must vary, not just
        # different texts.
        self._ack_at = itertools.count()

        # An empty vocabulary means "everything is supported", which is the
        # honest V0 default: this class has no idea what Diana can actually do,
        # and inventing a list here would be fake intelligence wearing the
        # costume of a feature. Set it once you have a handler that can really
        # reject things, and the UNSUPPORTED branch below starts firing.
        self.known_commands = [self._tokens(c) for c in (known_commands or ())]
        self.known_commands = [c for c in self.known_commands if c]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        """Lowercase word list, punctuation discarded.

        Vosk is inconsistent about case and the recogniser routinely drops or
        invents apostrophes, so "Turn on the TV's" and "turn on the tv" have to
        compare equal or the vocabulary misses half of what was actually said.
        """
        return re.findall(r"[a-z0-9]+", text.lower())

    def _is_supported(self, text: str) -> bool:
        """True unless a vocabulary is set and nothing in it matches.

        Matching is ordered token containment with gaps allowed, so
        "please turn on the TV now" matches a known "turn on tv". Gaps are
        mandatory rather than merely convenient here: filler words and
        articles are exactly what a recogniser inserts and drops, so requiring
        contiguity would miss most of what was genuinely said. Tokens must
        still appear *in order*, which is what stops "tvs are loud" from
        matching a known "tv".
        """
        if not self.known_commands:
            return True
        tokens = self._tokens(text)
        return any(self._contains(tokens, known) for known in self.known_commands)

    @staticmethod
    def _contains(haystack: list[str], needle: list[str]) -> bool:
        it = iter(haystack)
        return all(token in it for token in needle)

    def handle(self, command: Command) -> CommandResult:
        text = command.text.strip()
        if not text:
            log.info("empty command ignored")
            return CommandResult(status=CommandStatus.EMPTY, response="I didn't catch that.")

        if self.echo:
            # The exact line from the V0 spec. Kept verbatim so you can grep
            # your terminal history for real commands later.
            print(f"COMMAND: {text}", flush=True)

        log.info(
            "command handled: %r (confidence %.2f, utterance %.1fs)",
            text,
            command.confidence,
            command.utterance_s,
        )

        if not self._is_supported(text):
            log.info("no handler for %r", text)
            return CommandResult(
                status=CommandStatus.UNSUPPORTED,
                response=self.unsupported_response,
                data={"text": text, "confidence": command.confidence},
            )

        return CommandResult(
            status=CommandStatus.OK,
            response=self._reply_for(text),
            data={
                "text": text,
                "confidence": command.confidence,
                "word_count": len(command.words),
            },
        )

    def _reply_for(self, text: str) -> str:
        """What to say back, per the `reply` mode.

        "echo" is the default and it is a debugging tool, not a personality.
        It repeats the recogniser's output verbatim so a mishearing is
        unmistakable: if she says "Lice." instead of "Lights.", you have lost
        the bug. Any confirmation phrasing -- "Okay.", "Sure thing." -- is
        *worse* here precisely because it is fluent, because a confident reply
        in the wrong words hides the error it was supposed to surface. Flip to
        "ack" once recognition is good enough that you no longer need to check.
        """
        if self.reply == "echo":
            return text
        if self.reply == "ack":
            return self._ACKS[next(self._ack_at) % len(self._ACKS)]
        return ""


class IntentCommandHandler(CommandHandler):
    """Turns a command into an event code, an action, and a sentence.

        You: turn on the lights
        EVENT: LIGHTS_ON
        Diana: lights are now on.

    The `EVENT:` line is the honest half of the output. It is machine-readable,
    greppable, and it is what a node would act on; the sentence is for you. When
    the two disagree -- when she says "lights are now on" for something that was
    not about lights -- the event line is where the bug shows up first, because
    it cannot be a fluent mishearing.

    Three things happen here, in this order, and the order matters:

    1. `parse()` decides what was asked for. No state, no I/O.
    2. `registry.apply()` is the *only* thing that changes what the house is
       believed to look like. The parser never writes state and never guesses.
    3. `speak_for()` writes the sentence, using the state that resulted. That
       ordering is what lets "toggle the lights" say which way it went, and
       "are the lights on?" give an answer instead of an echo.

    `reply` still works, because debugging a voice assistant needs a mode where
    she says exactly what she heard. See PrintCommandHandler._reply_for.
    """

    name = "intents"

    def __init__(
        self,
        parser: CommandParser | None = None,
        registry: DeviceRegistry | None = None,
        reply: str = "intent",
        echo: bool = True,
        show_events: bool = True,
        known_commands: Iterable[str] | None = None,
        unsupported_response: str = "I don't know how to do that yet.",
        placeholder_response: str = "I don't have a real source for that yet.",
        chat=None,
        chat_unavailable_response: str = "I didn't catch an answer to that.",
    ) -> None:
        self.registry = registry if registry is not None else default_registry()
        self.parser = parser if parser is not None else CommandParser(self.registry)
        self.reply = reply
        self.echo = echo
        self.show_events = show_events
        self.unsupported_response = unsupported_response
        self.placeholder_response = placeholder_response
        self.chat_unavailable_response = chat_unavailable_response
        # NullChat rather than None, so there is exactly one code path for
        # "ask the model" and the disabled case cannot drift from the enabled
        # one. See chat.NullChat.
        self.chat = chat if chat is not None else NullChat()
        self._ack_at = itertools.count()

        # The same vocabulary check the print handler had, kept because it is
        # still the right way to say "in this deployment, that is not allowed",
        # independently of whether the parser understood the sentence.
        self.known_commands = [self._tokens(c) for c in (known_commands or ())]
        self.known_commands = [c for c in self.known_commands if c]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        """Lowercase word list, punctuation discarded.

        Vosk is inconsistent about case and the recogniser routinely drops or
        invents apostrophes, so "Turn on the TV's" and "turn on the tv" have to
        compare equal or the vocabulary misses half of what was actually said.
        """
        return re.findall(r"[a-z0-9]+", text.lower())

    def _is_supported(self, text: str) -> bool:
        """True unless a vocabulary is set and nothing in it matches.

        Matching is ordered token containment with gaps allowed, so
        "please turn on the TV now" matches a known "turn on tv". Gaps are
        mandatory rather than merely convenient here: filler words and articles
        are exactly what a recogniser inserts and drops, so requiring
        contiguity would miss most of what was genuinely said. Tokens must
        still appear *in order*, which is what stops "tvs are loud" from
        matching a known "tv".
        """
        if not self.known_commands:
            return True
        tokens = self._tokens(text)
        return any(self._contains(tokens, known) for known in self.known_commands)

    @staticmethod
    def _contains(haystack: list[str], needle: list[str]) -> bool:
        it = iter(haystack)
        return all(token in it for token in needle)

    def handle(self, command: Command) -> CommandResult:
        text = command.text.strip()
        if not text:
            log.info("empty command ignored")
            return CommandResult(status=CommandStatus.EMPTY, response="I didn't catch that.")

        match = self.parser.parse(text)

        if self.echo:
            # The exact line from the V0 spec. Kept verbatim so you can grep
            # your terminal history for real commands later.
            print(f"COMMAND: {text}", flush=True)

        if not self._is_supported(text):
            self._event(match)
            log.info("no handler for %r", text)
            return CommandResult(
                status=CommandStatus.UNSUPPORTED,
                response=self.unsupported_response,
                data={"text": text, "confidence": command.confidence},
            )

        if not match.known:
            # Deliberately not printing the event line here. An unknown match
            # whose code is UNKNOWN prints its own line -- CHAT or nothing --
            # and two event lines for one utterance ("EVENT: UNKNOWN" followed
            # by "EVENT: CHAT") is a worse console than one honest line.
            return self._handle_unknown(command, text, match)

        self._event(match)

        # Carry the action out, then read the state it produced. A question is
        # not a command, so it goes no further than the state table.
        if match.intent in QUERY_INTENTS:
            device = self.registry.get(match.target)
            state = dict(device.state) if device else {}
        else:
            targets = self.registry.apply(match.target, match.action)
            state = dict(targets[0].state) if targets else {}
            if not targets:
                # The registry refused. Something upstream believed a device
                # could do this and the registry knows better; say so instead of
                # claiming success.
                log.info("action refused by the registry: %s %s", match.code, match.action)
                return CommandResult(
                    status=CommandStatus.ERROR,
                    response="I couldn't do that.",
                    data=message_for(match, text, command.confidence),
                )

        response = self._reply_for(match, state, text)
        return CommandResult(
            status=CommandStatus.OK,
            response=response,
            data=message_for(match, text, command.confidence),
        )

    def _event(self, match: Match) -> None:
        """The one console line that says what Diana decided.

        This is the line the whole design hangs on. It is uppercase, it is
        greppable, and it cannot be a fluent mishearing: if it says LIGHTS_ON
        then the parser matched the lights, whatever the transcript looked
        like. The spoken sentence below it is the one that can be wrong in a
        way you cannot see.
        """
        if self.show_events:
            print(f"EVENT: {match.code}", flush=True)

    def _handle_unknown(
        self, command: Command, text: str, match: Match
    ) -> CommandResult:
        """No rule fired. Ask the small local model before giving up.

        This is the only place in the project the model is consulted, and it is
        reached only after the grammar has already failed. That ordering is the
        whole safety argument: the model is asked to *answer*, never to *act*,
        and the only way it can influence the house is by talking. It has no
        registry, no event codes and no transport, so a model that decides to
        ignore its instructions and claim it turned the lights on produces a
        confident lie in the audio and changes nothing at all.
        """
        answer = self.chat.reply(text)
        if not answer:
            # Two different failures that want two different fixes, so they get
            # two different sentences. Chat switched off is not an error at
            # all -- "I don't know how to do that yet" is then simply the truth
            # about a rule table. Chat enabled but silent means ollama is down
            # or the model is not pulled, and saying "I don't know how to do
            # that" would send you looking at intents.py, which is not where the
            # fault is.
            if getattr(self.chat, "disabled", False):
                log.info("no rule matched for %r (chat disabled)", text)
                response = self.unsupported_response
            else:
                log.info("no rule matched and chat gave no answer for %r", text)
                response = self.chat_unavailable_response
            return CommandResult(
                status=CommandStatus.UNSUPPORTED,
                response=response,
                data=message_for(match, text, command.confidence),
            )

        if self.show_events:
            # Still an event line, so the console shows what Diana did even
            # when the answer came from a model rather than a rule.
            print("EVENT: CHAT", flush=True)
        return CommandResult(
            status=CommandStatus.CHAT,
            response=answer,
            data={**message_for(match, text, command.confidence),
                  "source": "chat", "code": "CHAT"},
        )

    def _reply_for(self, match: Match, state: dict, text: str) -> str:
        """What to say back, per the `reply` mode.

        "intent" is the default and the only mode that is useful in normal
        operation: she says what happened, in words, after doing it. "echo" is
        the debugging mode, and it is worth keeping precisely because the
        default hides mistakes. A confident sentence in the wrong words -- "the
        music is now paused" for a misheard "pause the music" -- looks exactly
        like a working assistant. The transcript, with all its mistakes visible,
        does not. So: echo until the recogniser is trustworthy, then intent.
        """
        if self.reply == "echo":
            return text
        if self.reply == "ack":
            return PrintCommandHandler._ACKS[next(self._ack_at) % len(PrintCommandHandler._ACKS)]
        if self.reply == "none":
            return ""
        return speak_for(match, state, self.registry, self.placeholder_response)

    def close(self) -> None:
        # Worth reporting: the outbox is the only record of what a node would
        # have been told, and it disappears with the process.
        log.info("%d device messages queued", len(self.registry.outbox))
        # The model's own tally. "asked 0" is a very different evening from
        # "asked 9, failed 9", and only one of those is a recogniser problem.
        self.chat.close()


def command_from_transcription(
    transcript: Transcription, utterance_s: float = 0.0
) -> Command:
    """Adapter: stt.Transcription -> command.Command.

    This one function is the entire coupling between the speech layer and the
    brain. When your own recogniser produces a different structure, this is the
    single place you adapt it -- the rest of the pipeline never sees a
    Transcription at all.
    """
    return Command(
        text=transcript.text,
        at=time.time(),
        confidence=transcript.confidence,
        words=list(transcript.words),
        utterance_s=utterance_s,
    )
