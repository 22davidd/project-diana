"""
intents.py -- text in, a structured action out.

    "turn on the lights"  ->  Match(code="LIGHTS_ON", intent=Intent.DEVICE_ON,
                                     target="light", action="on")

The behaviour this replaces was `PrintCommandHandler` repeating whatever the
recogniser heard, which is a useful debugging tool and a useless assistant: it
can prove Diana listened to you, but it cannot *do* anything, and it cannot
answer a follow-up question because it remembers nothing. This module is the
first thing in the pipeline that decides anything.

What it deliberately is not
---------------------------
Not a neural model. It is a list of rules over tokens: fast enough to run on the
audio thread, nothing to download, and -- the reason it is worth building first
-- **it fails in a way you can read**. Every match comes back with the rule that
fired and the words that triggered it, so when Diana says "lights are now on"
for a phrase that had nothing to do with lights, the log names the rule that
was wrong instead of leaving you with a black box.

The shape is the shape your Transformer will have, and that is the point.
`CommandHandler.handle()` returns a `CommandResult` whose `data` is a
structured message either way; the only thing that changes is whether that
message came from a rule table or from a model. Swap this class out and nothing
else in the project moves.

Raw tokens versus clean tokens
------------------------------
Two token lists, and the difference is not pedantry:

    "what is the weather like"  ->  raw:    [what, is, the, weather, like]
                                 ->  tokens: [weather, like]

* `raw` is lowercase words with the punctuation gone. Multi-word phrase tables
  are matched against it, because those tables are written as people speak
  ("how hot is it", "turn the volume up") and the articles inside them are real
  words, not noise.
* `tokens` is `raw` minus filler, and it is what device names and single-word
  tests use. "the lights" and "lights" have to reach the same device.

Getting this backwards is a real bug rather than a theoretical one: stripping
filler *before* phrase matching deletes the "is" out of "how hot is it" and the
weather rule silently stops firing for half the ways people ask.

Where the vocabulary lives
-------------------------
Device words are **not** listed here. They live in devices.py, on the device
itself (`Device.names`), so "lamp" resolving to the lights is one entry in one
place rather than a word in a list inside a parser that has no idea what a lamp
is. This module adds the grammar -- which word orders mean what -- and the
devices supply the nouns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from .devices import Device, DeviceRegistry, capability_for, default_registry

log = logging.getLogger(__name__)


class Intent(Enum):
    """What was asked for.

    The value is the `intent` field of the message that goes to the device
    layer, so these strings are part of the wire format rather than decoration.
    """

    DEVICE_ON = "device.on"
    DEVICE_OFF = "device.off"
    DEVICE_TOGGLE = "device.toggle"
    DEVICE_QUERY = "device.query"

    MEDIA_PLAY = "media.play"
    MEDIA_PAUSE = "media.pause"
    MEDIA_STOP = "media.stop"
    MEDIA_NEXT = "media.next"
    MEDIA_PREVIOUS = "media.previous"
    MEDIA_VOLUME_UP = "media.volume_up"
    MEDIA_VOLUME_DOWN = "media.volume_down"
    MEDIA_MUTE = "media.mute"
    MEDIA_UNMUTE = "media.unmute"

    APP_OPEN = "app.open"
    APP_CLOSE = "app.close"

    WEATHER = "weather.query"
    TIMER_SET = "timer.set"

    UNKNOWN = "unknown"


# Which `action` each intent means to the device layer. Intent and action are
# separate because five intents share one action vocabulary, and because the
# wire message carries both: the intent says what was understood, the action
# says what to do about it. Intents absent from this table do not change
# anything -- a question is not a command.
ACTION_OF = {
    Intent.DEVICE_ON: "on",
    Intent.DEVICE_OFF: "off",
    Intent.DEVICE_TOGGLE: "toggle",
    Intent.MEDIA_PLAY: "play",
    Intent.MEDIA_PAUSE: "pause",
    Intent.MEDIA_STOP: "stop",
    Intent.MEDIA_NEXT: "next",
    Intent.MEDIA_PREVIOUS: "previous",
    Intent.MEDIA_VOLUME_UP: "volume_up",
    Intent.MEDIA_VOLUME_DOWN: "volume_down",
    Intent.MEDIA_MUTE: "mute",
    Intent.MEDIA_UNMUTE: "unmute",
    Intent.APP_OPEN: "open",
    Intent.APP_CLOSE: "close",
}

# Intents that only ask. They are answered from the registry and put no message
# in the outbox, because a question is not something a node has to be told to
# do -- yet.
#
# WEATHER belongs here even though it is also a placeholder. It was left out
# once, and the symptom was a `device message: weather -> ` line in the log for
# every weather question: an outbox entry with an empty action, addressed to a
# device that has nothing to do. A node implementing that contract would have
# no idea what to do with it. Asking is not doing, whichever module noticed.
QUERY_INTENTS = frozenset({Intent.DEVICE_QUERY, Intent.WEATHER})

# Intents with no data source behind them. They are understood perfectly well
# and cannot be *fulfilled*, which is a different failure from not
# understanding, and gets a different reply. See speak_for().
PLACEHOLDER_INTENTS = frozenset({Intent.WEATHER, Intent.TIMER_SET})


@dataclass(frozen=True)
class Match:
    """One utterance, understood.

    Frozen because it is a value handed between the parser, the handler and the
    device layer, and none of them has any business editing someone else's
    understanding of what you said.
    """

    intent: Intent
    target: str = ""
    """A device id from the registry: "light", "tv", "music", "all"."""

    action: str = ""
    """What to do: "on", "play", "volume_up". Empty for a question."""

    slots: dict = field(default_factory=dict)
    """Everything that is not the subject or the verb: {"app": "spotify"},
    {"seconds": 600}. Free-form on purpose -- an app name cannot be enumerated
    in advance, and a future intent may need a slot this version never heard of."""

    code: str = "UNKNOWN"
    """The event code: printed to the console, sent in the message.
    LIGHTS_ON, MUSIC_PLAY, APP_OPEN_SPOTIFY, WEATHER_QUERY."""

    rule: str = ""
    """Which rule fired. In the log, always; it is the difference between a
    parser you can debug and a parser you can only complain about."""

    response: str = ""
    """A default reply. The real one comes from speak_for(), which needs the
    state the action produced -- "toggle" cannot know what it is going to say
    until it has looked."""

    @property
    def known(self) -> bool:
        return self.intent is not Intent.UNKNOWN


@dataclass(frozen=True)
class Utterance:
    """One transcript, tokenised both ways."""

    text: str
    raw: list[str]
    """Lowercase words, punctuation removed. Phrase tables match against this."""

    tokens: list[str]
    """`raw` minus filler. Device names and single-word tests use this."""


# ======================================================================
# Grammar
# ======================================================================
# Words that carry no meaning, removed before anything else looks at the tokens.
# A recogniser produces them constantly ("please", "can you", "diana") and none
# of them should be able to change what you asked for.
#
# Note what is *not* in here: "on", "off", "up", "down", "everything". They look
# like filler and are not. "on" is the most common word in this entire command
# vocabulary, and stripping it would turn "turn the lights off" into "turn the
# lights" -- an on-switch in an off-switch's clothing.
FILLER = frozenset(
    """
    please kindly just now then ok okay yeah yes hey diana
    what whats how
    the a an my our your this that these those
    me us it them
    can could would will shall may might do does did
    to of for at with and also really quite some any something
    """.split()
)

# Word -> number, so "ten minutes" and "10 minutes" are the same request.
NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "ninety": 90, "half": 0.5,
}

TIME_UNITS = {
    "second": 1, "seconds": 1,
    "minute": 60, "minutes": 60,
    "hour": 3600, "hours": 3600,
}

# Verbs that mean power, used by the sentence-final-particle rule ("switch the
# lights *on*").
POWER_VERBS = frozenset("turn switch power put flip toggle kill shut".split())

# Verbs that mean "launch a program".
APP_VERBS = ("open", "launch", "start", "close", "quit", "exit")
APP_CLOSE_VERBS = ("close", "quit", "exit")

# Words that make a sentence a weather question whether or not it contains the
# word "weather". "how hot is it" is a weather question, and insisting on the
# literal noun would miss most of the ways people actually ask.
WEATHER_PHRASES = (
    "how hot is it", "how cold is it", "is it going to rain", "will it rain",
    "is it cold", "is it hot", "do i need an umbrella", "do i need a jacket",
    "is it snowing", "whats the weather", "what is the weather",
    "whats it like outside", "how is the weather", "is it warm",
)

# Action phrases, matched as contiguous runs against `raw`. The matcher takes
# the *longest* match rather than the first, so a specific phrase cannot be
# stolen by a shorter one that happens to start earlier.
#
# Volume is not in this table and has its own rule below. That is not tidiness:
# "turn the volume up" and "turn the volume down" both start with "turn", and
# a table that also contains "turn up" has to be scanned in a careful order to
# avoid volume changes turning into power commands.
ACTION_PHRASES: dict[str, Intent] = {
    "turn on": Intent.DEVICE_ON,
    "switch on": Intent.DEVICE_ON,
    "power on": Intent.DEVICE_ON,
    "turn off": Intent.DEVICE_OFF,
    "switch off": Intent.DEVICE_OFF,
    "power off": Intent.DEVICE_OFF,
    "turn out": Intent.DEVICE_OFF,
    "switch out": Intent.DEVICE_OFF,
    "shut off": Intent.DEVICE_OFF,
    "kill": Intent.DEVICE_OFF,
    "toggle": Intent.DEVICE_TOGGLE,
    "flip": Intent.DEVICE_TOGGLE,
    "play": Intent.MEDIA_PLAY,
    "resume": Intent.MEDIA_PLAY,
    "pause": Intent.MEDIA_PAUSE,
    "stop": Intent.MEDIA_STOP,
    "next": Intent.MEDIA_NEXT,
    "skip this": Intent.MEDIA_NEXT,
    "skip": Intent.MEDIA_NEXT,
    "previous track": Intent.MEDIA_PREVIOUS,
    "last track": Intent.MEDIA_PREVIOUS,
    "previous": Intent.MEDIA_PREVIOUS,
    "mute": Intent.MEDIA_MUTE,
    "unmute": Intent.MEDIA_UNMUTE,
}

# The intents an action phrase may produce. A rule that finds "play" and
# nothing else in the sentence is not a command to do anything.
POWER_INTENTS = frozenset({Intent.DEVICE_ON, Intent.DEVICE_OFF, Intent.DEVICE_TOGGLE})
MEDIA_INTENTS = frozenset(
    {
        Intent.MEDIA_PLAY, Intent.MEDIA_PAUSE, Intent.MEDIA_STOP,
        Intent.MEDIA_NEXT, Intent.MEDIA_PREVIOUS, Intent.MEDIA_VOLUME_UP,
        Intent.MEDIA_VOLUME_DOWN, Intent.MEDIA_MUTE, Intent.MEDIA_UNMUTE,
    }
)


def _words(text: str) -> list[str]:
    """Lowercase word list, punctuation discarded."""
    return re.findall(r"[a-z0-9']+", text.lower())


def parse_utterance(text: str) -> Utterance:
    """Tokenise once, both ways. See the module docstring for why both exist."""
    raw = _words(text)
    return Utterance(text=text, raw=raw, tokens=[t for t in raw if t not in FILLER])


def find_phrase(tokens: list[str], phrases) -> tuple[int, str] | None:
    """The longest phrase in `phrases` present as a contiguous run.

    Longest-wins, not first-wins: "skip this" must beat "skip", or the longest
    phrasing in the language is decorative.
    """
    best: tuple[int, str] | None = None
    for phrase in phrases:
        words = phrase.split()
        span = len(words)
        for i in range(len(tokens) - span + 1):
            if tokens[i : i + span] == words and (best is None or span > len(best[1].split())):
                best = (i, phrase)
    return best


# ======================================================================
# The parser
# ======================================================================
class CommandParser:
    """Turns a transcript into a `Match`.

    Rules are tried in order and the first that fires wins, so the ordering *is*
    the design: each rule is strictly more specific than the one after it.
    Weather is checked before apps, apps before volume, volume before power,
    power before media, media before questions. A rule that fires on everything
    is a rule that will one day fire on "make me a sandwich", so the general
    ones go last and each one has to justify its own trigger.
    """

    def __init__(self, registry: DeviceRegistry | None = None) -> None:
        self.registry = registry if registry is not None else default_registry()
        # Noun -> device, built from the catalogue rather than from a word list
        # in this file, so adding a device cannot leave the parser behind.
        # Names go through the same tokeniser as the input, so a name can never
        # contain a word the filter would have eaten ("all of it" -> "all").
        self._nouns: dict[str, Device] = {}
        for device in self.registry.all():
            for name in device.names:
                key = " ".join(_words(name))
                if key:
                    self._nouns.setdefault(key, device)
        # Longest phrase first, so "music player" is found before "music".
        self._noun_phrases = sorted(self._nouns, key=lambda n: -len(n.split()))
        log.info(
            "command parser ready: %d devices, %d words they answer to",
            len(self.registry), len(self._nouns),
        )

    # -- entry point ------------------------------------------------------
    def parse(self, text: str) -> Match:
        """Understand one utterance. Pure: no I/O, no state, no clock.

        Fast enough to run on the audio thread by four orders of magnitude --
        a 12-word sentence is a few dozen dict lookups.
        """
        utterance = parse_utterance(text)
        if not utterance.tokens:
            return Match(intent=Intent.UNKNOWN, rule="empty")

        for rule in (
            self._rule_weather,
            self._rule_timer,
            self._rule_app,
            self._rule_volume,
            self._rule_power,
            self._rule_media,
            self._rule_query,
        ):
            match = rule(utterance)
            if match is not None:
                log.info(
                    "parsed %r -> %s  (rule=%s intent=%s target=%r slots=%s)",
                    text, match.code, match.rule, match.intent.value,
                    match.target, match.slots or "{}",
                )
                return match

        log.info("parsed %r -> UNKNOWN (no rule fired)", text)
        return Match(intent=Intent.UNKNOWN, rule="none")

    # -- helpers ----------------------------------------------------------
    def _device_in(self, tokens: list[str]) -> tuple[int, str, Device] | None:
        """A device mentioned in the tokens, as (index, phrase, device)."""
        for phrase in self._noun_phrases:
            words = phrase.split()
            span = len(words)
            for i in range(len(tokens) - span + 1):
                if tokens[i : i + span] == words:
                    return i, phrase, self._nouns[phrase]
        return None

    def _action_phrase(self, utterance: Utterance) -> tuple[str, Intent] | None:
        """The longest action phrase in the sentence, with its intent."""
        hit = find_phrase(utterance.raw, ACTION_PHRASES)
        return None if hit is None else (hit[1], ACTION_PHRASES[hit[1]])

    def _default_media(self) -> Device | None:
        """The device a bare "play" or "pause" means.

        A default is defensible here and nowhere else: there is exactly one
        thing in the catalogue that can play music, so "pause" with no object is
        unambiguous. Power gets no such shortcut, because "turn it on" could
        mean any device in the house and guessing would be how you turn the
        kettle on by asking about the radio.
        """
        for device in self.registry.all():
            if device.expands_to:
                # A group is not a speaker, and it claims no capabilities of
                # its own, so without this it matches on capability alone and a
                # bare "play" could land on "everything".
                continue
            if device.supports("media") and device.supports("volume"):
                return device
        return None

    def _allowed(self, device: Device | None, intent: Intent) -> Device | None:
        """`device`, but only if it can do what `intent` asks.

        The registry is the authority on capability, so a rule that has found a
        device still has to ask. This is what stops "turn off the weather" from
        being carried out by a parser that matched a word.
        """
        if device is None:
            return None
        return device if device.supports(capability_for(ACTION_OF.get(intent, ""))) else None

    # -- rules ------------------------------------------------------------
    def _rule_weather(self, u: Utterance) -> Match | None:
        """"What's the weather?" -- and "how hot is it", and "will it rain".

        Matched on `raw`, because every phrasing in that table contains a word
        the filler filter removes ("is", "it"). Matching the stripped tokens
        instead is how a weather rule silently stops working while still looking
        correct in the source.
        """
        if find_phrase(u.raw, WEATHER_PHRASES):
            return Match(Intent.WEATHER, "weather", code="WEATHER_QUERY", rule="weather.phrase")

        found = self._device_in(u.tokens)
        if found and found[2].id == "weather":
            return Match(Intent.WEATHER, "weather", code="WEATHER_QUERY", rule="weather.noun")
        return None

    def _rule_timer(self, u: Utterance) -> Match | None:
        """"Set a timer for ten minutes." The duration is parsed, not ignored --
        a timer slot you cannot fill is a slot you will have to retrofit later,
        and the retrofit is always at the worst possible time."""
        wants_timer = "timer" in u.tokens or "alarm" in u.tokens or "remind" in u.tokens
        if not wants_timer:
            return None
        seconds = self._duration(u.tokens)
        if not seconds and "timer" not in u.tokens and "alarm" not in u.tokens:
            # "remind me" on its own is not a timer; there has to be a number.
            return None
        return Match(
            Intent.TIMER_SET, "timer",
            slots={"seconds": seconds} if seconds else {},
            code="TIMER_SET", rule="timer",
        )

    @staticmethod
    def _duration(tokens: list[str]) -> int:
        """Seconds in "for ten minutes", or 0 if there is no number to find.

        A number and a unit, in either order, with up to two words between
        them -- because a recogniser will cheerfully hand you "ten" and
        "minutes" three words apart, and "an hour and a half" is two pairs.
        """
        total = 0.0
        for i, token in enumerate(tokens):
            unit = TIME_UNITS.get(token)
            if unit is None:
                continue
            for back in range(1, 4):
                if i - back < 0:
                    break
                candidate = tokens[i - back]
                value = NUMBER_WORDS.get(candidate)
                if value is None and candidate.isdigit():
                    value = int(candidate)
                if value is not None:
                    total += value * unit
                    break
        return int(total)

    def _rule_app(self, u: Utterance) -> Match | None:
        """"Open spotify" -- a verb, and a noun that is not one of our devices.

        That second half is the entire rule. "Start the music" contains a verb
        this rule would otherwise claim, but the music *is* a device, so it
        declines and the media rule handles it.
        """
        found = self._device_in(u.tokens)
        if found and found[2].id not in ("weather", "timer"):
            return None

        verb = find_phrase(u.raw, APP_VERBS)
        if verb is None:
            return None
        index, phrase = verb
        # Drop the verb and the words that merely announce it; whatever is left
        # is the app's name. "open the browser" -> "browser".
        rest = [t for t in u.tokens if t not in APP_VERBS and t not in ("app", "application", "program")]
        if not rest:
            return None  # "open the app" is genuinely ambiguous; let it fail

        name = " ".join(rest)
        closing = phrase in APP_CLOSE_VERBS
        code = f"APP_{'CLOSE' if closing else 'OPEN'}_{_slug(name)}"
        return Match(
            Intent.APP_CLOSE if closing else Intent.APP_OPEN,
            target="app",
            action=ACTION_OF[Intent.APP_CLOSE if closing else Intent.APP_OPEN],
            slots={"app": name},
            code=code,
            rule=f"app.{phrase}",
        )

    def _rule_volume(self, u: Utterance) -> Match | None:
        """"Turn it up", "make it louder", "turn the music down".

        Its own rule because volume has a grammar the other actions do not: the
        direction can be a word ("up"), an adjective ("louder"), or implied by
        nothing at all ("volume" with no direction, which means the opposite of
        whatever it was -- a request this version declines rather than guesses).
        """
        tokens = u.tokens
        up, down = "up" in tokens, "down" in tokens
        louder, quieter = "louder" in tokens, "quieter" in tokens

        triggered = (
            (up or down) and ("volume" in tokens or any(v in tokens for v in POWER_VERBS))
        ) or louder or quieter
        if not triggered:
            return None

        if louder and not quieter:
            intent = Intent.MEDIA_VOLUME_UP
        elif quieter and not louder:
            intent = Intent.MEDIA_VOLUME_DOWN
        elif up and not down:
            intent = Intent.MEDIA_VOLUME_UP
        elif down and not up:
            intent = Intent.MEDIA_VOLUME_DOWN
        else:
            return None  # "up and down", or "volume" with no direction: ambiguous

        device = self._device_in(tokens)
        target = self._allowed(device[2] if device else None, intent) or self._default_media()
        if target is None:
            return None
        return Match(
            intent, target.id, ACTION_OF[intent],
            code=f"{target.code}_{ACTION_OF[intent].upper()}", rule=f"volume.{'up' if 'up' in tokens else 'down'}",
        )

    def _rule_power(self, u: Utterance) -> Match | None:
        """On, off and toggle, for anything with a power switch.

        Two shapes have to be recognised, because English puts the verb and the
        particle in different places depending on which is emphasised:

            turn on the lights         verb first
            switch the lights on       particle last

        Contiguous phrase matching only finds the first, so the second gets its
        own trigger: a power verb somewhere, and "on" or "off" somewhere.
        """
        tokens = u.tokens
        intent: Intent | None = None
        rule = ""

        action = self._action_phrase(u)
        if action is not None and action[1] in POWER_INTENTS:
            intent, rule = action[1], f"power.{action[0]}"
        elif any(v in tokens for v in POWER_VERBS) and ("on" in tokens or "off" in tokens):
            intent = Intent.DEVICE_ON if "on" in tokens else Intent.DEVICE_OFF
            rule = "power.trailing"

        if intent is None:
            return None

        found = self._device_in(tokens)
        target = self._allowed(found[2] if found else None, intent)
        if target is None:
            # No subject ("turn it on") or a subject that cannot be powered
            # ("turn off the weather"). Both are the parser declining to guess,
            # which is the correct answer in a house with more than one lamp.
            return None

        if not target.expands_to and target.supports("media"):
            # "Put the music on" is a power phrase about a speaker, and what a
            # person means by it is play. Without these two lines the rule
            # above would happily power the speaker on and say nothing, which
            # is technically correct and completely useless.
            #
            # Groups are excluded, and that exclusion is not a detail. The
            # "all" device lists no capabilities -- a group is not a thing, it
            # is a fan-out -- so `supports()` answers yes to everything and
            # "turn everything off" would come back as STOP. The check has to
            # be about a speaker specifically, not about a device that has not
            # said no.
            intent = Intent.MEDIA_PLAY if intent is Intent.DEVICE_ON else Intent.MEDIA_STOP
            rule += "+media"

        action_name = ACTION_OF[intent]
        return Match(
            intent, target.id, action_name,
            code=f"{target.code}_{action_name.upper()}", rule=rule,
        )

    def _rule_media(self, u: Utterance) -> Match | None:
        """Play, pause, skip -- for music and media players."""
        action = self._action_phrase(u)
        if action is None or action[1] not in MEDIA_INTENTS:
            return None
        intent = action[1]

        found = self._device_in(u.tokens)
        target = self._allowed(found[2] if found else None, intent) or self._default_media()
        if target is None:
            return None

        action_name = ACTION_OF[intent]
        return Match(
            intent, target.id, action_name,
            code=f"{target.code}_{action_name.upper()}", rule=f"media.{action[0]}",
        )

    def _rule_query(self, u: Utterance) -> Match | None:
        """"Are the lights on?" -- the reason the registry has to remember.

        A question word plus a device is the trigger, checked on `raw` because
        the question words are exactly the ones the filler filter eats.
        """
        if not find_phrase(u.raw, ("is", "are", "was", "does", "do", "did")):
            return None
        found = self._device_in(u.tokens)
        if found is None:
            return None
        target = found[2]

        # Which part of the state is being asked about. "Is the music
        # playing?" is about media state, not about whether the speaker has
        # power, and answering it from the wrong key is a confidently wrong
        # "no" -- so the question word picks the key.
        media_question = target.supports("media") and (
            "playing" in u.raw or "play" in u.raw or "on" not in u.tokens
        )
        return Match(
            Intent.DEVICE_QUERY, target.id, "",
            slots={"state": "media" if media_question else "power"},
            code=f"{target.code}_QUERY", rule="query.state",
        )


def _slug(name: str) -> str:
    """APP_OPEN_FIREFOX from "firefox"; non-alphanumerics become underscores."""
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_") or "APP"


# ======================================================================
# Speaking
# ======================================================================
def speak_for(
    match: Match,
    state: dict,
    registry: DeviceRegistry,
    placeholder: str = "I don't have a real source for that yet.",
) -> str:
    """The line to say, given the state the action produced.

    Takes the resulting state rather than the request, which is what makes
    three of these answers possible at all: "toggle the lights" cannot know what
    it will say until it has looked, and "are the lights on?" is only true if
    somebody remembered the last command.
    """
    if not match.known:
        return match.response or "I don't know how to do that yet."
    if match.intent in PLACEHOLDER_INTENTS:
        # Understood perfectly, cannot be fulfilled. The console line is the
        # honest record -- WEATHER_QUERY means she got the question -- and the
        # reply says plainly that she has no way to answer it. A made-up
        # forecast would be the one thing in this project that could actually
        # mislead somebody.
        return placeholder

    device = registry.get(match.target)
    if device is None:
        return placeholder

    who = device.spoken()
    if match.intent is Intent.DEVICE_ON:
        return f"{who} {_be(device)} now on."
    if match.intent is Intent.DEVICE_OFF:
        return f"{who} {_be(device)} now off."
    if match.intent is Intent.DEVICE_TOGGLE:
        return f"{who} {_be(device)} now {'on' if device.is_on() else 'off'}."
    if match.intent is Intent.DEVICE_QUERY:
        if match.slots.get("state") == "media":
            playing = state.get("media") == "playing"
            return f"{'yes' if playing else 'no'}, {who} {'is playing' if playing else 'is not playing'}."
        on = state.get("power") == "on"
        return f"{'yes' if on else 'no'}, {who} {_be(device)} {'on' if on else 'off'}."

    if match.intent is Intent.MEDIA_PLAY:
        return f"{who} is now playing."
    if match.intent is Intent.MEDIA_PAUSE:
        return f"{who} is now paused."
    if match.intent is Intent.MEDIA_STOP:
        return f"{who} is now stopped."
    if match.intent is Intent.MEDIA_NEXT:
        return "skipping to the next track."
    if match.intent is Intent.MEDIA_PREVIOUS:
        return "going back to the previous track."
    if match.intent in (Intent.MEDIA_VOLUME_UP, Intent.MEDIA_VOLUME_DOWN):
        return f"volume is now {int(state.get('volume', 0))}."
    if match.intent is Intent.MEDIA_MUTE:
        return f"{who} is muted."
    if match.intent is Intent.MEDIA_UNMUTE:
        return f"{who} is unmuted."

    if match.intent is Intent.APP_OPEN:
        return f"{match.slots.get('app', 'it')} is now open."
    if match.intent is Intent.APP_CLOSE:
        return f"{match.slots.get('app', 'it')} is now closed."

    return placeholder


def _be(device: Device) -> str:
    """"are" for the lights, "is" for the tv."""
    return "are" if device.plural else "is"


def message_for(match: Match, text: str, confidence: float) -> dict:
    """The structured payload: what a node would actually receive.

    Note what is not in it. No English. Every consumer of this message -- the
    device layer today, an MQTT node tomorrow -- is a program, and a program
    should not have to parse a sentence to discover whether the lights are on.
    The transcript rides along as `text` for the log, and nothing needs it.
    """
    return {
        "code": match.code,
        "intent": match.intent.value,
        "device": match.target,
        "action": match.action,
        "slots": dict(match.slots),
        "source": "voice",
        "text": text,
        "confidence": confidence,
    }
