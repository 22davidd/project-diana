"""
devices.py -- the things Diana can act on, and what she thinks their state is.

This started life as a deliberately empty placeholder: a dataclass and an
in-memory list, so the module boundary existed before there was anything to put
behind it. What is here now is still a placeholder in the only sense that
matters -- **nothing here touches the physical world**. There is no MQTT, no
HTTP, no ESP32. What there *is*, for the first time, is state.

Why state, and why here
-----------------------
Say "turn on the lights" and then "are the lights on?". The second question can
only be answered honestly if something remembers the first command. That memory
is the whole difference between an echo and an assistant, and it belongs in one
table rather than scattered through the parser.

So this module is the single source of truth for "what is on right now", and
`apply()` is the only thing allowed to change it. Two consequences worth knowing:

1. The parser (intents.py) never touches state. It decides *what* was asked
   for; this module decides what that means and what the device looks like
   afterwards. That is why `speak_for()` can take a device's new state as an
   argument instead of guessing.

2. A group is just a device that expands. "Turn everything off" is not a
   special case in the parser, it is `Device(id="all", expands_to=[...])`, and
   the same code path handles it.

What a real node will do
------------------------
`send()` already builds the message a node would receive, in the form the
project has agreed on since day one:

    {"intent": "device.control", "device": "living_room.light",
     "action": "on", "source": "voice", "ts": 1730000000}

It records it and logs it; it does not transmit it. That is the seam. The day
you add a transport, you put it in `send()` and nothing above this line moves --
the parser, the handler and the state table stay exactly as they are, which is
the point of putting the seam here in the first place.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)


# Which capability an action needs. Kept here, next to the state machine that
# has to honour it, because "turn off the weather" is not a parsing problem --
# it is the parser correctly failing to stop the registry from doing something
# silly, and the second check is the one that counts.
ACTION_CAPABILITY = {
    "on": "power",
    "off": "power",
    "toggle": "power",
    "play": "media",
    "pause": "media",
    "stop": "media",
    "next": "media",
    "previous": "media",
    "volume_up": "volume",
    "volume_down": "volume",
    "mute": "volume",
    "unmute": "volume",
    "open": "open",
    "close": "open",
}


def capability_for(action: str) -> str:
    """The capability an action requires, or "" if it needs none."""
    return ACTION_CAPABILITY.get(action, "")


class DeviceKind(Enum):
    """Coarse category, used for room grouping and default behaviour."""

    LIGHT = "light"
    SWITCH = "switch"
    CLIMATE = "climate"
    MEDIA = "media"
    SENSOR = "sensor"
    APPLIANCE = "appliance"
    APP = "app"
    SERVICE = "service"
    """A capability with no hardware behind it yet, e.g. the weather."""

    OTHER = "other"


@dataclass
class Device:
    """A controllable thing in the house.

    `address` is intentionally transport-agnostic. Today it could be an MQTT
    topic; later it might be a node id, a MAC address, or a mesh handle. By not
    committing to one, the registry does not have to be rewritten when the
    transport is chosen.

    The `code` / `label` / `names` / `plural` group exists for the other
    direction of travel: turning English into a structured action. `names` is
    what the recogniser may hear, `code` is what lands in the log and on the
    console (`LIGHTS_ON`), and `label` is what she says out loud. Keeping all
    three on the device means adding a lamp later is a data change, not a code
    change.
    """

    id: str
    name: str
    kind: DeviceKind = DeviceKind.OTHER
    room: str = ""
    address: str = ""
    online: bool = False
    state: dict = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)

    # -- the parser's view of the device ---------------------------------
    code: str = ""
    """Uppercase event-code prefix: "LIGHTS", "TV", "MUSIC". `code` + the
    action is what gets printed, e.g. LIGHTS_ON. Empty means this device is not
    the kind that gets a code -- the parser will not match it."""

    label: str = ""
    """What she calls it, in speech: "lights", "the tv". Defaults to the
    lowercase form of `name`."""

    names: list[str] = field(default_factory=list)
    """Every word or phrase that should resolve to this device. Matched as
    tokens, longest phrase first, so "living room light" beats "light"."""

    plural: bool = False
    """Decides "the lights are" versus "the tv is". Getting this wrong is the
    difference between an assistant that sounds like a person and one that
    sounds like a form letter."""

    expands_to: list[str] = field(default_factory=list)
    """Group members. "all" is the only one today, but nothing here is
    specialised to that: a group is a device whose action fans out."""

    def spoken(self) -> str:
        return self.label or self.name.lower()

    def is_on(self) -> bool:
        return self.state.get("power") == "on"

    def supports(self, capability: str) -> bool:
        return not self.capabilities or capability in self.capabilities


class DeviceRegistry:
    """Every known device, plus the authoritative copy of its state.

    The registry -- not the device, and definitely not the parser -- decides
    what an action means. That way a node that is offline, or answers late, or
    disagrees, is a fact in one table rather than a special case in every code
    path.
    """

    def __init__(self) -> None:
        self._devices: dict[str, Device] = {}
        # Everything `send()` was asked to transmit. The log is the test
        # double for a transport that does not exist yet: run a command and
        # read back the exact bytes a node would have got.
        self.outbox: list[dict] = []

    # -- membership -------------------------------------------------------
    def register(self, device: Device) -> None:
        self._devices[device.id] = device
        log.debug("registered device %s (%s)", device.id, device.kind.value)

    def get(self, device_id: str) -> Device | None:
        return self._devices.get(device_id)

    def all(self) -> list[Device]:
        return list(self._devices.values())

    def in_room(self, room: str) -> list[Device]:
        return [d for d in self._devices.values() if d.room == room]

    def __len__(self) -> int:
        return len(self._devices)

    def __contains__(self, device_id: object) -> bool:
        return device_id in self._devices

    # -- the seam a real transport plugs into ------------------------------
    def send(self, device_id: str, action: str, **params) -> dict:
        """Hand one action to the device layer.

        There is no transport yet, so this logs the message and appends it to
        `outbox`. The message is the contract:

            {"intent": "device.control", "device": "light", "action": "on",
             "source": "voice", "ts": 1730000000.0}

        `source` is here, not in the parser, because the source is a fact about
        *how* the request arrived, and the same request can arrive from a
        wall switch, a script, or a phone. Anything that reaches the devices
        should be able to say which, and a node may well care: a guest turning
        on the living-room lights is a different situation from the house doing
        it on a schedule.
        """
        message = {
            "intent": "device.control",
            "device": device_id,
            "action": action,
            "source": params.pop("source", "voice"),
            "ts": time.time(),
        }
        if params:
            message["params"] = params
        self.outbox.append(message)
        log.info("device message: %s -> %s%s", device_id, action, f" {params}" if params else "")
        return message

    # -- state -------------------------------------------------------------
    def apply(self, device_id: str, action: str, **params) -> list[Device]:
        """Carry out `action` on `device_id` and return every device affected.

        Returns a list because of groups: "turn everything off" affects five
        devices and the caller needs all five to write an honest confirmation.

        Unknown device or unknown action is not an exception. A voice assistant
        that raises on a phrase it has not seen is a voice assistant that dies
        mid-conversation, so the honest answer is an empty list plus a log line.
        """
        device = self._devices.get(device_id)
        if device is None:
            log.info("no such device: %s", device_id)
            return []

        needed = capability_for(action)
        targets = [device]
        if device.expands_to:
            targets = [
                self._devices[m]
                for m in device.expands_to
                if m in self._devices and self._devices[m].supports(needed)
            ]
            if not targets:
                log.info("nothing in group %s can %s", device_id, action)
                return []
        elif needed and not device.supports(needed):
            # The parser should not have offered this, but the registry is the
            # authority on what a device can do, so it says no as well.
            log.info("%s does not support %s", device_id, action)
            return []

        for target in targets:
            self.send(target.id, action, **params)
            self._mutate(target, action, **params)
        return targets

    def _mutate(self, device: Device, action: str, **params) -> None:
        """Update one device's state. The only writer of `state` in the project."""
        state = device.state
        if action == "on":
            state["power"] = "on"
        elif action == "off":
            state["power"] = "off"
        elif action == "toggle":
            state["power"] = "off" if device.is_on() else "on"
        elif action == "play":
            state["power"] = "on"
            state["media"] = "playing"
        elif action == "pause":
            state["media"] = "paused"
        elif action == "stop":
            state["media"] = "stopped"
            state["power"] = "off"
        elif action in ("next", "previous"):
            # Deliberately does not change `media`: skipping while paused stays
            # paused. The state table is the authority, so "play, pause, next"
            # must not silently resume the music.
            state["skipped"] = action
        elif action in ("volume_up", "volume_down"):
            # Clamped, because a device that can be turned up forever is a
            # device that will eventually be asked for volume 4000.
            step = int(params.get("step", 10))
            level = int(state.get("volume", 40))
            state["volume"] = max(0, min(100, level + (step if action == "volume_up" else -step)))
        elif action in ("mute", "unmute"):
            state["muted"] = action == "mute"
        elif action in ("open", "close"):
            state["open"] = action == "open"
        # Anything else: still sent, still logged, but no state we can track.
        # A real node would report the result back; until then the state table
        # is simply the part of the world we know about.
        log.debug("%s.%s -> %s", device.id, action, state)


# ======================================================================
# The catalogue
# ======================================================================
# Every one of these is a placeholder: there is no lamp, no television and no
# weather feed behind any of them. They are here so the *shape* of the system
# is real and testable, and so adding the actual hardware later is a one-line
# change rather than a refactor.
#
# To add a device: give it an id, a code, the words that mean it, and a plural
# flag. The parser picks it up automatically -- it builds its vocabulary from
# this list rather than from a second list inside intents.py.


def default_registry() -> DeviceRegistry:
    """A registry pre-loaded with the placeholder catalogue."""
    registry = DeviceRegistry()

    registry.register(
        Device(
            id="light",
            name="The lights",
            kind=DeviceKind.LIGHT,
            code="LIGHTS",
            label="lights",
            plural=True,
            names=[
                "light", "lights", "lamp", "lamps", "bulb", "bulbs",
                "lighting", "chandelier", "lightbulb",
            ],
            capabilities=["power"],
        )
    )
    registry.register(
        Device(
            id="tv",
            name="The TV",
            kind=DeviceKind.MEDIA,
            code="TV",
            label="tv",
            plural=False,
            names=["tv", "telly", "television", "screen", "tvs"],
            capabilities=["power"],
        )
    )
    registry.register(
        Device(
            id="music",
            name="The music",
            kind=DeviceKind.MEDIA,
            code="MUSIC",
            label="music",
            plural=False,
            names=["music", "radio", "song", "songs", "track", "playlist", "speaker"],
            capabilities=["power", "media", "volume"],
        )
    )
    registry.register(
        Device(
            id="weather",
            name="The weather",
            kind=DeviceKind.SERVICE,
            code="WEATHER",
            label="weather",
            plural=False,
            names=["weather", "forecast", "temperature", "outside"],
            # "query" and nothing else. Weather is a question, not a thing with
            # a power switch, and this is the field that stops "turn off the
            # weather" from being carried out.
            capabilities=["query"],
        )
    )
    registry.register(
        Device(
            # Not a thing in the house -- the thing an app runs *on*. It is
            # registered anyway because the parser targets it: "open spotify"
            # has to produce a device id, and the outbox message for it has to
            # look exactly like every other one. Without this entry the parser
            # matched correctly and the registry then refused to act, so the
            # honest sentence "I couldn't do that" was attached to a command
            # that had in fact been understood perfectly.
            id="app",
            name="The apps",
            kind=DeviceKind.APP,
            code="APP",
            label="it",
            plural=False,
            names=["app", "application", "program", "programme"],
            # "open" and nothing else, for the same reason the weather has
            # "query" and nothing else: it is the field that stops "turn off
            # the apps" from being carried out.
            capabilities=["open"],
        )
    )
    registry.register(
        Device(
            id="timer",
            name="The timers",
            kind=DeviceKind.SERVICE,
            code="TIMER",
            label="timer",
            plural=True,
            names=["timer", "timers", "alarm", "reminder"],
            capabilities=["query"],
        )
    )
    registry.register(
        Device(
            id="music_player",
            name="The music player",
            kind=DeviceKind.MEDIA,
            code="MUSIC",
            label="music player",
            plural=False,
            names=["player", "music player", "stereo"],
            capabilities=["power", "media", "volume"],
        )
    )

    # The group. Note it names every other device, and that list is built from
    # the registry itself rather than repeated by hand -- a second hand-written
    # copy of "what devices exist" is a list that is wrong by the time you add
    # the lamp.
    registry.register(
        Device(
            id="all",
            name="Everything",
            kind=DeviceKind.OTHER,
            code="ALL",
            label="everything",
            plural=False,
            names=["everything", "all", "all of it"],
            expands_to=[d.id for d in registry.all()],
        )
    )
    return registry
