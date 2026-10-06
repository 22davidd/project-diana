"""
chat.py -- what Diana says when nobody asked her to do anything.

The grammar in intents.py is a closed world. It knows about a fixed list of
devices and a fixed list of verbs, and when a sentence falls outside both, the
only honest answer is "I don't know how to do that yet." That answer is correct
and it is also the last thing you want to hear when you were not giving an
order -- you were asking what time it is, or complaining about the weather, or
just talking.

So there is a second thing behind the parser, and it is only consulted when
the parser found nothing:

    "turn on the lights"  ->  rules match           ->  LIGHTS_ON, act on it
    "how do I make lasagne" -> no rule, chat it     ->  speak the model's answer

The order is not negotiable. Commands are handled by the rule table and
*never* by the model, because a 0.5b model asked to act on your house is a
liability: it will cheerfully agree that it has turned the lights on when it
has not touched anything. The model is a conversational fallback and nothing
else. It has no tools, no access to the registry, and no way to emit an event
code, and that is by construction rather than by policy.

Why a local model
-----------------
Ollama, on localhost, with qwen2.5:0.5b. Not a cloud API, and the difference
matters more than the usual reasons: a voice assistant is a microphone with a
person attached to it, the latency budget is about a second because the answer
is spoken, and a 400 MB model on your own CPU is the only thing that fits in
that budget. It also means `chat_enabled: false` gives back an assistant that
works with no network at all, which is a property worth being able to test.

What a small model needs, and does not get for free
---------------------------------------------------
A 0.5b model will happily answer a spoken question in four paragraphs of
markdown, which is unusable here for reasons that have nothing to do with
quality. Piper reads at about three words a second and the user has stopped
listening well before that, so a long answer is not a thorough answer, it is a
reply that gets interrupted by the next wake word. Three defences, in order:
the system prompt, `num_predict`, and `shorten()`.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


DEFAULT_SYSTEM = (
    "You are Diana, a voice assistant in someone's house. "
    "You are answering out loud, so reply in ONE short sentence, under 15 words. "
    "Never use lists, markdown, or emoji. "
    "Never say you turned anything on or off or opened anything -- you cannot "
    "control the house, another system does that. "
    "If you are asked to do something physical, say you cannot."
)


@dataclass
class ChatStats:
    """Counters, so the log can tell you whether the model is being reached."""

    asked: int = 0
    answered: int = 0
    failed: int = 0
    total_s: float = 0.0
    last_error: str = ""

    def record(self, ok: bool, elapsed: float, error: str = "") -> None:
        if ok:
            self.answered += 1
            self.total_s += elapsed
        else:
            self.failed += 1
            self.last_error = error


class OllamaChat:
    """A minimal client for one local Ollama model.

    Stdlib `urllib` rather than `requests`, deliberately. The project's one
    runtime rule is that it works offline, and a client that can only speak
    HTTP because a package is installed is a client that breaks on a fresh
    checkout in a way that is annoying to diagnose.
    """

    def __init__(
        self,
        model: str = "qwen2.5:0.5b",
        host: str = "http://localhost:11434",
        timeout_s: float = 20.0,
        max_reply_chars: int = 200,
        system: str = DEFAULT_SYSTEM,
        num_predict: int = 64,
        temperature: float = 0.6,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.max_reply_chars = max_reply_chars
        self.system = system
        self.num_predict = num_predict
        self.temperature = temperature
        # `num_predict` is the more reliable of the two length controls. The
        # system prompt asks for one short sentence and is obeyed most of the
        # time; when it is not obeyed the model keeps going, and a hard cap on
        # tokens is the only thing that reliably stops it. A long answer here
        # is not a thorough answer -- piper reads at about three words a second
        # and the reply is still being spoken when the user gives up.
        self.stats = ChatStats()
        # Turnover matters. qwen2.5:0.5b has a ~32k context, so a long
        # conversation would not blow up -- but the prompt is re-evaluated on
        # every single request, and re-reading an hour of chat to answer
        # "thanks" is exactly the latency this fallback cannot afford.
        self._history: list[dict[str, str]] = []
        self.history_turns = 6
        self._warned_unavailable = False

    disabled = False
    """Chat was asked for and is wired up. See NullChat for the other answer.

    Read by the handler to decide *which* failure to apologise for, which is
    the only reason it exists: "I don't know how to do that" is a reasonable
    thing to say when chat is switched off, and a misleading one when ollama
    is simply not running.
    """

    # -- availability ------------------------------------------------------
    def available(self, timeout_s: float = 1.5) -> bool:
        """Is the model there? Never raises, and never takes long.

        A separate short-timeout probe rather than reusing the reply timeout,
        because this is called for the startup banner and a banner that waits
        thirty seconds for a server that is not running is worse than no
        banner.
        """
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=timeout_s) as fh:
                names = [m.get("name", "") for m in json.load(fh).get("models", [])]
        except (urllib.error.URLError, OSError, ValueError) as exc:
            log.debug("ollama not reachable at %s: %s", self.host, exc)
            return False
        # Ollama reports "qwen2.5:0.5b" and also bare "qwen2.5:0.5b:latest"
        # depending on version, so compare on the part before the tag.
        want = self.model.split(":")[0]
        return any(n.split(":")[0] == want for n in names)

    def describe(self) -> str:
        """One line for the startup banner."""
        if not self.available():
            return f"ollama[{self.model}] (not running)"
        return f"ollama[{self.model}]"

    # -- the one method that matters ---------------------------------------
    def reply(self, text: str) -> str:
        """The model's answer, or "" if it did not give one.

        Never raises. This is called from the audio thread's command handler,
        where an exception is a crash rather than an apology, and "no answer"
        is a perfectly good outcome that the caller already knows how to speak.
        """
        if not text.strip():
            return ""

        payload = {
            "model": self.model,
            "stream": False,
            "messages": self._messages(text),
            "options": {
                "num_predict": self.num_predict,
                "temperature": self.temperature,
            },
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        self.stats.asked += 1
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as fh:
                body = json.load(fh)
            content = body.get("message", {}).get("content", "")
        except urllib.error.HTTPError as exc:
            # A 404 here means the model is not pulled. That is the single
            # most likely failure and it deserves a real message rather than a
            # generic one, because it is fixed by one command.
            if exc.code == 404:
                log.warning(
                    "model %r is not installed; run: ollama pull %s", self.model, self.model
                )
                self.stats.record(False, time.monotonic() - started, "model not found")
            else:
                log.warning("ollama returned HTTP %s", exc.code)
                self.stats.record(False, time.monotonic() - started, f"http {exc.code}")
            return ""
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if not self._warned_unavailable:
                log.warning(
                    "chat unavailable (%s); commands still work, "
                    "set chat_enabled=false to silence this",
                    exc,
                )
                self._warned_unavailable = True
            self.stats.record(False, time.monotonic() - started, str(exc))
            return ""

        answer = shorten(content, self.max_reply_chars)
        elapsed = time.monotonic() - started
        if not answer:
            self.stats.record(False, elapsed, "empty answer")
            return ""

        self._remember(text, answer)
        self.stats.record(True, elapsed)
        log.info(
            "chat answered in %.1fs: %r", elapsed, answer,
        )
        return answer

    # -- history -----------------------------------------------------------
    def _messages(self, text: str) -> list[dict[str, str]]:
        system = {"role": "system", "content": self.system}
        recent = self._history[-self.history_turns :] if self.history_turns else []
        return [system, *recent, {"role": "user", "content": text}]

    def _remember(self, question: str, answer: str) -> None:
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})

    def reset(self) -> None:
        self._history.clear()

    def close(self) -> None:
        """Report, so the log distinguishes "never tried" from "always failed"."""
        s = self.stats
        if not s.asked:
            return
        log.info(
            "chat: %d asked, %d answered, %d failed%s",
            s.asked, s.answered, s.failed,
            f" ({s.total_s / max(1, s.answered):.1f}s avg)" if s.answered else "",
        )
        if s.failed:
            log.info("chat last error: %s", s.last_error)


# ======================================================================
# Making a model's answer speakable
# ======================================================================
_MD_NOISE = re.compile(r"[*_`#>\[\]()]")
_WHITESPACE = re.compile(r"\s+")


def shorten(text: str, limit: int) -> str:
    """Turn generated text into one short spoken sentence.

    Three separate jobs, and the order is forced:

    1. Markdown removal. A small model decorates its answer the way a
       documentation-writing model does, and piper reads "*" as nothing and
       "#" as "hash", so a bulleted answer becomes spoken gibberish.
    2. Whitespace collapse and newline removal. Piper reads a line break as a
       pause at best and as an error at worst.
    3. Truncation at a sentence boundary, not at the character limit. Cutting
       mid-word produces a reply that sounds broken, which is a worse outcome
       than saying less.
    """
    cleaned = _WHITESPACE.sub(" ", _MD_NOISE.sub("", text or "")).strip()
    if not cleaned:
        return ""

    if len(cleaned) <= limit:
        return cleaned

    # Prefer the last complete sentence that fits, so the answer ends the way
    # a spoken answer should: on a full stop.
    window = cleaned[: limit + 1]
    boundary = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if boundary >= 20:
        return window[: boundary + 1].strip()
    if boundary >= 0:
        return window[: boundary + 1].strip()

    # No punctuation to cut on -- fall back to a word boundary so the reply
    # ends between words rather than mid-word.
    cut = window.rfind(" ", 0, limit)
    return (window[:cut] if cut > 0 else window[:limit]).rstrip(" ,;:-") + "."


class NullChat:
    """Stands in for OllamaChat when chat is switched off.

    Same interface, no network, always declines. Exists so that the caller can
    hold a chat object unconditionally and never write `if chat is not None`
    around every call site -- a null object removes a branch, it does not
    move it.
    """

    stats = ChatStats()
    disabled = True
    """No model is behind this, and saying so is the point. See
    OllamaChat.disabled for why the handler cares."""

    def __init__(self, reason: str = "chat disabled") -> None:
        self.model = ""
        self.reason = reason

    def available(self, timeout_s: float = 1.5) -> bool:
        return False

    def describe(self) -> str:
        return "off"

    def reply(self, text: str) -> str:
        return ""

    def reset(self) -> None:
        """Nothing is remembered, so there is nothing to forget."""

    def close(self) -> None:
        """Nothing to release."""


def build_chat(cfg) -> object:
    """Config -> a chat object. See build_diana's wiring notes."""
    if not cfg.chat_enabled:
        return NullChat()
    return OllamaChat(
        model=cfg.chat_model,
        host=cfg.chat_host,
        timeout_s=cfg.chat_timeout_s,
        max_reply_chars=cfg.chat_max_reply_chars,
    )
