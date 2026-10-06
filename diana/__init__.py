"""
diana -- Project Diana, V0.

An offline voice assistant loop:

    microphone -> wake word -> "Yes?" -> listen -> speech-to-text -> print

Each concern lives in its own module behind an abstract interface, so the
speech recogniser and the command handler can be replaced with your own
trained models without touching anything else.
"""

__version__ = "0.1.0"
