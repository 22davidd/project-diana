# Project Diana

An offline voice assistant loop on your laptop. That is the entire scope.

```
microphone → wake word → "Yes?" → listen → speech-to-text → reply → idle
```

She runs continuously as a background daemon, hears her name, acknowledges it
out loud, transcribes what you say, answers, and goes back to waiting. No home
automation and no neural model in the middle yet, but the module layout is
arranged so that adding them does not mean rewriting this.

---

## Quick start

```bash
make install     # create .venv, install the packages
make models      # download the 40 MB speech model (once)
make voices      # download a 60 MB voice, so she can speak (once)
make selftest    # verify the logic with no microphone and no model
make run         # talk to her
```

Then say **"Diana"**, wait for her to answer `Yes?`, and say your command.

`make voices` is optional. Without it she prints what she would have said,
which is how she behaved before TTS arrived.

To run her in the background instead:

```bash
make start       # detach into the background
make status      # is she alive?
make log         # follow the log
make stop        # clean shutdown
```

---

## What she should look like

```
$ make run

==============================================================
  Project Diana  --  V0 voice loop
==============================================================
  sample rate      : 16000 Hz  (block 30 ms = 480 samples)
  input device     : system default
  wake phrases     : "diana", "hey diana", "ok diana"
  wake cooldown    : 1.5 s
  command window   : max 12.0 s, silence ends after 1.6 s
  model            : vosk-model-small-en-us-0.15
  audio           : microphone[default] @ 16000 Hz, 480 samples/block
  wake detector  : vosk-grammar
  speech-to-text : vosk
  command handler: intents
  fallback chat  : ollama[qwen2.5:0.5b]
  speaker         : piper[en_US-amy-medium]
--------------------------------------------------------------
  Listening for: diana, hey diana, ok diana
  Ctrl-C to stop.
==============================================================

Diana: Yes?
You: turn on the lights
COMMAND: turn on the lights
EVENT: LIGHTS_ON
Diana: lights are now on.
```

Two lines per command, and the split between them is the whole design. The
`EVENT:` line is machine-readable and cannot be a fluent mishearing: if it says
`LIGHTS_ON` then the grammar matched the lights, whatever the transcript looked
like. The sentence below it is the one that can be wrong in a way you cannot
see. `make text TEXT="..."` runs any phrase through the same stack with no
microphone at all, which is how you find out which rule fired.

---

## The architecture

```
diana/
├── main.py       the state machine and the event loop
├── config.py     every tunable setting, in one dataclass
├── audio.py      the microphone, and who is allowed to listen to it
├── wakeword.py   "did she hear her name?"        ─┐
├── stt.py        "what was just said?"            │  each defines an
├── command.py    "what should we do about it?"    │  abstract interface
├── speaker.py    "how do we answer?"             ─┘
├── intents.py    the grammar: words in, a structured action out
├── devices.py    the device catalogue and the state table
├── chat.py       the small local model, for everything else
└── selftest.py   the whole loop, tested with fakes
```

Dependencies are `sounddevice`, `vosk`, `numpy` and `piper-tts`. Everything
runs offline; there is no API key and no network call at runtime. The one
optional extra is `chat.py`, which talks to an Ollama server on localhost and
can be switched off with `chat_enabled: false`.

### The one idea that holds it together

Every replaceable piece is an abstract base class with a tiny contract:

| Module | Interface | Method you implement |
|---|---|---|
| `audio.py` | `AudioSource` | `start()`, `read() -> ndarray \| None`, `stop()` |
| `wakeword.py` | `WakeWordDetector` | `detect(block) -> WakeEvent \| None` |
| `stt.py` | `SpeechToText` | `start()`, `feed(block)`, `finish()` |
| `command.py` | `CommandHandler` | `handle(Command) -> CommandResult` |
| `speaker.py` | `Speaker` | `speak(text)`, `wait()` |

`main.py` knows only these interfaces. It never imports `vosk`. That is not
architectural decoration — it is the whole reason you will be able to drop in
your own trained speech model and your own Transformer without touching
anything else. There is one function where the wiring happens:

```python
# diana/main.py, in build_diana()
wake     = VoskWakeWordDetector(...)    # ← replace with your model
stt      = VoskSpeechToText(...)        # ← replace with your model
handler  = IntentCommandHandler(...)    # ← replace with your Transformer
speaker  = build_speaker(cfg)           # ← or swap in a different voice
```

Three lines, marked in the source. Everything else in the project is
unaffected.

To prove this is real rather than aspirational, `selftest.py` builds the
*same* `Diana` object with fakes for all five collaborators and runs the real
state machine against them. Run `make selftest`.

---

## How the loop works

`Diana` is an explicit state machine. Four states, and every transition is
logged with its reason:

```
              ┌──────────────────────────────────────────┐
              v                                          │
    ┌──────────────┐   wake word    ┌──────────┐  "Yes?"  ┌───────────┐
    │     IDLE     │───────────────>│ GREETING │─────────>│ LISTENING │
    │ listen for   │                │ say      │           │ speech to │
    │ "Diana"      │                │ response │           │ text      │
    └──────────────┘                └──────────┘           └───────────┘
          ^                                                      │
          │        done            silence        ┌────────────┐
          └───────────────────────/  \─────────────│ PROCESSING │
                 back to                timeout   │ run the    │
                 waiting                     /    │ handler    │
                                              /     └────────────┘
```

An enum rather than a set of booleans, because it makes certain bugs
*unrepresentable*. With an `is_listening` flag and a cooldown timer, a timer
expiring mid-command lets the wake word fire on the tail of an utterance, and
now two things run at once with no way to tell from the code which. With one
enum there is exactly one state, one owner per timeout, and the log line
`IDLE -> LISTENING (wake word)` tells you precisely what happened.

The loop itself is trivial:

```python
while not self._stop_requested:
    block = self.audio.read()      # 30 ms of audio
    self._dispatch(block)          # whoever owns the current state uses it
```

Everything interesting is in *when* each collaborator gets asked to act.

---

## Four details worth understanding


### 1. One owner for the microphone

`AudioSource` is the only thing that touches PortAudio. The wake word detector
and the recogniser never open the mic themselves — the state machine routes
each block to whichever one currently owns it. Two processes (or two streams)
competing for one input device is a real and confusing failure on Linux, and
the fix is structural rather than careful.

`read()` returns `None` when the source is finished, which is how a finite
source says "no more audio" without looking like an error.

### 2. Float32 in, 16-bit PCM out

PortAudio gives us float32 samples in `[-1, 1]`. Kaldi — and therefore Vosk,
and therefore every model you will probably train — wants signed 16-bit PCM.

So you must not write `block.tobytes()`. That hands Kaldi 4 bytes per sample,
it decodes them as 2, and you get twice the audio at the wrong amplitude. It
does not raise, does not warn, and returns confident garbage. This cost me
real debugging time during the build, so it is isolated in one function with
the explanation attached:

```python
# audio.py
def to_pcm16(block: np.ndarray) -> bytes:
    clipped = np.clip(block, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()
```

### 3. A closed grammar will invent your wake word

The wake word uses Vosk in *grammar mode*: instead of any word being legal,
you hand it a list of the only phrases allowed, plus `[unk]` for anything
else.

```
["diana", "hey diana", "ok diana", "[unk]"]
```

This is fully offline, needs no API key, and needs no training. But it has a
dirty secret: **the decoder is not allowed to output anything else.** When it
hears speech that is not one of your phrases, it does not stay quiet — it
picks whichever legal phrase is phonetically closest and pads with `[unk]`.

Measured on a real 8-second recording whose true transcript was a phone
number, the grammar returned:

```
[unk] diana [unk]
```

Diana would have woken up in the middle of a number. "Diana" was not in the
audio at all.

The fix is one rule: **the wake phrase must be the first thing said**
(`require_at_start=True`). That rejects the case above, and it cannot reject a
real interaction, because a wake word by definition opens the conversation —
nobody says "turn on the lights, Diana". The asymmetry is deliberate: a false
positive means Diana interrupts you at random, a false negative just means you
say her name again.

When you outgrow this — when you want a detector cheap enough for a
battery-powered node — swap in a purpose-built wake-word model. That is what
`WakeWordDetector` is for.

### 4. She must never listen to herself

The moment she gets a real voice, the loop acquires a new failure mode, and it
is not a subtle one:

```
Diana hears "Yes?" through the speakers → treats it as her name
→ says "Yes?" → hears "Yes?" → says "Yes?" → until you unplug her
```

This is why `Speaker` has a second method, `wait()`. The microphone is not
stopped while she talks — it keeps recording, and the sound card's input buffer
is filling up with her own voice the entire time. So by the time the state
machine looks at the microphone again, the first few blocks it reads are the
tail of what she just said. Measured on this laptop: two blocks, about 60 ms of
her own voice, waiting at the front of the queue.

Three separate defences, because each one covers a case the others do not:

| Defence | Where | What it stops |
|---|---|---|
| `speaker.wait()` before the microphone goes back to work | `_return_to_idle` | Reading her reply at all. Without it she hears the whole sentence. |
| `post_speak_discard_ms` after playback | `_begin_command_capture` | Her `Yes?` becoming the first words of your command. |
| `wake_cooldown_s`, re-armed after playback | `_return_to_idle` | Her reply re-triggering the wake word, which is the loop above. |

The reason this is a *table* rather than one clever fix is that each defence
protects a different consumer: the recogniser, the wake detector, and the
person trying to have a conversation. `selftest.py` scenario 7 uses a fake
speaker that takes real time to "play" and checks all three — a console
speaker, which finishes instantly, cannot tell a correct implementation from a
broken one.

---

## Speaking

```bash
make voices                                       # download a voice (~60 MB)
make speak TEXT="Yes? I'm listening."              # test it with no microphone
python -m diana.main --speak "the kettle has boiled"
```

`--speak` is the fastest way to answer "is the voice installed and does it
sound right?", because it touches no microphone, no pid file and no state
machine. If `--speak` works and `make run` does not, the problem is the voice
loop rather than the voice.

`tts_engine` is `auto` by default, which means: speak if a voice is present,
print if not. A fresh checkout with no voice behaves exactly as it did before
TTS existed, and the moment you run `make voices` she starts talking without
you editing any configuration.

**Why piper.** One ONNX file per voice, CPU-only, no account, no key, no
network at runtime — the same properties as everything else here. On this
laptop it synthesises about *ten times faster than real time* (a 2.3 s reply
generated in 0.20 s, RTF 0.09), which is the only reason a neural voice is
viable inside a voice loop at all. Any other engine works the same way: write a
`Speaker` subclass and change one line in `main.py`.

Two implementation details that are easy to get wrong, both handled in
`PiperSpeaker`:

- **Headroom.** Piper normalises every utterance so its loudest sample is
  exactly 1.0, which is precisely where a sound card starts clipping. Replies
  are scaled to `tts_peak` (0.85) instead.
- **Knowing when to stop waiting.** `wait()` cannot just join the synthesis
  thread: that thread finishing only means the audio was *generated*, not that
  it was *played*. Every chunk is counted into the queue and every sample is
  counted out of the PortAudio callback, and `wait()` returns when the two
  agree. Return early and you are back to the loop at the top of this page.

Leave `tts_output_device` unset unless you have a reason not to. A Piper voice
outputs at 22.05 kHz, and on Linux the system default (PulseAudio/PipeWire)
resamples that to your sound card, while a raw ALSA device pinned by index
almost certainly will not and simply refuses to open. `make devices` prints the
rate next to each output so you can see which is which.

---

## Understanding what you said

Two modules sit between the recogniser and the speaker.

### `intents.py` — the grammar

A list of rules over tokens. "turn on the lights" becomes
`Match(code="LIGHTS_ON", target="light", action="on")`, which the device layer
acts on and the speaker turns into a sentence. It is not a neural model: it
runs on the audio thread in microseconds, downloads nothing, and — the reason
it is worth having first — **it fails in a way you can read**. Every match logs
the rule that fired, so a wrong answer names the rule that was wrong instead
of leaving you with a black box.

```bash
make text TEXT="turn on the lights"
# COMMAND: turn on the lights
# EVENT: LIGHTS_ON
# Diana: lights are now on.
# status: ok
```

`--text` uses the same handler the microphone does and touches nothing else, so
you can check forty phrasings in a second rather than saying them out loud to a
running assistant. That is the intended way to extend the grammar. Pass `-` to
read a list, one utterance per line:

```bash
printf 'turn on the lights\nplay music\nare the lights on\n' | make text TEXT=-
```

State persists across those lines, so this is also how you check that a
question answers from what came before it.

Some things worth knowing before you change it:

- **Devices own their own vocabulary.** "lamp" resolving to the lights is
  `names=["light", "lights", "lamp", ...]` on a `Device` in `devices.py`, not a
  word in a list inside the parser. Adding a device is a data change.
- **Filler words are removed before matching, but the phrase tables are matched
  against the unstripped words.** Both token lists exist for a reason: stripping
  "is" out of "how hot is it" deletes the weather rule's only trigger, and it
  fails silently, which is the worst way to fail.
- **The registry is the authority on capability.** A rule that found a device
  still has to ask whether it can do what was asked. That is what stops "turn
  off the weather" from being carried out by a parser that matched a word.
- **A group is a device that expands.** "Turn everything off" is
  `Device(id="all", expands_to=[...])`, not a special case in the parser.
- **State is remembered, which is the point.** "Are the lights on?" is only
  answerable because something stored the last command. Toggle cannot know what
  it will say until it has looked.

### `chat.py` — everything else

The grammar is a closed world, so most real sentences end up matching nothing.
Answering those with "I don't know how to do that yet" is correct and is also
the last thing you want to hear when you were not giving an order. So when no
rule fires, the utterance goes to a small local model instead.

```bash
ollama serve
ollama pull qwen2.5:0.5b
make chat-test
```

```
chat: ollama[qwen2.5:0.5b]
  -> Paris is the capital of France.
  -> Why don't scientists trust atoms? Because they make up everything.
```

**The model answers, and only answers.** It is consulted exclusively after the
grammar has already failed, it has no registry, no event code and no transport,
and it cannot reach the house by any path. A command is handled by the rule
table or not at all. A 0.5b model asked to act on your home is a liability; a
0.5b model asked what the capital of France is is fine, and that asymmetry is
why the two live in separate modules.

It also has to be a *spoken* answer, which is a different problem from a good
one. Piper reads at about three words a second and you will have stopped
listening well before that, so the system prompt, the token limit and
`shorten()` all push toward one short sentence, and a long answer is not a
thorough answer — it is a reply that gets cut off by the next wake word.

Two failure modes are reported differently on purpose. Chat switched off gives
"I don't know how to do that yet", because that is simply true of a rule table.
Chat enabled but silent gives "I didn't catch an answer to that", because the
fault is in Ollama and not in the grammar, and the difference is an afternoon
of debugging saved.

Set `chat_enabled: false` and everything else keeps working with no network and
no server of any kind. That property is worth being able to test, so the self
test never requires Ollama to be running.

---

## Tuning

Run `make run-debug` to see state transitions and the levels the energy gate is
comparing against. The settings you are most likely to touch, all in
`config.py` and overridable in `config.json` or via `DIANA_*` environment
variables:

| Setting | Default | What it does |
|---|---|---|
| `idle_silence_rms` | `0.008` | Below this, blocks are skipped while idle. Raise if she wakes on background noise; lower if she ignores you. |
| `wake_phrases` | 3 phrases | Keep it short. Every extra phrase enlarges the search space. |
| `wake_cooldown_s` | `1.5` | Ignore the wake word for this long after one fires, and again after she stops talking. Stops her re-triggering on her own tail. |
| `post_wake_discard_ms` | `180` | Audio to throw away after the wake word, because your mouth is already forming the command. |
| `post_speak_discard_ms` | `300` | Audio to throw away after she stops talking, so her own `Yes?` is not transcribed as your command. |
| `command_silence_s` | `1.6` | Silence that ends a command. Must survive a mid-sentence pause. |
| `command_max_s` | `12.0` | Hard cap on one command. |
| `show_partials` | `true` | Print the transcript as you speak, so you can watch recognition converge. |
| `command_handler` | `intents` | `intents` runs the grammar; `print` is the original echo-only V0 handler. |
| `command_reply` | `intent` | `intent` says what happened, `echo` repeats the transcript (best for spotting a mishearing), `ack` is a short "Okay.", `none` is silent. |
| `command_show_events` | `true` | Print the `EVENT: LIGHTS_ON` line. The one unambiguous record of a decision — leave it on. |
| `chat_enabled` | `true` | Off means no model is contacted and unknown input gets the "don't know how" reply. |
| `chat_model` | `qwen2.5:0.5b` | `1.5b` converses better and waits noticeably longer. The reply is spoken, so speed wins. |
| `chat_timeout_s` | `20.0` | Short on purpose. Waiting longer has failed more completely than admitting it. |
| `tts_voice` | `en_US-amy-medium` | Which voice to speak with. `make voices` fetches it. |
| `tts_length_scale` | `1.0` | Speaking rate. 1.0 is natural, 1.2 is faster, 0.8 slower. |
| `tts_peak` | `0.85` | Headroom. Piper normalises to 1.0, which is where a sound card clips. |
| `tts_echo` | `true` | Also print what she says, so the terminal stays readable. |

Three layers, lowest priority first: dataclass defaults → `config.json` →
`DIANA_*` environment variables.

```bash
DIANA_LOG_LEVEL=DEBUG DIANA_COMMAND_MAX_S=20 make run
```

An unknown key in `config.json` is a hard error, not a silent no-op. A typo'd
setting is otherwise invisible.

---

## Dependencies, and why these

- **vosk** — offline speech recognition. Small CPU-only models, no account, no
  key. The small English model is 40 MB and genuinely good enough for a
  command in a quiet room.
- **piper-tts** — offline neural text-to-speech, one 60 MB ONNX file per voice.
- **sounddevice** — a thin, pleasant wrapper over PortAudio. Now used for
  input *and* output.
- **numpy** — audio buffers; also what every model framework expects.
- **ollama** (optional, not a pip package) — serves the small local model behind
  `chat.py`. Nothing in `requirements.txt`; it is a separate install and it is
  only contacted when `chat_enabled` is true. The client is stdlib `urllib`, so
  there is no HTTP dependency to add.

On Linux you may also need PortAudio itself:

```bash
sudo apt install libportaudio2      # Debian/Ubuntu
sudo dnf install portaudio          # Fedora
sudo pacman -S portaudio            # Arch
```

---

## What is deliberately not here

- **No home automation.** `devices.py` is a dataclass and an in-memory list.
  `send()` builds the message a node *would* receive, logs it and appends it to
  an `outbox` you can read back; it does not transmit. That is the seam: put a
  transport in `send()` and nothing above it moves. The two decisions worth
  making early are that nodes should receive *structured* messages
  (`{"intent": ..., "device": ..., "action": ...}`), not English, and that the
  registry should own state rather than the nodes.
- **No Transformer.** `CommandHandler.handle()` receives a `Command` and
  returns a `CommandResult` whose `data` field is a dict. `IntentCommandHandler`
  fills that dict from a rule table, and your model will fill it from weights.
  Nothing else changes — see the wiring diagram above.
- **No weather.** It is understood and there is no data source behind it, so
  she says so rather than inventing a forecast. That is a one-line change in
  `_rule_weather`'s handler when you have an API key.
- **No barge-in.** Calling `speak()` while she is already talking abandons the
  current line and says the new one instead, but nothing yet *listens* for an
  interrupt while she is speaking — the microphone is closed until she is
  finished. That is the safe default; the interesting version stops the audio
  the moment you say "stop", which needs a real-time mixer rather than a
  blocking `wait()`.

---

## Command reference

```
python -m diana.main                    run in the foreground
python -m diana.main --daemon           detach into the background
python -m diana.main --status           is a daemon running?
python -m diana.main --stop             stop it cleanly
python -m diana.main --selftest         test the loop, no hardware needed
python -m diana.main --list-devices     list microphones and speakers
python -m diana.main --speak "hello"    say one line, no microphone (use - for stdin)
python -m diana.main --text "turn on the lights"
                                       run a phrase through the command stack, no mic
python -m diana.main --config my.json   use a specific config file
python -m diana.main --max-commands 3   exit after 3 commands (testing)

python download_models.py               fetch the speech model
python download_models.py --large       the 1.8 GB model, more accurate
python download_models.py --voices      fetch a TTS voice (--voices en_GB-alan-medium)
python download_models.py --verify      check what is installed
```

---

## Where to go next

1. **Replace the command handler.** Load your model once in `__init__`, return
   a `CommandResult` with a structured `data` dict. Keep `handle()` fast — it
   runs on the audio timeline.
2. **Replace the recogniser.** Implement `SpeechToText` against your own
   model. The streaming `feed()`/`finish()` shape is deliberate: it is what a
   real-time assistant needs and what an incremental decoder provides.
3. **Then** the device layer, and the transport to the nodes.
4. **Then** barge-in, which is the one piece of the voice loop that is still
   solved by blocking rather than by design.

One thing to decide before step 3: have your Transformer emit a *structured
intent*, not a sentence. Then what Diana can do is defined by the protocol
rather than by however many English phrasings you happened to train on.
