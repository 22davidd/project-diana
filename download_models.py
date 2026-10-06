#!/usr/bin/env python3
"""
download_models.py -- fetch the models Diana needs.

Two kinds of model live in ./models, and they are fetched differently:

  * the Vosk speech model, used for both the wake word and speech-to-text
  * a Piper voice, used for text-to-speech

Vosk models are plain directories of Kaldi weights, not pip packages, so they
have to be downloaded separately. Piper voices are single .onnx files, fetched
through the `piper-tts` package you installed with requirements.txt.

Which speech model
------------------
vosk-model-small-en-us-0.15   ~40 MB   <- default
vosk-model-en-us-0.22        ~1.8 GB  <- larger, more accurate

Start with the small one. It is genuinely good enough to recognise a command in
a quiet room, and it runs on a laptop CPU without a fan spinning up. If you
find it mishearing things, try the large one -- but check the CPU usage, because
a voice assistant that saturates a core is no use.

Which voice
-----------
en_US-amy-medium   ~63 MB   <- default
Any other voice from https://huggingface.co/rhasspy/piper-voices works;
`--voices` with no name downloads the default, `--voices en_GB-alan-medium`
downloads whichever you name. A voice is optional: without one Diana prints
instead of speaking, which is the V0 behaviour.

Usage:
    python download_models.py                      # the default speech model
    python download_models.py --large              # the 1.8 GB speech model
    python download_models.py --voices             # the default voice (~63 MB)
    python download_models.py --voices en_GB-alan-medium
    python download_models.py --verify             # just check what is present
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://alphacephei.com/vosk/models"

MODELS = {
    "small": "vosk-model-small-en-us-0.15",
    "large": "vosk-model-en-us-0.22",
}

# Must match Config.tts_voice, so `make voices` gets the voice Diana will
# actually use. Duplicated rather than imported because this script has to run
# before the package is importable in any way that matters.
DEFAULT_VOICE = "en_US-amy-medium"

# Marker files that must exist inside a valid model directory. Checking these
# is how we tell a complete download from a truncated one -- a half-downloaded
# model loads without error and then produces garbage, which is a miserable
# thing to debug.
REQUIRED_FILES = ("am/final.mdl", "conf/mfcc.conf")

# A Piper voice is one .onnx plus its .onnx.json sidecar. The config file is
# what carries the sample rate and the phoneme id mapping, so a voice without
# it is useless rather than merely quiet.
VOICE_SUFFIX = ".onnx"
VOICE_CONFIG_SUFFIX = ".json"


def _human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024 or unit == "GB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} GB"


def is_valid(path: Path) -> bool:
    """True if `path` looks like a complete Vosk model."""
    return path.is_dir() and all((path / rel).is_file() for rel in REQUIRED_FILES)


def voice_files(models_dir: Path) -> list[Path]:
    """Every complete Piper voice in models_dir, sorted by name."""
    if not models_dir.is_dir():
        return []
    return sorted(
        p
        for p in models_dir.glob(f"*{VOICE_SUFFIX}")
        if (p.parent / f"{p.name}{VOICE_CONFIG_SUFFIX}").is_file()
    )


def verify(models_dir: Path) -> int:
    """Report which models are already present. Returns an exit code."""
    print(f"Looking in {models_dir}")
    print("\nSpeech recognition (vosk)")
    found = False
    for size, name in MODELS.items():
        path = models_dir / name
        if is_valid(path):
            found = True
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            print(f"  [ok]      {name}  ({_human(total)})")
        elif path.exists():
            print(f"  [BROKEN]  {name}  -- incomplete, delete it and re-run")
        else:
            print(f"  [missing] {name}")

    print("\nVoice (piper)")
    voices = voice_files(models_dir)
    for path in voices:
        total = path.stat().st_size + (models_dir / f"{path.name}{VOICE_CONFIG_SUFFIX}").stat().st_size
        print(f"  [ok]      {path.stem}  ({_human(total)})")
    if not voices:
        print(f"  [missing] {DEFAULT_VOICE}  -- Diana will print instead of speak")
        print(f"            Fetch it with:  python download_models.py --voices")

    if not found:
        print("\nNo speech model installed. Run:  python download_models.py")
        return 1
    return 0


def download(models_dir: Path, size: str) -> int:
    name = MODELS[size]
    target = models_dir / name

    if is_valid(target):
        print(f"{name} is already installed at {target}")
        print("Delete that directory first if you want a fresh copy.")
        return 0

    if target.exists():
        print(f"Removing incomplete download at {target}")
        shutil.rmtree(target)

    url = f"{BASE_URL}/{name}.zip"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {name}")
    print(f"  from {url}")
    print("  This is a few tens of MB and only needs doing once.\n")

    # Download to a temporary file and unzip afterwards. Writing straight into
    # models/ means an interrupted download leaves a broken directory that
    # looks valid -- exactly what the is_valid() check above is guarding
    # against, so let's not create the problem in the first place.
    with tempfile.TemporaryDirectory(dir=models_dir) as tmpdir:
        archive = Path(tmpdir) / f"{name}.zip"
        try:
            _fetch(url, archive)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"\nDownload failed: {exc}")
            print("Check your internet connection and try again.")
            return 1

        print(f"  downloaded {_human(archive.stat().st_size)}, extracting...")
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmpdir)
        except zipfile.BadZipFile:
            print("\nThe archive is corrupt. Re-run to try again.")
            return 1

        extracted = Path(tmpdir) / name
        if not is_valid(extracted):
            print(f"\nExtracted archive is missing {REQUIRED_FILES}; treating as a bad download.")
            return 1

        # Atomic-ish: move it into place only once it is known good.
        shutil.move(str(extracted), str(target))

    total = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    print(f"\nInstalled {name} ({_human(total)}) to {target}")
    print("\nReady. Start Diana with:\n\n    python -m diana.main\n")
    return 0


def download_voices(models_dir: Path, names: list[str]) -> int:
    """Fetch one or more Piper voices.

    Piper's own downloader is used rather than a hand-rolled one, because it
    already knows the file naming, the sidecar config, and which voices exist.
    Its list of voices is public and changes, so a name that is not there gets
    a clear error rather than a 404 halfway through a 60 MB download.
    """
    try:
        from piper.download_voices import download_voice
    except ImportError:
        print(
            "The 'piper-tts' package is missing, which is what fetches voices.\n"
            "Install it with:\n\n    pip install piper-tts\n"
        )
        return 1

    models_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = models_dir / f"{name}{VOICE_SUFFIX}"
        if target.is_file() and (models_dir / f"{target.name}{VOICE_CONFIG_SUFFIX}").is_file():
            print(f"{name} is already installed at {target}")
            continue
        print(f"Downloading voice {name} (~63 MB, one time only)")
        try:
            download_voice(name, models_dir)
        except Exception as exc:
            print(f"\nFailed to download {name}: {exc}")
            print("Check the voice name at https://huggingface.co/rhasspy/piper-voices")
            return 1

    print("\nInstalled. Try it without a microphone:")
    print(f"\n    python -m diana.main --speak \"Yes? I'm listening.\"\n")
    return 0


def _fetch(url: str, dest: Path) -> None:
    """Stream a URL to disk, printing progress as it goes."""
    with urllib.request.urlopen(url, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        last_report = 0.0

        with open(dest, "wb") as out:
            while chunk := response.read(64 * 1024):
                out.write(chunk)
                downloaded += len(chunk)
                # Report at most a few times a second; a progress line on
                # every 64 KB block floods the terminal and slows the download.
                now = _now()
                if now - last_report > 0.2:
                    if total:
                        pct = 100 * downloaded / total
                        print(f"\r  {pct:5.1f}%  {_human(downloaded)} / {_human(total)}", end="")
                    else:
                        print(f"\r  {_human(downloaded)}", end="")
                    last_report = now
    print(f"\r  100.0%  {_human(downloaded)}                    ")


def _now() -> float:
    import time

    return time.monotonic()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the models Diana needs.")
    parser.add_argument("--large", action="store_true", help="use the 1.8 GB speech model instead of the small one")
    parser.add_argument(
        "--voices", nargs="*", metavar="NAME",
        help=(
            "download a Piper voice for text-to-speech instead of a speech model "
            f"(default: {DEFAULT_VOICE}). No name downloads the default."
        ),
    )
    parser.add_argument("--verify", action="store_true", help="check what is installed and exit")
    parser.add_argument(
        "--dir",
        default=str(Path(__file__).resolve().parent / "models"),
        help="where to install models (default: ./models)",
    )
    args = parser.parse_args(argv)

    models_dir = Path(args.dir)
    if args.verify:
        return verify(models_dir)
    if args.voices is not None:
        return download_voices(models_dir, args.voices or [DEFAULT_VOICE])
    return download(models_dir, "large" if args.large else "small")


if __name__ == "__main__":
    sys.exit(main())
