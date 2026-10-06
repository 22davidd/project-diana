# Project Diana -- V0 voice loop
#
# Everything runs through the venv in .venv, so you never have to remember to
# activate it. `make help` lists the targets.

PY      := .venv/bin/python
PIP     := .venv/bin/pip

.DEFAULT_GOAL := help
.PHONY: help venv install models voices models-verify run run-debug selftest devices \
        speak text chat-test daemon start stop status log clean distclean

help:  ## Show this help
	@echo "Project Diana"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Typical first run:"
	@echo "  make install && make models && make voices && make run"

# ---------------------------------------------------------------- setup ----
venv:  ## Create the virtualenv
	python3 -m venv .venv

install: venv  ## Install Python dependencies
	$(PIP) install -r requirements.txt

models:  ## Download the speech model (~40 MB, once)
	$(PY) download_models.py

voices:  ## Download a text-to-speech voice (~60 MB, once)
	$(PY) download_models.py --voices

models-verify:  ## Check which models and voices are installed
	$(PY) download_models.py --verify

# ------------------------------------------------------------------ run ----
run:  ## Run Diana in the foreground
	$(PY) -m diana.main

run-debug:  ## Run with DEBUG logging (shows state transitions and levels)
	DIANA_LOG_LEVEL=DEBUG $(PY) -m diana.main

selftest:  ## Test the loop with no microphone and no model
	$(PY) -m diana.main --selftest

devices:  ## List audio input and output devices
	$(PY) -m diana.main --list-devices

speak:  ## Say a line out loud without a mic:  make speak TEXT="Yes?"
	$(PY) -m diana.main --speak "$(TEXT)"

text:  ## Run a phrase through the command stack, no mic:  make text TEXT="turn on the lights"
	$(PY) -m diana.main --text "$(TEXT)"

chat-test:  ## Check the fallback model answers, no mic:  make chat-test
	@$(PY) -c "from diana.config import Config; from diana.chat import build_chat; \
	c = build_chat(Config()); print('chat:', c.describe()); \
	[print('  ->', c.reply(q) or '(no answer)') for q in \
	 ['what is the capital of france', 'tell me a joke', 'how do i make lasagne']]; c.close()"

# --------------------------------------------------------------- daemon ----
daemon: start  ## Alias for `make start`
	@:

start:  ## Start Diana in the background
	$(PY) -m diana.main --daemon
	@sleep 1
	@$(PY) -m diana.main --status

stop:  ## Stop the background Diana
	$(PY) -m diana.main --stop

status:  ## Is Diana running?
	$(PY) -m diana.main --status

log:  ## Follow the daemon log
	@touch var/diana.log
	tail -f var/diana.log

# ---------------------------------------------------------------- clean ----
clean:  ## Remove runtime state (pid, log) but keep the model
	rm -rf var

distclean: clean  ## Also remove the venv and the model
	rm -rf .venv models
