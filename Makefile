PY      := .venv/bin/python
PIP     := .venv/bin/pip
PYTEST  := .venv/bin/pytest

.PHONY: all venv goldens test run clean

all: test

# Python virtualenv with pinned dependencies
venv: .venv/ok
.venv/ok: requirements-dev.txt
	python3 -m venv .venv
	$(PIP) install -q -r requirements-dev.txt
	touch $@

# Run the GUI application
run: venv
	$(PY) test/imageshow.py

# Regenerate the k-NN golden reference files (destructive -- only run
# deliberately; goldens are committed to git). The R-transform goldens are
# frozen: they were captured from the original C++ implementation, which was
# removed in Phase 2 (retrievable via git history, see REFACTORING.md).
goldens: venv
	$(PY) tests/capture_goldens.py

test: venv
	$(PYTEST) tests/ -v

clean:
	rm -rf .venv tests/__pycache__ falldetect/__pycache__ .pytest_cache
