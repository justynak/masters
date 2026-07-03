PY      := .venv/bin/python
PIP     := .venv/bin/pip
PYTEST  := .venv/bin/pytest
CXX     := g++
CXXFLAGS := -O2 -Wall

HARNESS := bin/rtransform_dump

.PHONY: all venv harness ext goldens test run clean

all: harness test

# Python virtualenv with pinned dependencies
venv: .venv/ok
.venv/ok: requirements-dev.txt
	python3 -m venv .venv
	$(PIP) install -q -r requirements-dev.txt
	touch $@

# Standalone C++ harness around the original R-transform
harness: $(HARNESS)
$(HARNESS): tools/rtransform_dump.cpp test/RTransform.cpp test/RTransform.h
	mkdir -p bin
	$(CXX) $(CXXFLAGS) tools/rtransform_dump.cpp test/RTransform.cpp -o $@

# Cython extension (rTransform) built in-place for the venv's Python
ext: venv
	cd test && ../$(PY) setup.py build_ext --inplace

# Run the GUI application
run: ext
	$(PY) test/imageshow.py

# Regenerate golden reference files (destructive -- only run deliberately,
# e.g. before starting the port; goldens are committed to git)
goldens: harness venv
	$(PY) tests/capture_goldens.py

test: harness venv ext
	$(PYTEST) tests/ -v

clean:
	rm -rf bin .venv tests/__pycache__ .pytest_cache \
	    test/rTransform.cpp test/*.so test/build/lib.* test/build/temp.*
