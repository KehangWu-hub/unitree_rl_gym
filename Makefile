PYTHON ?= python

.PHONY: help test check clean

help:
	@echo 'make test                 Run offline tests (DDS disabled)'
	@echo 'make check                Check patch whitespace'
	@echo 'make clean                Remove Python caches from source directories'
	@echo 'Override interpreter with PYTHON=/path/to/python'

test:
	PYTHONDONTWRITEBYTECODE=1 GO2_DDS_TESTS=0 PYTHONPATH=. $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

check:
	git diff --check

clean:
	find deploy legged_gym tests third_party -type d -name __pycache__ -prune -exec rm -rf {} +
