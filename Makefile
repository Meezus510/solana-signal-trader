.PHONY: install test run demo

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

run:
	python run.py

demo:
	python run.py --demo
