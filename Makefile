.PHONY: install test run demo summary backtest

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

run:
	python run.py

demo:
	python run.py --demo

summary:
	python scripts/summary.py

backtest:
	python scripts/backtest_chart.py
