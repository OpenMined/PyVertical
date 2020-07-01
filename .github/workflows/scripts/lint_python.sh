#!/bin/sh
set -e

flake8 --config=.flake8 .
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics

black --check src

mypy src --ignore-missing-imports
