#!/bin/sh
set -e

flake8 --config=.flake8 .
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics

black --check src

mypy src --ignore-missing-imports

coverage run -m pytest -r src --omit="*/bazel*"

coverage report --ignore-errors --fail-under 95 -m

# coverage report --ignore-errors --fail-under 100 -m --include="src/*"

# Print changes.
git diff
# Already well formated if 'git diff' doesn't output anything.
! ( git diff |  grep -q ^ ) || exit 1
