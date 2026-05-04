.PHONY: clean clean-test clean-pyc clean-build docs help lint test test-all
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys
from urllib.request import pathname2url

webbrowser.open(sys.argv[1])
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-docs: ## remove docs artifacts
	rm -f docs/apidoc/xclim*.rst
	rm -f docs/apidoc/modules.rst
	rm -f docs/notebooks/data/*.nc
	$(MAKE) -C docs clean

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

install-lint: ## install dependencies needed for linting
	python -m pip install --quiet --group lint

install-docs: ## install dependencies needed for building the docs
	python -m pip install --quiet --group docs

install-test: ## install dependencies needed for standard testing
	python -m pip install --quiet --group test

install-test-notebooks: ## install dependencies needed for doctest testing
	python -m pip install --quiet --group test-notebooks

lint: install-lint ## check style with flake8 and black
	python -m ruff check --quiet .
	python -m flake8 --config=.flake8 src/xclim tests
	python -m vulture src/xclim tests
	codespell src/xclim tests docs
	python -m numpydoc lint src/xclim/*.py src/xclim/ensembles/*.py src/xclim/indices/*.py src/xclim/indicators/*.py src/xclim/testing/*.py
	python -m deptry src
	python -m yamllint --config-file=.yamllint.yaml src/xclim

test: install-test ## run tests quickly with the default Python
	python -m pytest --numprocesses=auto

test-doctests: install-test ## run doctests
	python -m pytest --rootdir=tests/ --numprocesses=auto --xdoctest src/xclim

test-notebooks: install-test-notebooks
	python -m pytest --no-cov --nbval --numprocesses=auto --dist=loadscope --rootdir=tests/ docs/notebooks --ignore=docs/notebooks/example.ipynb

test-all: test test-doctests test-notebooks ## run all tests

coverage: ## check code coverage quickly with the default Python
	python -m coverage run --source xclim -m pytest src/xclim
	python -m coverage report -m
	python -m coverage html
	$(BROWSER) htmlcov/index.html

autodoc-obsolete: install-docs clean-docs ## create sphinx-apidoc files (obsolete)
	mkdir -p docs/apidoc/
	sphinx-apidoc -o docs/apidoc/ --private --module-first src/xclim

autodoc-custom-index: install-docs clean-docs ## create sphinx-apidoc files but with special index handling for indices and indicators
	mkdir -p docs/apidoc/
	sphinx-apidoc -o docs/apidoc/ --private --module-first src/xclim src/xclim/indicators src/xclim/indices
	rm docs/apidoc/xclim.rst
	env SPHINX_APIDOC_OPTIONS="members,undoc-members,show-inheritance,noindex" sphinx-apidoc -o docs/apidoc/ --private --module-first src/xclim

linkcheck: autodoc-custom-index ## run checks over all external links found throughout the documentation
	$(MAKE) -C docs linkcheck

build-docs: autodoc-custom-index ## generate Sphinx HTML documentation, including API docs, but without indexes for for indices and indicators
	$(MAKE) -C docs html

docs: build-docs  ## open the built documentation in a web browser
	## Start http server and show in browser.
	## We want to have the cli command run in the foreground, so it's easy to kill.
	## And we wait 2 sec for the server to start before opening the browser.
	{ sleep 2; $(BROWSER) "http://localhost:54345"; } &
	python -m http.server 54345 --directory docs/_build/html/

servedocs: autodoc-custom-index ## generate Sphinx HTML documentation, including API docs, but without indexes for for indices and indicators, and watch for changes
	$(MAKE) -C docs livehtml

release: dist ## package and upload a release
	python -m flit publish dist/*

dist: clean ## builds source and wheel package
	python -m flit build
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python -m pip install --no-user .

development: clean ## install the package and development dependencies in editable mode to the active Python's site-packages
	python -m pip install --group dev
	python -m pip install --no-user --editable ".[complete]"

upstream: clean develop ## install the GitHub-based development branches of dependencies in editable mode to the active Python's site-packages
	python -m pip install --no-user --requirement requirements_upstream.txt
