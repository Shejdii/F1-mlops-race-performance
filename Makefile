PYTHON := python

.PHONY: install features train predict all test format lint check clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

features:
	$(PYTHON) -m src.cli features

train:
	$(PYTHON) -m src.cli train

predict:
	$(PYTHON) -m src.cli predict

all:
	$(PYTHON) -m src.cli all

test:
	$(PYTHON) -m pytest -q

format:
	$(PYTHON) -m black .

lint:
	$(PYTHON) -m pylint src

check:
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test

clean:
	rm -rf artifacts/*
	rm -rf __pycache__