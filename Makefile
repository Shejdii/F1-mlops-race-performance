PYTHON := python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

features:
	$(PYTHON) -m src.cli features

train:
	$(PYTHON) -m src.cli train

predict:
	$(PYTHON) -m src.cli predict

season:
	python -m src.cli season-analysis

all:
	$(PYTHON) -m src.cli all

test:
	$(PYTHON) -m pytest -q

format:
	$(PYTHON) -m black .
	