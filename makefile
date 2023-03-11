BLACK = black
RUFF = ruff
POETRY = poetry
PIP = pip

.PHONY: format
format:
	$(BLACK) .

.PHONY: lint
lint:
	$(RUFF) check . --fix --exit-non-zero-on-fix

.PHONY: configure-poetry
configure-poetry:
	$(PIP) install poetry
	$(POETRY) config virtualenvs.in-project true
