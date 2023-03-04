BLACK = black
RUFF = ruff

.PHONY: format
format:
	$(BLACK) .

.PHONY: lint
lint:
	$(RUFF) check . --fix --exit-non-zero-on-fix

