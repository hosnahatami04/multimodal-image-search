# ---------------------------------------------------------------------------
# multimodal-image-search -- developer entry points
#
# Every long-running or multi-step command in this project has a name here, so
# that "how do I run X" never has to be answered from memory or shell history.
# ---------------------------------------------------------------------------
PYTHON ?= python

.PHONY: help install fetch download stats test lint format clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install all Python dependencies
	$(PYTHON) -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r requirements.txt

fetch:  ## Download the parquet files resumably (use when the Hub client stalls)
	$(PYTHON) scripts/fetch_parquet.py

download:  ## Download Flickr8k into data/raw and verify integrity
	$(PYTHON) -m src.data.download

stats:  ## Print dataset statistics (counts, caption lengths)
	$(PYTHON) -m src.data.dataset --stats

test:  ## Run the test suite
	$(PYTHON) -m pytest -v

lint:  ## Check style and lint rules
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

format:  ## Auto-format the codebase
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

clean:  ## Remove caches (does NOT delete downloaded data)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
