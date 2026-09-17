# ---------------------------------------------------------------------------
# multimodal-image-search -- developer entry points
#
# Every long-running or multi-step command in this project has a name here, so
# that "how do I run X" never has to be answered from memory or shell history.
# ---------------------------------------------------------------------------
PYTHON ?= python

.PHONY: help install fetch fetch-blip download stats agreement check-clip encode space search smoke queries eval hard-negatives latency vqa-questions vqa failures plots report evaluate serve check docker-build docker-run test test-all lint format clean-index clean

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

agreement:  ## Measure inter-annotator caption agreement and save results
	$(PYTHON) -m src.data.agreement --save

check-clip:  ## Sanity check: an image should sit closer to its own captions
	$(PYTHON) -m src.embedding.clip_encoder --check

encode:  ## Encode all images, cache the vectors, build the index (slow, once)
	$(PYTHON) -m src.embedding.build_index

search:  ## Example text query -- make search Q="a dog jumping over a fence"
	$(PYTHON) -m src.search.text_to_image $(Q)

space:  ## Measure the embedding space geometry and save results
	$(PYTHON) -m src.embedding.stats --save

smoke:  ## Run the manual smoke test over a fixed set of queries
	$(PYTHON) -m src.search.smoke

queries:  ## Write the retrieval query set
	$(PYTHON) -m src.eval.build_queries

vqa-questions:  ## Write the 100-question VQA evaluation set
	$(PYTHON) -m src.vqa.build_questions

vqa:  ## Answer the VQA question set and save results
	$(PYTHON) -m src.eval.run_vqa --save
	$(PYTHON) -m src.eval.vqa_alternatives --save

fetch-blip:  ## Resumably fetch the BLIP weights (use when the Hub client stalls)
	$(PYTHON) scripts/fetch_model.py --model Salesforce/blip-vqa-base

eval:  ## Score the query set and save results
	$(PYTHON) -m src.eval.run_retrieval --save
	$(PYTHON) -m src.eval.false_negatives --save

hard-negatives:  ## Build and score the hard-negative groups
	$(PYTHON) -m src.eval.hard_negatives --build --save

latency:  ## Benchmark query latency on CPU
	$(PYTHON) -m src.eval.latency --save

failures:  ## Analyse the failure patterns and assemble the worked cases
	$(PYTHON) -m src.eval.retrieval_failures --save
	$(PYTHON) -m src.eval.failure_cases --save

plots:  ## Render the evaluation figures
	$(PYTHON) -m src.eval.plots

report:  ## Regenerate results/report.md from the committed result files
	$(PYTHON) -m src.eval.report

evaluate: queries eval hard-negatives latency vqa-questions vqa failures plots report  ## Run the whole evaluation

serve:  ## Run the API locally -- http://127.0.0.1:8000/docs
	$(PYTHON) -m uvicorn src.api:app --reload

check:  ## Verify the committed metrics have not regressed (what CI gates on)
	$(PYTHON) scripts/check_metrics.py

docker-build:  ## Build the container image (needs data/embeddings and data/chroma)
	docker build -t multimodal-search .

docker-run:  ## Run the container -- http://localhost:8000/docs
	docker run --rm -p 8000:8000 multimodal-search

test:  ## Run the tests that need no model weights or dataset (what CI runs)
	$(PYTHON) -m pytest -q -m "not model and not data"

test-all:  ## Run every test, including those needing CLIP and the dataset
	$(PYTHON) -m pytest -v

lint:  ## Check style and lint rules
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

format:  ## Auto-format the codebase
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

clean-index:  ## Delete the embedding cache and the Chroma index
	rm -rf data/embeddings data/chroma

clean:  ## Remove caches (does NOT delete downloaded data)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
