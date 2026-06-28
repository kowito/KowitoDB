# KowitoDB developer commands. Run `make` or `make help` for the list.
# (Targets are thin wrappers around cargo — no install required.)

.DEFAULT_GOAL := help
.PHONY: help build release run ask test lint fmt fmt-check ci example \
        bench bench-compare docker docker-run gen-python clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the whole workspace (debug)
	cargo build --workspace

release: ## Build the optimized release binary
	cargo build --release -p kowitodb

run: ## Run the gRPC server in dev mode (127.0.0.1:50051)
	cargo run -p kowitodb -- serve

ask: ## Ask a question in embedded mode: make ask Q="your question"
	cargo run -p kowitodb -- ask "$(Q)"

test: ## Run the full test suite
	cargo test --workspace

lint: ## Clippy with warnings-as-errors (matches CI)
	cargo clippy --workspace --all-targets -- -D warnings

fmt: ## Format the codebase
	cargo fmt --all

fmt-check: ## Check formatting (matches CI)
	cargo fmt --all --check

ci: fmt-check lint test ## Run the exact CI gate locally (fmt + clippy + test)

example: ## Run the embedded-library example (no server)
	cargo run -p kowitodb-server --example embedded

bench: ## HNSW recall/latency/QPS benchmark (release)
	cargo run --release -p kowitodb-index --example bench_hnsw

bench-compare: ## Generate the shared dataset + KowitoDB ANN numbers (release)
	cargo run --release -p kowitodb-index --example bench_compare -- dataset.bin

docker: ## Build the server Docker image (tag: kowitodb)
	docker build -t kowitodb .

docker-run: ## Run the server image, persisting to ./data
	docker run --rm -p 50051:50051 -p 9090:9090 -v "$$PWD/data:/data" kowitodb

gen-python: ## Regenerate the Python SDK gRPC stubs from proto/
	bash sdk/python/scripts/gen.sh

clean: ## Remove build artifacts
	cargo clean
