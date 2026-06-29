# Makefile for the KowitoDB Go SDK.

# Path to the proto file (relative to this directory).
PROTO_DIR := ../../kowitodb-server/proto
PROTO     := kowitodb.proto
PB_DIR    := kowitodbpb
GO_MODULE := github.com/kowito/kowitodb/sdk/go
GO_PKG    := $(GO_MODULE)/$(PB_DIR)

.PHONY: all generate tools build test fmt vet tidy clean

all: generate build

## tools: install the protoc Go plugins.
tools:
	go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.34.2
	go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.5.1

## generate: regenerate the protobuf + gRPC Go code from the proto.
generate:
	mkdir -p $(PB_DIR)
	protoc \
		--proto_path=$(PROTO_DIR) \
		--go_out=$(PB_DIR) \
		--go_opt=paths=source_relative \
		--go_opt=M$(PROTO)=$(GO_PKG) \
		--go-grpc_out=$(PB_DIR) \
		--go-grpc_opt=paths=source_relative \
		--go-grpc_opt=M$(PROTO)=$(GO_PKG) \
		$(PROTO_DIR)/$(PROTO)

## build: compile the SDK.
build:
	go build ./...

## test: run tests.
test:
	go test ./...

## fmt: format the code.
fmt:
	gofmt -w .

## vet: run go vet.
vet:
	go vet ./...

## tidy: tidy go.mod / go.sum.
tidy:
	go mod tidy

## clean: remove generated protobuf code.
clean:
	rm -f $(PB_DIR)/*.pb.go
