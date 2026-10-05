#!/usr/bin/env bash
# Regenerate the Python gRPC stubs from kowitodb-server/proto/kowitodb.proto
# (the single source of truth) and fix the generated absolute import so the
# package imports cleanly. Run from anywhere:
#   bash sdk/python/scripts/gen.sh        (or: make gen-python)
#
# Requires: pip install "kowitodb[codegen]"   (i.e. grpcio-tools==1.81.1)
# The grpcio-tools version determines the runtime minimums the generated code
# enforces (grpcio>=1.81.1, protobuf>=6.33.5) — if you change it, update
# `dependencies` in sdk/python/pyproject.toml to match.
set -euo pipefail

cd "$(dirname "$0")/.."   # sdk/python

python -m grpc_tools.protoc -I ../../kowitodb-server/proto \
  --python_out=kowitodb --grpc_python_out=kowitodb \
  ../../kowitodb-server/proto/kowitodb.proto

# protoc emits `import kowitodb_pb2` (absolute); rewrite to a relative import so
# `from kowitodb import ...` works as an installed package.
sed -i.bak 's/^import kowitodb_pb2/from . import kowitodb_pb2/' \
  kowitodb/kowitodb_pb2_grpc.py
rm -f kowitodb/kowitodb_pb2_grpc.py.bak

echo "Regenerated Python stubs in sdk/python/kowitodb/."
