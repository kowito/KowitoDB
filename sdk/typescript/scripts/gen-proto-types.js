#!/usr/bin/env node
/**
 * proto:gen — sync the canonical proto into this package.
 *
 * The TypeScript SDK loads the proto dynamically at runtime via
 * @grpc/proto-loader (no static *_pb codegen). The typed surface lives in
 * hand-written interfaces in `src/types.ts`. This script keeps the bundled
 * copy of the proto (`proto/kowitodb.proto`) in sync with the repo's
 * canonical proto so the package stays self-contained when published.
 *
 * If you ever change `proto/kowitodb.proto`, remember to update the matching
 * interfaces in `src/types.ts`.
 */

const fs = require("fs");
const path = require("path");

const PACKAGE_DIR = path.resolve(__dirname, "..");
const CANONICAL_PROTO = path.resolve(PACKAGE_DIR, "..", "..", "proto", "kowitodb.proto");
const BUNDLED_PROTO = path.resolve(PACKAGE_DIR, "proto", "kowitodb.proto");

function main() {
  if (!fs.existsSync(CANONICAL_PROTO)) {
    console.error(`Canonical proto not found at ${CANONICAL_PROTO}`);
    console.error("Skipping sync; using the bundled copy as-is.");
    process.exit(0);
  }

  fs.mkdirSync(path.dirname(BUNDLED_PROTO), { recursive: true });
  fs.copyFileSync(CANONICAL_PROTO, BUNDLED_PROTO);
  console.log(`Synced proto: ${CANONICAL_PROTO} -> ${BUNDLED_PROTO}`);
  console.log("Reminder: keep src/types.ts in sync with the proto messages.");
}

main();
