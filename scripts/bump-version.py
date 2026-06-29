#!/usr/bin/env python3
"""Bump the workspace patch version in the root Cargo.toml.

Updates BOTH the `[workspace.package]` version and the `version = "…"` on every
internal `kowitodb-*` dependency in `[workspace.dependencies]` (they must match
for crates.io publishing). Prints the new version. No external deps — fast
enough to run on every commit.

Usage: python3 scripts/bump-version.py [major|minor|patch]   (default: patch)
"""
import re
import sys
from pathlib import Path

part = sys.argv[1] if len(sys.argv) > 1 else "patch"
cargo = Path(__file__).resolve().parent.parent / "Cargo.toml"
text = cargo.read_text()

m = re.search(r'(?m)^version = "(\d+)\.(\d+)\.(\d+)"', text)
if not m:
    sys.exit("could not find a `version = \"X.Y.Z\"` line in [workspace.package]")
major, minor, patch = (int(g) for g in m.groups())
old = f"{major}.{minor}.{patch}"
if part == "major":
    new = f"{major + 1}.0.0"
elif part == "minor":
    new = f"{major}.{minor + 1}.0"
else:
    new = f"{major}.{minor}.{patch + 1}"

# 1) the [workspace.package] version (first top-level `version = ` line)
text = re.sub(r'(?m)^version = "%s"' % re.escape(old),
              f'version = "{new}"', text, count=1)
# 2) every internal kowitodb-* dependency's version
text = re.sub(
    r'(kowitodb-[a-z]+ = \{ path = "[^"]+", version = ")%s(")' % re.escape(old),
    r"\g<1>%s\g<2>" % new, text)

cargo.write_text(text)
print(new)
