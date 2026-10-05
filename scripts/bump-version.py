#!/usr/bin/env python3
"""Bump (or set) the workspace version in the root Cargo.toml AND Cargo.lock.

Updates, in lockstep:
  1. the `[workspace.package]` version in Cargo.toml,
  2. the `version = "…"` on every internal `kowitodb-*` dependency in
     `[workspace.dependencies]` (they must match for crates.io publishing),
  3. the `version = "…"` of the workspace's own packages in Cargo.lock, so
     `cargo publish --locked` / `cargo build --locked` keep working.

Prints the new version on stdout (nothing else). No external deps — fast enough
to run on every commit.

Usage:
  python3 scripts/bump-version.py                 # patch bump (default)
  python3 scripts/bump-version.py major|minor|patch
  python3 scripts/bump-version.py --set 0.41.0    # set an explicit version
  python3 scripts/bump-version.py --root DIR ...  # operate on DIR/Cargo.{toml,lock}
"""
import argparse
import re
import sys
from pathlib import Path

# The workspace's own packages (their Cargo.lock entries carry the workspace
# version and have no `source = …` line).
WORKSPACE_PACKAGES = {
    "kowitodb",
    "kowitodb-core",
    "kowitodb-storage",
    "kowitodb-index",
    "kowitodb-planner",
    "kowitodb-sql",
    "kowitodb-server",
}

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("part", nargs="?", default="patch",
                   choices=["major", "minor", "patch"],
                   help="which component to bump (default: patch)")
    p.add_argument("--set", dest="set_version", metavar="X.Y.Z",
                   help="set this exact version instead of bumping")
    p.add_argument("--root", type=Path,
                   default=Path(__file__).resolve().parent.parent,
                   help="workspace root containing Cargo.toml/Cargo.lock "
                        "(default: the repo root)")
    return p.parse_args(argv)


def bump_lock(lock_text, new):
    """Rewrite the version of every workspace package in Cargo.lock.

    Returns (new_text, set_of_package_names_updated).
    """
    seen = set()

    def repl(m):
        name = m.group("name")
        seen.add(name)
        return f'{m.group("head")}{new}{m.group("tail")}'

    # A workspace package entry looks like:
    #   [[package]]
    #   name = "kowitodb-core"
    #   version = "0.40.5"
    #   dependencies = [ ... ]          <- no `source = ` line (path package)
    # Registry/git packages have `source = "…"` right after `version`, so the
    # negative lookahead keeps a same-named crates.io package untouched.
    names = "|".join(re.escape(n) for n in sorted(WORKSPACE_PACKAGES))
    pattern = re.compile(
        r'(?m)(?P<head>^\[\[package\]\]\nname = "(?P<name>' + names + r')"\n'
        r'version = ")[^"]+(?P<tail>"\n)(?!source = )'
    )
    return pattern.sub(repl, lock_text), seen


def main(argv):
    args = parse_args(argv)
    cargo = args.root / "Cargo.toml"
    lock = args.root / "Cargo.lock"
    text = cargo.read_text()

    m = re.search(r'(?m)^version = "(\d+)\.(\d+)\.(\d+)"', text)
    if not m:
        sys.exit("could not find a `version = \"X.Y.Z\"` line in [workspace.package]")
    major, minor, patch = (int(g) for g in m.groups())

    if args.set_version:
        if not SEMVER.match(args.set_version):
            sys.exit(f"--set expects X.Y.Z, got {args.set_version!r}")
        new = args.set_version
    elif args.part == "major":
        new = f"{major + 1}.0.0"
    elif args.part == "minor":
        new = f"{major}.{minor + 1}.0"
    else:
        new = f"{major}.{minor}.{patch + 1}"

    # 1) the [workspace.package] version (first top-level `version = ` line)
    text = re.sub(r'(?m)^version = "\d+\.\d+\.\d+"', f'version = "{new}"',
                  text, count=1)
    # 2) every internal kowitodb-* dependency's version (whatever it was)
    text = re.sub(
        r'(?m)^(kowitodb-[a-z]+ = \{ path = "[^"]+", version = ")[^"]+(")',
        r"\g<1>%s\g<2>" % new, text)
    cargo.write_text(text)

    # 3) the workspace packages' own entries in Cargo.lock
    if lock.exists():
        lock_text, seen = bump_lock(lock.read_text(), new)
        missing = WORKSPACE_PACKAGES - seen
        if missing:
            print(f"warning: no Cargo.lock entry updated for: "
                  f"{', '.join(sorted(missing))}", file=sys.stderr)
        lock.write_text(lock_text)
    else:
        print("warning: no Cargo.lock found; only Cargo.toml was updated",
              file=sys.stderr)

    print(new)


if __name__ == "__main__":
    main(sys.argv[1:])
