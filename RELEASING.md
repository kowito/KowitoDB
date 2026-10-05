# Releasing KowitoDB

All workspace crates share one version and are published to
[crates.io](https://crates.io) by [`.github/workflows/publish.yml`](.github/workflows/publish.yml)
when a `vX.Y.Z` tag is pushed. Nothing is published without a tag.

## One-time setup

1. Create a crates.io API token: <https://crates.io/settings/tokens> (scope:
   *publish-new* + *publish-update*).
2. Add it as a GitHub Actions secret named **`CARGO_REGISTRY_TOKEN`**
   (repo *Settings ▸ Secrets and variables ▸ Actions ▸ New repository secret*).
3. Make sure the crate names (`kowitodb`, `kowitodb-core`, …) are available or
   owned by your crates.io account. The first publish claims them.

## How versions move

A version lives in three places that must agree, or `cargo publish --locked`
fails:

- `version` under `[workspace.package]` in the root `Cargo.toml`;
- the `version = "…"` on every internal `kowitodb-*` crate in
  `[workspace.dependencies]` (root `Cargo.toml`);
- the workspace packages' own entries in `Cargo.lock`.

[`scripts/bump-version.py`](scripts/bump-version.py) updates all three and
prints the new version:

```bash
python3 scripts/bump-version.py                # patch bump: 0.40.5 -> 0.40.6
python3 scripts/bump-version.py minor          # 0.40.5 -> 0.41.0
python3 scripts/bump-version.py --set 0.41.0   # explicit version
```

### The auto-bump bot

[`.github/workflows/bump-version.yml`](.github/workflows/bump-version.yml) runs
on every push to `main`. Unless the head commit's message starts with
`chore: bump` (or contains `[skip ci]`), it runs the script (patch bump),
commits `Cargo.toml` + `Cargo.lock` as **`chore: bump version to X.Y.Z`**, and
pushes. It cannot loop: pushes made with the workflow's `GITHUB_TOKEN` don't
trigger workflows, and the `chore: bump` guard skips them anyway.

So `main` always carries a fresh, never-published patch version, but **the bot
never tags and never publishes**. Its commits deliberately do *not* contain
`[skip ci]` — GitHub honours skip instructions on tag pushes too, so a tag on
such a commit would never run `publish.yml`.

## Cutting a release

Pick one:

**A. Release what the bot already bumped** (patch release). Tag the bot's
`chore: bump version to X.Y.Z` commit on `main` and push the tag:

```bash
git pull
grep -m1 '^version = ' Cargo.toml       # -> version = "X.Y.Z"
git tag vX.Y.Z                          # on the bump commit (HEAD of main)
git push origin vX.Y.Z
```

**B. Choose the version yourself** (minor/major, or any explicit version):

```bash
make bump V=0.41.0          # bump-version.py --set + make ci + commit + tag
git push origin main v0.41.0
```

`make bump` commits with the `chore: bump version to 0.41.0` message, so the bot
does not bump again on top of it.

Either way, pushing the tag triggers the `publish` workflow, which verifies the
tag matches the workspace version and publishes the crates **in dependency
order** with `cargo publish --locked` (`kowitodb-core` → `-storage`/`-index` →
`-planner`/`-sql` → `-server` → `kowitodb`). `cargo publish` waits for each
crate to appear in the index before the next dependent is published. Re-running
on an already-published version is a no-op (idempotent), so a failed publish can
be retried from the Actions tab.

> Bumping by hand? Edit all three locations listed above (or just run the
> script). After editing only `Cargo.toml` (e.g. with cargo-edit's
> `cargo set-version --workspace X.Y.Z`), run `cargo update --workspace` to sync
> `Cargo.lock`. CI builds with `--locked`, so a stale `Cargo.lock` fails CI
> rather than the release.

## SDK versions

The Python (`sdk/python/pyproject.toml`), TypeScript (`sdk/typescript/package.json`)
and Go (`sdk/go`; Go sub-module tags look like `sdk/go/vX.Y.Z`, which do not
match the `v*` publish trigger) SDKs are versioned independently of the crates
and are not touched by the bot or `make bump`.

## Dry run

Validate packaging without uploading anytime from the Actions tab: run the
**publish** workflow manually (`workflow_dispatch`) with *Dry run* checked
(the default). It packages every crate (`cargo publish --dry-run --no-verify`)
and uploads nothing.
