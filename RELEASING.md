# Releasing KowitoDB

All workspace crates share one version and are auto-published to
[crates.io](https://crates.io) by [`.github/workflows/publish.yml`](.github/workflows/publish.yml)
when you push a `vX.Y.Z` tag.

## One-time setup

1. Create a crates.io API token: <https://crates.io/settings/tokens> (scope:
   *publish-new* + *publish-update*).
2. Add it as a GitHub Actions secret named **`CARGO_REGISTRY_TOKEN`**
   (repo *Settings ▸ Secrets and variables ▸ Actions ▸ New repository secret*).
3. Make sure the crate names (`kowitodb`, `kowitodb-core`, …) are available or
   owned by your crates.io account. The first publish claims them.

## Cutting a release

The crates' versions **and** their internal dependency versions must move
together. The easiest way is [`cargo-edit`](https://github.com/killercup/cargo-edit):

```bash
cargo install cargo-edit            # once
make bump V=0.41.0                  # set-version --workspace + ci + commit + tag
git push origin main --tags         # the tag triggers the crates.io publish
```

`make bump` runs `cargo set-version --workspace` (bumps every crate **and** its
internal dependency versions in lockstep), then `make ci`, then commits and tags.

Pushing the tag triggers the `publish` workflow, which verifies the tag matches
the workspace version and publishes the crates **in dependency order**
(`kowitodb-core` → `-storage`/`-index` → `-planner`/`-sql` → `-server` →
`kowitodb`). `cargo publish` waits for each crate to appear in the index before
the next dependent is published. Re-running on an already-published version is a
no-op (idempotent).

> If you don't use `cargo-edit`, bump `version` under `[workspace.package]` **and**
> the `version = "…"` on every internal crate in `[workspace.dependencies]` in the
> root `Cargo.toml` by hand — they must match, or publishing fails.

## Dry run

Validate packaging without uploading anytime from the Actions tab: run the
**publish** workflow manually (`workflow_dispatch`) with *Dry run* checked
(the default). It packages every crate (`cargo publish --dry-run --no-verify`)
and uploads nothing.
