# Releasing Systrophe

This repo uses **PyPI Trusted Publishing** (OIDC). Releases are
triggered by pushing a `v*` git tag; GitHub Actions builds the
distribution and uploads it to PyPI without any stored API token.

## One-time setup: register the trusted publisher on PyPI

Required before the first tag-triggered release works.

1. Log in to https://pypi.org with the `systrophe` project owner
   account.

2. Visit https://pypi.org/manage/project/systrophe/settings/publishing/
   (or: project page -> Settings -> Publishing).

3. Add a new pending publisher with these values:

   - **PyPI Project Name**: `systrophe`
   - **Owner**: `Zynerji`
   - **Repository name**: `systrophe`
   - **Workflow name**: `publish.yml`
   - **Environment name**: leave blank (or set to `release` if you
     also want to create a GitHub environment with manual-approval
     gating)

4. Click **Add**.

After this, the API token used to publish v0.14.1 from the local
machine can be revoked --- all future publishes flow through
GitHub Actions.

## Cutting a release

```bash
# 1. Bump version in BOTH files
$EDITOR pyproject.toml                  # version = "0.X.Y"
$EDITOR src/systrophe/__init__.py       # __version__ = "0.X.Y"

# 2. Update CHANGELOG.md (add a new section at the top)
$EDITOR CHANGELOG.md

# 3. Commit and push
git add pyproject.toml src/systrophe/__init__.py CHANGELOG.md
git commit -m "release v0.X.Y"
git push

# 4. Tag and push tag
git tag v0.X.Y
git push origin v0.X.Y
```

GitHub Actions takes it from there:

- `tests.yml` runs the full pytest suite on Linux + Windows + macOS
  across Python 3.10/3.11/3.12 (also fires on every push to main).
- `publish.yml` builds `dist/*.whl` + `dist/*.tar.gz` and uploads to
  PyPI via Trusted Publishing on any `v*` tag push.

After ~3 minutes, the new version appears at
https://pypi.org/project/systrophe/ and `pip install --upgrade systrophe`
pulls it.

## Manual trigger (workflow_dispatch)

If a release ever needs to be re-run without a new tag, go to
Actions -> Publish to PyPI -> Run workflow. This uses
the latest commit on `main` and re-builds.

## Failure recovery

If the publish fails partway:

- **Build failed**: fix the build, push a new commit, then re-tag
  (delete the old tag locally and remotely with `git tag -d vX.Y.Z`
  + `git push --delete origin vX.Y.Z` before re-tagging).
- **Upload failed (token / permission)**: re-check Trusted Publishing
  config on PyPI. Re-run via workflow_dispatch.
- **PyPI rejected duplicate version**: PyPI does NOT allow
  overwriting an existing version. Bump version, push new tag.

## Yanking a broken release

If you ship a release with a serious bug:

1. Yank the version on PyPI:
   https://pypi.org/manage/project/systrophe/release/0.X.Y/ -> Yank.
   This hides it from new installs but keeps existing pins working.
2. Publish a patched version (bump patch number).

Yanking is preferred over deletion. PyPI normally does not allow
deletion of a version once published.
