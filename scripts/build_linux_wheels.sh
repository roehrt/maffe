#!/usr/bin/env bash
set -euo pipefail

# Run on the host; cibuildwheel builds and tests inside manylinux containers.
# Pass --only cp311-manylinux_x86_64 to build just the first supported Python.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Run this script on a Linux host with Docker available.' >&2
  exit 1
fi

# Keep local build directories and downloaded subprojects out of the container.
# This also checks that the pinned dependencies can be fetched from scratch.
wheel_source=$(mktemp -d)
wheel_output="$PWD/dist-wheels"
trap 'rm -rf -- "$wheel_source"' EXIT
cp -R bindings cmake include src tests "$wheel_source/"
cp meson.build meson_options.txt pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.txt "$wheel_source/"
mkdir -p "$wheel_source/subprojects"
cp subprojects/*.wrap "$wheel_source/subprojects/"
cp -R subprojects/packagefiles "$wheel_source/subprojects/"

cd -- "$wheel_source"
uv run --no-project --with 'cibuildwheel==3.4.1' \
  cibuildwheel --output-dir "$wheel_output" "$@"
