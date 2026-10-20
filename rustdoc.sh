#!/bin/sh
set -eu
if [ -z "${OUT_DIR:-}" ]; then
    echo "rustdoc-wrapper: Cargo did not set OUT_DIR" >&2
    exit 1
fi
rustdoc=$(command -v rustdoc)
exec "$rustdoc" -L "all=$OUT_DIR" "$@"
