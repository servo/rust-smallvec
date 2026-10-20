#!/bin/sh
set -eu
rustdoc=$(command -v rustdoc)
if [ -n "${OUT_DIR:-}" ]; then
    exec "$rustdoc" -L "all=$OUT_DIR" "$@"
fi
exec "$rustdoc" "$@"
