#!/bin/sh
set -eu
exec "$(dirname "$0")/tex/build.sh" "$@"
