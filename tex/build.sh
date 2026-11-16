#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"

# Для установленного локально TeX-bundle Tectonic нужен Biber 2.17.
# BIBER_BIN позволяет выбрать другую установленную версию.
if [ -n "${BIBER_BIN:-}" ]; then
    PATH="$(dirname "$BIBER_BIN"):$PATH"
elif [ -x "$HOME/Library/Application Support/tectonic/biber-2.17/biber" ]; then
    PATH="$HOME/Library/Application Support/tectonic/biber-2.17:$PATH"
fi
export PATH
mkdir -p build

if command -v tectonic >/dev/null 2>&1; then
    tectonic --keep-logs --keep-intermediates --outdir build "$@" report.tex
    cp build/report.pdf report.pdf
elif command -v latexmk >/dev/null 2>&1; then
    latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build report.tex
    cp build/report.pdf report.pdf
else
    printf '%s\n' 'Нужен Tectonic с Biber либо TeX Live с latexmk и XeLaTeX.' >&2
    exit 1
fi
printf '%s\n' "Готово: $(pwd)/report.pdf"
