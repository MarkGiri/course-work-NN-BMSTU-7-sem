.PHONY: all clean
all: report.pdf

report.pdf: $(wildcard *.tex) ref-lib.bib $(wildcard images/*) build.sh
	./build.sh

clean:
	rm -rf build
