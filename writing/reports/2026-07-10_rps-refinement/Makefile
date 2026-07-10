TITLE := $(notdir $(CURDIR))
ROOT := $(shell git rev-parse --show-toplevel)

all: figures report.pdf

figures:
	PYTHONPATH=$(ROOT) python3 prepare.py

report.pdf: report.typ figures
	typst compile --root $(ROOT) report.typ

watch:
	typst watch --root $(ROOT) report.typ

check: report.pdf
	mkdir -p check
	pdftoppm -png -r 150 $< check/page

.PHONY: all figures watch check
