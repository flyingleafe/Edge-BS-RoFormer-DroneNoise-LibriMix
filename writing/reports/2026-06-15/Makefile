TITLE := $(notdir $(CURDIR))
ROOT := $(shell git rev-parse --show-toplevel)

all: figures report.pdf

figures:
	python3 prepare.py

report.pdf: report.typ figures
	typst compile --root $(ROOT) report.typ

watch:
	typst watch --root $(ROOT) report.typ

check: report.typ
	mkdir -p check
	typst compile --root $(ROOT) --format png report.typ check/page-{p}.png

.PHONY: all figures watch check
