TITLE := $(notdir $(CURDIR))
ROOT := $(shell git rev-parse --show-toplevel)

all: figures slides.pdf

figures:
	PYTHONPATH=$(ROOT)/src python3 prepare.py

slides.pdf: slides.typ
	typst compile --root $(ROOT) slides.typ

watch:
	typst watch --root $(ROOT) slides.typ

check: slides.pdf
	mkdir -p check
	rm -f check/page-*.png
	pdftoppm -png -r 150 $< check/page

.PHONY: all figures watch check
