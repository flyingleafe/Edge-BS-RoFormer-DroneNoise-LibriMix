TITLE := $(notdir $(CURDIR))
ROOT := $(shell git rev-parse --show-toplevel)

all: figures slides.pdf

figures:
	python3 prepare.py

slides.pdf: slides.typ figures
	typst compile --root $(ROOT) slides.typ

watch:
	typst watch --root $(ROOT) slides.typ

check: slides.typ
	mkdir -p check
	typst compile --root $(ROOT) --format png slides.typ check/page-{p}.png

.PHONY: all figures watch check
