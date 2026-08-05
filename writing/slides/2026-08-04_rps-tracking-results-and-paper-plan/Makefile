TITLE := $(notdir $(CURDIR))
ROOT := $(shell git rev-parse --show-toplevel)

all: figures slides.pdf

figures:
	PYTHONPATH=$(ROOT) python3 prepare_figs.py

slides.pdf: slides.typ figures
	typst compile --root $(ROOT) slides.typ

notes: slides.typ figures
	typst compile --root $(ROOT) --input notes=1 slides.typ slides-notes.pdf

watch:
	typst watch --root $(ROOT) slides.typ

check: slides.pdf
	rm -rf check
	mkdir -p check
	pdftoppm -png -r 150 $< check/page

.PHONY: all figures watch check
