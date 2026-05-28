.PHONY: all clean figures pdf make_figures

# Project root (two levels up from this directory)
ROOT := $(shell git rev-parse --show-toplevel 2>/dev/null || pwd)

# TeX build toolchain via nix flake
LATEXMK := latexmk
PDFLATEX := pdflatex
FIG_DIR  := figures
RESULTS  := results

all: pdf

# ── Figures ──────────────────────────────────────────────────────────────────────
# Regenerate all figures from pre-computed results
figures: make_figures.py results/spectrum_comparison.json
	python $(shell pwd)/make_figures.py

# ── LaTeX ─────────────────────────────────────────────────────────────────────
# Requires: nix develop (provides pdflatex, biber, IEEEtran class)
pdf: figures
	$(LATEXMK) -pdf -interaction=nonstopmode main.tex
	@mv -f main.pdf $(FIG_DIR)/..

# The main.tex uses the QD_2026.cls from the project template
# Ensure we can find it
TEXINPUTS:=$(ROOT)/papers/templates:$(ROOT)/papers/rps-from-drone-sound:$(TEXINPUTS)
export TEXINPUTS

clean:
	rm -f *.synctex.gz *.bbl *.blg *.aux *.log *.fls *.fdb_latexmk
	rm -f main.pdf
	$(LATEXMK) -C 2>/dev/null || true
