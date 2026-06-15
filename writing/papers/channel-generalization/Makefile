# Makefile for Channel-Generalization Report
# Usage: make pdf     — compile LaTeX
#        make clean   — remove generated PDFs and auxiliary files

LATEX     := latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all pdf clean

all: pdf

pdf: channel-generalization.pdf

channel-generalization.pdf: main.tex
	@echo "=== Compiling LaTeX ==="
	$(LATEX) main.tex
	@cp main.pdf channel-generalization.pdf

clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz *.fdb_latexmk *.fls *.bbl *.blg
	rm -f main.pdf channel-generalization.pdf
