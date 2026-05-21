# Build the paper.  Run from the project root inside `nix develop`,
# or from this directory after entering the dev shell.

DOC = main
BIB = refs

.PHONY: pdf figures clean veryclean

pdf: figures
	pdflatex -interaction=nonstopmode $(DOC).tex
	bibtex $(DOC)
	pdflatex -interaction=nonstopmode $(DOC).tex
	pdflatex -interaction=nonstopmode $(DOC).tex

figures:
	cd ../.. && python papers/rps-from-drone-sound/make_figures.py

clean:
	latexmk -c

veryclean:
	latexmk -C
	rm -f $(DOC).bbl $(DOC).run.xml
