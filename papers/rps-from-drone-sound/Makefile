# Build the paper.  Run from the project root inside `nix develop`,
# or from this directory after entering the dev shell.

DOC = main
BIB = refs

.PHONY: pdf figures clean veryclean

pdf: figures
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(DOC).tex

figures:
	cd ../.. && python papers/rps-from-drone-sound/make_figures.py

clean:
	latexmk -c

veryclean:
	latexmk -C
	rm -f $(DOC).bbl $(DOC).run.xml
