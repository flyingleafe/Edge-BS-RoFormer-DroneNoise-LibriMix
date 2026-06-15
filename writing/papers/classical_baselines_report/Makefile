# Classical baselines report — unified build
# Replaces the old 810-line duo with one script using tasks.rps_prediction

.PHONY: all figures tables pdf clean clobber

all: figures tables pdf

figures tables:
	@python3 build.py

pdf:
	@pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
	@pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
	@echo "Built main.pdf"

clean:
	@rm -f main.aux main.log main.out

clobber: clean
	@rm -f figures/*.pdf figures/*.tex figures/*.json
