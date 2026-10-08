PDFLATEX ?= pdflatex -interaction=nonstopmode -halt-on-error
all: main.pdf
main.pdf: main.tex sections/*.tex refs.bib
	$(PDFLATEX) main && bibtex main && $(PDFLATEX) main && $(PDFLATEX) main
assets:
	cd .. && deadeye report results/sweep_gpu results/sweep_controls results/ablations --out report/paper --paper-dir paper
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc main.pdf
.PHONY: all assets clean
