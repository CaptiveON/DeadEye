PDFLATEX ?= pdflatex -interaction=nonstopmode -halt-on-error
all: main.pdf
main.pdf: main.tex sections/*.tex refs.bib
	$(PDFLATEX) main && bibtex main && $(PDFLATEX) main && $(PDFLATEX) main
assets:
	cd .. && deadeye report results/mac_main results/mac_lora results/mac_free_reply results/mac_controls results/ablations results/decision_models --out report/paper --paper-dir paper
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc main.pdf
.PHONY: all assets clean
