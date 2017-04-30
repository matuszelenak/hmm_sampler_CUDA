all: main.pdf

main.pdf: main.tex *.tex *.bib 
	pdflatex main
	bibtex main
	pdflatex main
	pdflatex main
