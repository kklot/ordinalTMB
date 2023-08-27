all: ordinalTMB.so

ordinalTMB.so: ordinalTMB.cpp
	Rscript --vanilla -e "TMB::compile('ordinalTMB.cpp', '-I../inst/include')"

clean:
	rm -rf *.o