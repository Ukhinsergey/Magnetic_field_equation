FFTW_FLAGS 	=	-lfftw3_mpi -lfftw3 -lm
FLAGS1		=	-O3 -std=c++14 -Wall

task1:task1.o
	mpic++ -o task1 task1.o $(FLAGS1) $(FFTW_FLAGS)

task1.o:task1.cpp
	mpic++ -o task1.o $(FLAGS1) -c task1.cpp $(FFTW_FLAGS)





clean:
	rm -f *.o task1 task2