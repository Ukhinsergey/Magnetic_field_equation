FFTW_FLAGS 	=	-lfftw3_mpi -lfftw3 -lm
FLAGS1		=	-O3 -std=c++11 -Wall

task1:task1.o
	mpic++ -o task1 $(FLAGS1) task1.o $(FFTW_FLAGS)

task1.o:task1.cpp
	mpic++ -o task1.o $(FLAGS1) -c task1.cpp $(FFTW_FLAGS)


task2:task2.o
	mpic++ -o task2  $(FLAGS1) task2.o $(FFTW_FLAGS)

task2.o:task2.cpp
	mpic++ -o task2.o $(FLAGS1) -c task2.cpp $(FFTW_FLAGS)


report: task2
	#mpisubmit.pl -w 00:30 -p 1 ./task2 -- 1000 128 0.0001 1 proc1.data
	#mpisubmit.pl -w 00:30 -p 2 ./task2 -- 1000 128 0.0001 1 proc2.data
	#mpisubmit.pl -w 00:30 -p 4 ./task2 -- 1000 128 0.0001 1 proc4.data
	#mpisubmit.pl -w 00:30 -p 8 ./task2 -- 1000 128 0.0001 1 proc8.data
	#mpisubmit.pl -w 00:30 -p 16 ./task2 -- 1000 128 0.0001 1 proc16.data
	#mpisubmit.pl -w 00:30 -p 32 ./task2 -- 1000 128 0.0001 1 proc32.data
	#mpisubmit.pl -w 00:30 -p 60 ./task2 -- 1000 128 0.0001 1 proc60.data
	#mpisubmit.pl -w 00:30 -p 1 ./task2 -- 1000 96 0.0001 1 proc1.data
	#mpisubmit.pl -w 00:30 -p 2 ./task2 -- 1000 96 0.0001 1 proc2.data
	#mpisubmit.pl -w 00:30 -p 4 ./task2 -- 1000 96 0.0001 1 proc4.data
	mpisubmit.pl -w 00:30 -p 8 ./task2 -- 1000 96 0.0001 1 proc8.data
	mpisubmit.pl -w 00:30 -p 16 ./task2 -- 1000 96 0.0001 1 proc16.data
	mpisubmit.pl -w 00:30 -p 32 ./task2 -- 1000 96 0.0001 1 proc32.data
	#mpisubmit.pl -w 00:30 -p 60 ./task2 -- 1000 96 0.0001 1 proc60.data

submit:task2
	#mpisubmit.pl -w 00:30 -p 60 ./task2 -- 300000 128 0.001 0.1 file_energy.data
	mpisubmit.pl -w 00:30 -p 60 ./task2 -- 300000 128 0.0001 1 file_energy.data
	#mpisubmit.pl -w 00:30 -p 60 ./task2 -- 300000 128 0.0001 0.1 file_energy.data
	#bsub -n 32 -W 01:05 -q normal -eo err -oo out ./task2 300000 128 0.0001 0.1 file_energy.data

clean:
	rm -f *.o task1 task2 *.data