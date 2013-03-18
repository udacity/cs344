NVCC=nvcc
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64

histo: main.cu reference.cpp student.o Makefile
	nvcc -o hw main.cu student.o $(NVCC_OPTS)

student.o: student.cu
	nvcc -c student.cu $(NVCC_OPTS)

clean:
	rm -f *.o hw *.bin
