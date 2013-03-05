NVCC=nvcc
NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64

histo: compare main.cu reference.cpp student.o Makefile
	nvcc -o hw main.cu student.o $(NVCC_OPTS)

student.o: student.cu
	nvcc -c student.cu $(NVCC_OPTS)

compare: compare.cu
	nvcc -o compare compare.cu $(NVCC_OPTS)

clean:
	rm -f *.o compare hw *.bin
