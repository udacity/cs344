NVCC=/usr/local/cuda-5.0/bin/nvcc
#NVCC=nvcc

###################################
# These are the default install   #
# locations on most linux distros #
###################################

OPENCV_LIBPATH=/usr/lib
OPENCV_INCLUDEPATH=/usr/include

###################################################
# On Macs the default install locations are below #
###################################################

#OPENCV_LIBPATH=/usr/local/lib
#OPENCV_INCLUDEPATH=/usr/local/include

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui

CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
# CUDA_INCLUDEPATH=/usr/local/cuda/lib64/include
# CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include
# CUDA_INCLUDEPATH=/Developer/NVIDIA/CUDA-5.0/include

######################################################
# On Macs the default install locations are below    #
# ####################################################

#CUDA_INCLUDEPATH=/usr/local/cuda/include
#CUDA_LIBPATH=/usr/local/cuda/lib
CUDA_LIBPATH=/usr/local/cuda-5.0/lib64

#no warnings otherwise thrust explodes output

NVCC_OPTS=-O3 -arch=sm_20 -m64

GCC_OPTS=-O3 -m64

student: main.o student_func.o HW6.o loadSaveImage.o compare.o reference_calc.o Makefile
	$(NVCC) -o HW6 main.o student_func.o HW6.o loadSaveImage.o compare.o reference_calc.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cpp timer.h utils.h
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

HW6.o: HW6.cu loadSaveImage.h utils.h
	$(NVCC) -c HW6.cu -I $(OPENCV_INCLUDEPATH) $(NVCC_OPTS)

loadSaveImage.o: loadSaveImage.cpp loadSaveImage.h
	g++ -c loadSaveImage.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

student_func.o: student_func.cu reference_calc.cpp utils.h
	$(NVCC) -c student_func.cu $(NVCC_OPTS)

compare.o: compare.cpp compare.h
	g++ -c compare.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

reference_calc.o: reference_calc.cpp reference_calc.h
	g++ -c reference_calc.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

clean:
	rm -f *.o hw
	find . -type f -name '*.png' | grep -v source.png | grep -v destination.png | xargs rm -f
