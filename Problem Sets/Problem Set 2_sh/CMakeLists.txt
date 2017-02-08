############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

cmake_minimum_required(VERSION 3.0)
FIND_PACKAGE(CUDA)
set ( OpenCV_DIR   "/home/guilherme/opencv/build/")
include_directories(${OpenCV_DIR})
find_package( OpenCV 3.0 REQUIRED )
include_directories(AFTER /usr/local/cuda-8.0/include ${OpenCV_INCLUDE_DIRS})

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW2_files main.cpp reference_calc.cpp compare.cpp)
link_libraries()

CUDA_ADD_EXECUTABLE(HW2 ${HW2_files} ${hdr} ${cu})

TARGET_LINK_LIBRARIES(HW2 ${CMAKE_THREAD_LIBS_INIT} ${CUDA_LIBRARIES} ${NVVM_LIB} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_CUFFT_LIBRARIES} ${CUDA_nvvm_LIBRARY} ${CUDA_CUDA_LIBRARY} ${OpenCV_LIBS})
