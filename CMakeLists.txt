############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com</email>
############################################################################

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(cs344)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

#message("CMAKE_SIZEOF_VOID_P = ${CMAKE_SIZEOF_VOID_P}")

link_libraries(${OpenCV_LIBS} )

if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "-arch;sm_20;-Xptxas;-v")
    add_subdirectory (HW1/student)
    add_subdirectory (HW2)
    add_subdirectory (HW3)
    add_subdirectory (HW4)
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()