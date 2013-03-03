############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

SET (compare_files compare.cpp)

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW1_files main.cpp)
    
    
CUDA_ADD_EXECUTABLE(HW1 ${HW1_files} ${hdr} ${img} ${cu})
CUDA_ADD_CUFFT_TO_TARGET(HW1)
CUDA_ADD_CUBLAS_TO_TARGET(HW1)

add_executable(hw1_compare ${compare_files})