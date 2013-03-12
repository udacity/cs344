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
SET (HW4_files main.cpp loadSaveImage.cpp)

CUDA_ADD_EXECUTABLE(HW4 ${HW4_files} ${hdr} ${img} ${cu})

ADD_EXECUTABLE(hw4_compare ${compare_files})

