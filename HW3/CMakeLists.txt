############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

SET (compare_files compare.cpp)

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)
SET (HW3_files main.cpp loadSaveImage.cpp)
    
CUDA_ADD_EXECUTABLE(HW3 ${HW3_files} ${hdr} ${img} ${cu})

ADD_EXECUTABLE(hw3_compare ${compare_files} )
#link_libraries(hw3_compare HW3)