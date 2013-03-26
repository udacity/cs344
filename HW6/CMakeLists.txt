############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

SET (compare_files compare.cpp)

file( GLOB  hdr *.hpp *.h )

SET (HW6_files student_func.cu HW6.cu main.cpp loadSaveImage.cpp)

CUDA_ADD_EXECUTABLE(HW6 ${HW6_files} ${hdr})

ADD_EXECUTABLE(hw6_compare ${hdr} ${compare_files})

