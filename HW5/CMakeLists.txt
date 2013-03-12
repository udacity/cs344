############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

# collect source files

SET (compare_files compare.cu)

file( GLOB  hdr *.hpp *.h )

SET (HW5_files main.cu student.cu)

CUDA_ADD_EXECUTABLE(HW5 ${HW5_files} ${hdr})

CUDA_ADD_EXECUTABLE(hw5_compare ${hdr} ${compare_files})

