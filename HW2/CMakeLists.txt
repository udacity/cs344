############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com</email>
############################################################################


if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "-arch;sm_20;-Xptxas;-v")
    # collect source files

	SET (compare_files compare.cpp)
	
    file( GLOB  hdr *.hpp *.h )
    file( GLOB  cu  *.cu)
	SET (HW2_files main.cpp)
    
	link_libraries(HW2 ${OpenCV_LIBS})
    CUDA_ADD_EXECUTABLE(HW2 ${HW2_files} ${hdr} ${img} ${cu})
    CUDA_ADD_CUFFT_TO_TARGET(HW2)
    CUDA_ADD_CUBLAS_TO_TARGET(HW2)

	add_executable(hw2_compare ${compare_files})
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()