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
	SET (HW4_files main.cpp loadSaveImage.cpp)
    
	link_libraries(HW4 ${OpenCV_LIBS})
    CUDA_ADD_EXECUTABLE(HW4 ${HW4_files} ${hdr} ${img} ${cu})
    CUDA_ADD_CUFFT_TO_TARGET(HW4)
    CUDA_ADD_CUBLAS_TO_TARGET(HW4)

	add_executable(hw4_compare ${compare_files} ${HW4_files} ${hdr} ${img} ${cu})
	link_libraries(hw4_compare HW4)
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()