#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>

#include "loadSaveImage.h"
#include <stdio.h>


//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
void preProcess( uchar4 **sourceImg,
                 size_t &numRows,  size_t &numCols,
                 uchar4 **destImg, 
                 uchar4 **blendedImg, const std::string& source_filename,
                 const std::string& dest_filename){

  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  size_t numRowsSource, numColsSource, numRowsDest, numColsDest;

  loadImageRGBA(source_filename, sourceImg, &numRowsSource, &numColsSource);
  loadImageRGBA(dest_filename, destImg, &numRowsDest, &numColsDest);

  assert(numRowsSource == numRowsDest);
  assert(numColsSource == numColsDest);

  numRows = numRowsSource;
  numCols = numColsSource;

  *blendedImg = new uchar4[numRows * numCols];

}

void postProcess(const uchar4* const blendedImg,
                 const size_t numRowsDest, const size_t numColsDest,
                 const std::string& output_file)
{
  //just need to save the image...
  saveImageRGBA(blendedImg, numRowsDest, numColsDest, output_file);
}
