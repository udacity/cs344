#ifndef LOADSAVEIMAGE_H__
#define LOADSAVEIMAGE_H__

#include <string>
#include <cuda_runtime.h> //for uchar4

void loadImageHDR(const std::string &filename,
                  float **imagePtr,
                  size_t *numRows, size_t *numCols);

void loadImageRGBA(const std::string &filename,
                   uchar4 **imagePtr,
                   size_t *numRows, size_t *numCols);

void saveImageRGBA(const uchar4* const image,
                   const size_t numRows, const size_t numCols,
                   const std::string &output_file);

void saveImageHDR(const float* const image,
                  const size_t numRows, const size_t numCols,
                  const std::string &output_file);

#endif
