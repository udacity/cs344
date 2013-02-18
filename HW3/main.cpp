//Udacity HW3 Driver

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>

void preProcess(float **d_luminance, unsigned int **d_cdf,
                size_t *numRows, size_t *numCols, unsigned int *numBins,
                const std::string& filename);

void postProcess(const std::string& output_file, size_t numRows, size_t numCols,
                 float min_logLum, float max_logLum);

void your_histogram_and_prefixsum(const float* const d_luminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins);

int main(int argc, char **argv) {
  float *d_luminance;
  unsigned int *d_cdf;

  size_t numRows, numCols;
  unsigned int numBins;

  std::string input_file;
  std::string output_file;
  if (argc == 3) {
    input_file  = std::string(argv[1]);
    output_file = std::string(argv[2]);
  }
  else {
    std::cerr << "Usage: ./hw input_file output_file" << std::endl;
    exit(1);
  }
  //load the image and give us our input and output pointers
  preProcess(&d_luminance, &d_cdf,
             &numRows, &numCols, &numBins, input_file);

  GpuTimer timer;
  float min_logLum, max_logLum;
  min_logLum = 0.f;
  max_logLum = 1.f;
  timer.Start();
  //call the students' code
  your_histogram_and_prefixsum(d_luminance, d_cdf, min_logLum, max_logLum,
                               numRows, numCols, numBins);
  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  int err = printf("%f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  //check results and output the tone-mapped image
  postProcess(output_file, numRows, numCols, min_logLum, max_logLum);

  return 0;
}
