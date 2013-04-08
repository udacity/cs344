//Udacity HW3 Driver

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
#include <algorithm>

#include "compare.h"
#include "reference_calc.h"

// Functions from HW3.cu
void preProcess(float **d_luminance, unsigned int **d_cdf,
                size_t *numRows, size_t *numCols, unsigned int *numBins,
                const std::string& filename);

void postProcess(const std::string& output_file, size_t numRows, size_t numCols,
                 float min_logLum, float max_logLum);

void cleanupGlobalMemory(void);

// Function from student_func.cu
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
  std::string reference_file;
  double perPixelError = 0.0;
  double globalError   = 0.0;
  bool useEpsCheck = false;

  switch (argc)
  {
	case 2:
	  input_file = std::string(argv[1]);
	  output_file = "HW3_output.png";
	  reference_file = "HW3_reference.png";
	  break;
	case 3:
	  input_file  = std::string(argv[1]);
      output_file = std::string(argv[2]);
	  reference_file = "HW3_reference.png";
	  break;
	case 4:
	  input_file  = std::string(argv[1]);
      output_file = std::string(argv[2]);
	  reference_file = std::string(argv[3]);
	  break;
	case 6:
	  useEpsCheck=true;
	  input_file  = std::string(argv[1]);
	  output_file = std::string(argv[2]);
	  reference_file = std::string(argv[3]);
	  perPixelError = atof(argv[4]);
      globalError   = atof(argv[5]);
	  break;
	default:
      std::cerr << "Usage: ./HW3 input_file [output_filename] [reference_filename] [perPixelError] [globalError]" << std::endl;
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
  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  float *h_luminance = (float *) malloc(sizeof(float)*numRows*numCols);
  unsigned int *h_cdf = (unsigned int *) malloc(sizeof(unsigned int)*numBins);

  checkCudaErrors(cudaMemcpy(h_luminance, d_luminance, numRows*numCols*sizeof(float), cudaMemcpyDeviceToHost));

  //check results and output the tone-mapped image
  postProcess(output_file, numRows, numCols, min_logLum, max_logLum);

  for (size_t i = 1; i < numCols * numRows; ++i) {
	min_logLum = std::min(h_luminance[i], min_logLum);
    max_logLum = std::max(h_luminance[i], max_logLum);
  }

  referenceCalculation(h_luminance, h_cdf, numRows, numCols, numBins, min_logLum, max_logLum);

  checkCudaErrors(cudaMemcpy(d_cdf, h_cdf, sizeof(unsigned int) * numBins, cudaMemcpyHostToDevice));

  //check results and output the tone-mapped image
  postProcess(reference_file, numRows, numCols, min_logLum, max_logLum);

  cleanupGlobalMemory();

  compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);

  return 0;
}
