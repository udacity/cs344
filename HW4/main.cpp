//Udacity HW4 Driver

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "HW4.h"
#include "reference_calc.h"

void preProcess(unsigned int **inputVals,
                unsigned int **inputPos,
                unsigned int **outputVals,
                unsigned int **outputPos,
                size_t &numElems,
                const std::string& filename);

void postProcess(const unsigned int* const outputVals,
                 const unsigned int* const outputPos,
                 const size_t numElems,
                 const std::string& output_file);

void your_sort(unsigned int* const inputVals,
               unsigned int* const inputPos,
               unsigned int* const outputVals,
               unsigned int* const outputPos,
               const size_t numElems);

int main(int argc, char **argv) {
  unsigned int *inputVals;
  unsigned int *inputPos;
  unsigned int *outputVals;
  unsigned int *outputPos;

  size_t numElems;

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
	  output_file = "HW4_output.png";
	  reference_file = "HW4_reference.png";
	  break;
	case 3:
	  input_file  = std::string(argv[1]);
      output_file = std::string(argv[2]);
	  reference_file = "HW4_reference.png";
	  break;
	case 4:
	  input_file  = std::string(argv[1]);
      output_file = std::string(argv[2]);
	  reference_file = std::string(argv[3]);
	  break;
	case 6:
	  useEpsCheck=true;
	  perPixelError = atof(argv[4]);
      globalError   = atof(argv[5]);
	  break;
	default:
      std::cerr << "Usage: ./HW4 input_file [output_filename] [reference_filename] [perPixelError] [globalError]" << std::endl;
      exit(1);
  }
  //load the image and give us our input and output pointers
  preProcess(&inputVals, &inputPos, &outputVals, &outputPos, numElems, input_file);

  GpuTimer timer;
  timer.Start();

  //call the students' code
  your_sort(inputVals, inputPos, outputVals, outputPos, numElems);

  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  printf("\n");
  int err = printf("%f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  //check results and output the red-eye corrected image
  postProcess(outputVals, outputPos, numElems, output_file);

  // check code moved from HW4.cu
  /****************************************************************************
  * You can use the code below to help with debugging, but make sure to       *
  * comment it out again before submitting your assignment for grading,       *
  * otherwise this code will take too much time and make it seem like your    *
  * GPU implementation isn't fast enough.                                     *
  *                                                                           *
  * This code MUST RUN BEFORE YOUR CODE in case you accidentally change       *
  * the input values when implementing your radix sort.                       *
  *                                                                           *
  * This code performs the reference radix sort on the host and compares your *
  * sorted values to the reference.                                           *
  *                                                                           *
  * Thrust containers are used for copying memory from the GPU                *
  * ************************************************************************* */

  thrust::host_vector<unsigned int> h_inputVals(thrust::device_ptr<unsigned int>(inputVals),
									thrust::device_ptr<unsigned int>(inputVals) + numElems);
  thrust::host_vector<unsigned int> h_inputPos(thrust::device_ptr<unsigned int>(inputPos),
									thrust::device_ptr<unsigned int>(inputPos) + numElems);

  thrust::host_vector<unsigned int> h_outputVals(numElems);
  thrust::host_vector<unsigned int> h_outputPos(numElems);

  reference_calculation(&h_inputVals[0], &h_inputPos[0],
						&h_outputVals[0], &h_outputPos[0],
						numElems);

  // post-processing uses the GPU - we have to move our host-side output back to the GPU.
  thrust::device_vector<unsigned int> d_outputVals = h_outputVals;
  thrust::device_vector<unsigned int> d_outputPos = h_outputPos;

  unsigned int *valsPtr = thrust::raw_pointer_cast(&d_outputVals[0]);
  unsigned int *posPtr = thrust::raw_pointer_cast(&d_outputPos[0]);

  postProcess(valsPtr, posPtr, numElems, reference_file);

  compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);

  checkCudaErrors(cudaFree(inputVals));
  checkCudaErrors(cudaFree(inputPos));
  checkCudaErrors(cudaFree(outputVals));
  checkCudaErrors(cudaFree(outputPos));

  return 0;
}
