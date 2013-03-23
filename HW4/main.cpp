//Udacity HW4 Driver

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "compare.h"
#include "reference_calc.h"

void preProcess(unsigned int **inputVals,
                unsigned int **inputPos,
                unsigned int **outputVals,
                unsigned int **outputPos,
                size_t &numElems,
                const std::string& filename,
				const std::string& template_file);

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
  std::string template_file;
  std::string output_file;
  std::string reference_file;
  double perPixelError = 0.0;
  double globalError   = 0.0;
  bool useEpsCheck = false;

  switch (argc)
  {
	case 3:
	  input_file  = std::string(argv[1]);
      template_file = std::string(argv[2]);
	  output_file = "HW4_output.png";
	  break;
	case 4:
	  input_file  = std::string(argv[1]);
      template_file = std::string(argv[2]);
	  output_file = std::string(argv[3]);
	  break;
	default:
          std::cerr << "Usage: ./HW4 input_file template_file [output_filename]" << std::endl;
          exit(1);
  }
  //load the image and give us our input and output pointers
  preProcess(&inputVals, &inputPos, &outputVals, &outputPos, numElems, input_file, template_file);

  GpuTimer timer;
  timer.Start();

  //call the students' code
  your_sort(inputVals, inputPos, outputVals, outputPos, numElems);

  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  printf("\n");
  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

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
  thrust::device_ptr<unsigned int> d_inputVals(inputVals);
  thrust::device_ptr<unsigned int> d_inputPos(inputPos);

  thrust::host_vector<unsigned int> h_inputVals(d_inputVals,
                                                d_inputVals+numElems);
  thrust::host_vector<unsigned int> h_inputPos(d_inputPos,
                                               d_inputPos + numElems);

  thrust::host_vector<unsigned int> h_outputVals(numElems);
  thrust::host_vector<unsigned int> h_outputPos(numElems);

  reference_calculation(&h_inputVals[0], &h_inputPos[0],
						&h_outputVals[0], &h_outputPos[0],
						numElems);

  //postProcess(valsPtr, posPtr, numElems, reference_file);

  //compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);

  thrust::device_ptr<unsigned int> d_outputVals(outputVals);
  thrust::device_ptr<unsigned int> d_outputPos(outputPos);

  thrust::host_vector<unsigned int> h_yourOutputVals(d_outputVals,
                                                     d_outputVals + numElems);
  thrust::host_vector<unsigned int> h_yourOutputPos(d_outputPos,
                                                    d_outputPos + numElems);

  checkResultsExact(&h_outputVals[0], &h_yourOutputVals[0], numElems);
  checkResultsExact(&h_outputPos[0], &h_yourOutputPos[0], numElems);

  checkCudaErrors(cudaFree(inputVals));
  checkCudaErrors(cudaFree(inputPos));
  checkCudaErrors(cudaFree(outputVals));
  checkCudaErrors(cudaFree(outputPos));

  return 0;
}
