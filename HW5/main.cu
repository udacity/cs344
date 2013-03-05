#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <fstream>
#include "utils.h"
#include "timer.h"
#include <cstdio>
#include <sys/time.h>

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>

void computeHistogram(const unsigned int* d_vals,
                      unsigned int* const d_histo,
                      const unsigned int numBins,
                      const unsigned int numElems);

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cerr << "You must supply an output file" << std::endl;
    exit(1);
  }

  const unsigned int numBins = 1024;
  const unsigned int numElems = 10000 * numBins;
  const float stddev = 100.f;

  unsigned int *vals = new unsigned int[numElems];
  unsigned int *histo = new unsigned int[numBins];

  timeval tv;
  gettimeofday(&tv, NULL);

  srand(tv.tv_usec);

  //make the mean unpredictable, but close enough to the middle
  //so that timings are unaffected
  unsigned int mean = rand() % 100 + 462;

  //Output mean so that grading can happen with the same inputs
  std::cout << "e57__MEAN__f82 " << mean << std::endl;

  thrust::minstd_rand rng;

  thrust::random::experimental::normal_distribution<float> normalDist((float)mean, stddev);

  for (size_t i = 0; i < numElems; ++i) {
    vals[i] = min(max((int)normalDist(rng), 0), numBins - 1);
  }

  unsigned int *d_vals, *d_histo;

  GpuTimer timer;

  checkCudaErrors(cudaMalloc(&d_vals,    sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc(&d_histo,   sizeof(unsigned int) * numBins));
  checkCudaErrors(cudaMemset(d_histo, 0, sizeof(unsigned int) * numBins));

  checkCudaErrors(cudaMemcpy(d_vals, vals, sizeof(unsigned int) * numElems, cudaMemcpyHostToDevice));

  timer.Start();
  computeHistogram(d_vals, d_histo, numBins, numElems);
  timer.Stop();
  int err = printf("e57__TIMING__f82 %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  unsigned int *h_gpu = new unsigned int[numBins];

  checkCudaErrors(cudaMemcpy(h_gpu, d_histo, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

  std::ofstream ofs(argv[1], std::ios::out | std::iostream::binary);

  ofs.write(reinterpret_cast<char *>(h_gpu), numBins * sizeof(unsigned int));
  ofs.close();

  delete[] h_gpu;
  delete[] vals;
  delete[] histo;

  cudaFree(d_vals);
  cudaFree(d_histo);

  return 0;
}
