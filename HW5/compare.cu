#include <fstream>
#include <cassert>

#include "reference.cpp"
#include "utils.h"

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>

const unsigned int numBins = 1024;
const unsigned int numElems = 10000 * numBins;
const float stddev = 100.f;

int main(int argc, char **argv)
{
  if (argc != 3) {
    std::cerr << "Usage: ./compare file mean" << std::endl;
    exit(1);
  }

  std::ifstream ifs(argv[1], std::ios::in | std::ios::binary);

  if (!ifs.good()) {
    std::cerr << "Couldn't open " << argv[1] << std::endl;
    exit(1);
  }

  ifs.seekg(0, std::ios::end);
  size_t bytes = ifs.tellg();
  ifs.seekg(0);

  unsigned int numBinsInFile = bytes / sizeof(unsigned int);

  assert(numBinsInFile == numBins);

  unsigned int* student_histo = new unsigned int[numBins];

  ifs.read(reinterpret_cast<char *>(student_histo), bytes);
  ifs.close();

  //now we need to recreate the results using the same mean
  int mean = atoi(argv[2]);
  thrust::minstd_rand rng;

  thrust::random::experimental::normal_distribution<float> normalDist((float)mean, stddev);

  unsigned int *vals = new unsigned int[numElems];
  unsigned int *histo = new unsigned int[numBins];

  for (size_t i = 0; i < numElems; ++i) {
    vals[i] = min(max((int)normalDist(rng), 0), numBins - 1);
  }

  //generate reference for the given mean
  reference_calculation(vals, histo, numBins, numElems);

  //Now do the comparison
  checkResultsExact(histo, student_histo, numBins);

  delete[] vals;
  delete[] histo;
  delete[] student_histo;

  return 0;
}
