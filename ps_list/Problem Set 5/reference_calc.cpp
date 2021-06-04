#include <cstdlib>
//Reference Histogram calculation

void reference_calculation(const unsigned int* const vals,
                           unsigned int* const histo,
                           const size_t numBins,
                           const size_t numElems)

{
  //zero out bins
  for (size_t i = 0; i < numBins; ++i)
    histo[i] = 0;

  //go through vals and increment appropriate bin
  for (size_t i = 0; i < numElems; ++i)
    histo[vals[i]]++;
}
