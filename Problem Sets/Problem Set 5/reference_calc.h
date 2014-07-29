#ifndef REFERENCE_H__
#define REFERENCE_H__

//Reference Histogram calculation

void reference_calculation(const unsigned int* const vals,
                           unsigned int* const histo,
                           const size_t numBins,
                           const size_t numElems);

#endif