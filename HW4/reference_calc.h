#ifndef REFERENCE_H__
#define REFERENCE_H__


//A simple un-optimized reference radix sort calculation
//Only deals with power-of-2 radices


void reference_calculation(unsigned int* inputVals,
                           unsigned int* inputPos,
                           unsigned int* outputVals,
                           unsigned int* outputPos,
                           const size_t numElems);
#endif