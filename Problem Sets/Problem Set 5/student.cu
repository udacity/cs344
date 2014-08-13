/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include "utils.h"
#include "reference.cpp"

#define elementsPerThread 64

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const int numVals,
               const int numBins)
{
  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + tid;
    
  // create sharedMemory histogram and initialize to 0
  extern __shared__ unsigned int localHistogram[];
  for (int i = tid; i < numBins; i += blockDim.x) {
    localHistogram[i] = 0;
  }
 
  __syncthreads();
  
  // use offset to ensure coalescing read from global memory
  // each thread reads with a stride of the offset
  // parallel threads read one coalesced piece of memory
  int offset = numVals / elementsPerThread;
  for(int i = 0; i < elementsPerThread; ++i)
  {
    atomicAdd(&(localHistogram[vals[index + offset*i]]),1);
  }
  
  __syncthreads();
  
  for (int i = tid; i < numBins; i += blockDim.x) {
    atomicAdd(&histo[i], localHistogram[i]);
  }
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  int powerOfTwo = 4;
  int threadsPerBlock = 1024/powerOfTwo;
  int blocks = (numElems + (threadsPerBlock*elementsPerThread) - 1) / (threadsPerBlock*elementsPerThread);
    
  yourHisto<<<blocks, threadsPerBlock, (numBins + 250) * sizeof(unsigned int)>>>(d_vals, d_histo, numElems, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

