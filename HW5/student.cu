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


__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel

  //if you want to use/launch more than one kernel,
  //feel free
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //Again we have provided a reference calculation for you
  //to help with debugging.  Uncomment the code below
  //to activate it.
  //REMEMBER TO COMMENT IT OUT BEFORE GRADING
  //otherwise your code will be too slow

  /*unsigned int *h_vals = new unsigned int[numElems];
  unsigned int *h_histo = new unsigned int[numBins];

  checkCudaErrors(cudaMemcpy(h_vals, d_vals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));

  reference_calculation(h_vals, h_histo, numBins, numElems);

  unsigned int *your_histo = new unsigned int[numBins];
  checkCudaErrors(cudaMemcpy(your_histo, d_histo, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

  checkResultsExact(h_histo, your_histo, numBins); 

  delete[] h_vals;
  delete[] h_histo;
  delete[] your_histo;*/
}
