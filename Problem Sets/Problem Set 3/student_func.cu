/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

__global__ void reduceLuminance(const float* const d_logLuminance, bool isMax, float *result, int inputSize)
{
    //  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];
    
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;
    
    // load shared mem from global mem
    if(myId < inputSize)
    {
        sdata[tid] = d_logLuminance[myId];
    }
    __syncthreads();            // make sure entire block is loaded!
    
    // do reduction in shared mem
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s && (myId+s) < inputSize)
        {
            float val1=sdata[tid];
            float val2=sdata[tid + s];
            sdata[tid] = isMax?(val1 > val2 ? val1 : val2) : (val1 < val2 ? val1 : val2);
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }
    
    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        result[blockIdx.x] = sdata[0];
    }
}

__global__ void assignHistogram(int *d_bins, const float *d_in, int numBins, float lumRange, float lumMin, int inputSize)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    
    if(myId<inputSize)
    {
        //bin = (lum[i] - lumMin) / lumRange * numBins
        int myBin = (d_in[myId] - lumMin) / lumRange * numBins;
        atomicAdd(&(d_bins[myBin]), 1);
    }
}


__global__ void scanHistogram(const int* const d_bins, unsigned int *result, int inputSize)
{
    //  // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ int shareddata[];
    
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;
    
    // load shared mem from global mem
    if(myId < inputSize)
    {
        shareddata[tid] = d_bins[myId];
    }
    __syncthreads();            // make sure entire block is loaded!
    
    //Step hillis / Steele
    for (int step = 1; step<inputSize;step <<= 1)
    {
        if (tid >= step)
        {
            shareddata[tid] += shareddata[tid - step];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }
    
    // every thread writes result for this block back to global mem
    if(myId < inputSize)
    {
        result[myId] = shareddata[tid];
    }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
unsigned int* const d_cdf,
float &min_logLum,
float &max_logLum,
const size_t numRows,
const size_t numCols,
const size_t numBins)
{
    //TODO
    /*Here are the steps you need to implement
     1) find the minimum and maximum value in the input logLuminance channel
     store in min_logLum and max_logLum
     2) subtract them to find the range
     3) generate a histogram of all the values in the logLuminance channel using
     the formula: bin = (lum[i] - lumMin) / lumRange * numBins
     4) Perform an exclusive scan (prefix sum) on the histogram to get
     the cumulative distribution of luminance values (this should go in the
     incoming d_cdf pointer which already has been allocated for you)       */
    
    // Two step reduce on one dimension
    const int inputSize = numRows * numCols;
    const int maxThreadsPerBlock = 1024;
    unsigned int threads = maxThreadsPerBlock;
    const int blocks = (inputSize / maxThreadsPerBlock)+2; //more blocks to avoid int round loss
    
    const int sharedMemorySize_1 = threads * sizeof(float);
    const int sharedMemorySize_2 = blocks * sizeof(float);
    
    float *d_intermediate, *d_result;
    checkCudaErrors(cudaMalloc(&d_intermediate, sizeof(float) * blocks));
    checkCudaErrors(cudaMalloc(&d_result, sizeof(float)));
    
    // Step 1: min luminance
    reduceLuminance<<<blocks, threads, sharedMemorySize_1>>>(d_logLuminance, false, d_intermediate, inputSize);
    reduceLuminance<<<1, blocks, sharedMemorySize_2>>>(d_intermediate, false, d_result, blocks);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&min_logLum, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Step 1: max luminance
    reduceLuminance<<<blocks, threads, sharedMemorySize_1>>>(d_logLuminance, true, d_intermediate, inputSize);
    reduceLuminance<<<1, blocks, sharedMemorySize_2>>>(d_intermediate, true, d_result, blocks);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&max_logLum, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    
    // Step 2: luminance range
    float range = max_logLum - min_logLum;
    
    // Step 3: histogram
    int *d_bins;
    checkCudaErrors(cudaMalloc(&d_bins, sizeof(int) * numBins));
    checkCudaErrors(cudaMemset(d_bins, 0, sizeof(int) * numBins));
    assignHistogram<<<(inputSize/threads) +1, threads>>>(d_bins, d_logLuminance, numBins, range, min_logLum, inputSize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    // Step 4: Exclusive Scan on d_bins
    threads = 2;
    while(threads < numBins)
    {
        threads <<= 1;
    }
    
    scanHistogram<<<1, threads, sizeof(unsigned int) * numBins>>>(d_bins, d_cdf, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    //free resources
    
    cudaFree(d_intermediate);
    cudaFree(d_result);
    cudaFree(d_bins);
}
