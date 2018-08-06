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
#include "float.h"

#include "stdio.h"

#define THREADS_PER_BLOCK 1024

// utility for debugging
__global__ void printIntArray(int *d_array, int size) 
{
	if (threadIdx.x != 0) return; 

	for (int i = 0; i < size; i++)
	{   
		printf("%d\t:\t%d\n", i, d_array[i]); 
	}
}

__global__ void printUnsignedIntArray(unsigned int *d_array, int size) 
{
	if (threadIdx.x != 0) return; 

	for (int i = 0; i < size; i++)
	{
		if (d_array[i] != 0)
			printf("%d\t:\t%u\n", i, d_array[i]); 
	}
}

// This function assumes that the blocks 
// and the grids are 1-D and 
// blockDim.x is a power of 2. 
__global__ void g_reduce_max(float* d_out, 
		                   const float* const d_in, 
		                   const size_t size) 
{
	extern __shared__ float sdata[]; 

	int myId = blockDim.x * blockIdx.x + threadIdx.x; 
	int tid = threadIdx.x; 

	float value = (myId < size)? d_in[myId] : FLT_MIN; 
    sdata[tid] = value; 	
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
	{   
		if (tid < s) {
			sdata[tid] = max(sdata[tid], sdata[tid + s]); 
		}

		__syncthreads(); 
	}

	if (tid == 0)
	{
		d_out[blockIdx.x] = sdata[0]; 
	}
}

// reduce min. This is essentially a duplicate of g_reduce_max. 
// Is there a way to pass in a function pointer? 
__global__ void g_reduce_min(float* d_out, 
		                     const float* const d_in, 
		                     const size_t size) 
{
	extern __shared__ float sdata[]; 

	int myId = blockDim.x * blockIdx.x + threadIdx.x; 
	int tid = threadIdx.x; 

	float value = (myId < size)? d_in[myId] : FLT_MAX; 
    sdata[tid] = value; 	
	__syncthreads(); 

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
	{
		if (tid < s) {
			sdata[tid] = min(sdata[tid], sdata[tid + s]); 
		}
		__syncthreads(); 
	}

	if (tid == 0)
	{
		d_out[blockIdx.x] = sdata[0]; 
	}
}

// Helper function to find the smallest power of 2 bigger than an 
// unsigned int input. 
unsigned nextPow2(unsigned n) {
	if (!(n & (n - 1))) return n; 
	unsigned count = 0; 
	while (n != 0) 
	{
		n >>= 1; 
        count++; 
	}

	return 1 << count; 
}

// optn == 0 min
// optn != 0 max
// function assumes size <= 2^20
float reduce_extrema(const float* const d_in, const size_t size, int optn) {
	unsigned threadsPerBlock = THREADS_PER_BLOCK; 
	unsigned numGrids = (size + threadsPerBlock - 1) / threadsPerBlock; 
	const dim3 blockSize(threadsPerBlock, 1, 1); 
	const dim3 gridSize(numGrids, 1, 1); 

	float* d_intermediate;
    float* d_result; 	
	checkCudaErrors(cudaMalloc((void **) &d_intermediate, 
				              numGrids * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **) &d_result, 
				              sizeof(float)));
    
	size_t sharedMemSize = threadsPerBlock * sizeof(float); 

	if (optn == 0) 
	{
    	g_reduce_min<<<gridSize, blockSize, sharedMemSize>>>(d_intermediate, 
				                                             d_in, size);
	} else 
	{
    	g_reduce_max<<<gridSize, blockSize, sharedMemSize>>>(d_intermediate, 
				                                             d_in, size);
	}


	// call g_reduce a second time to process the results from 
	// each block of the previous call.
    unsigned paddedNumThreads = nextPow2(numGrids); 	
	sharedMemSize = paddedNumThreads * sizeof(float); 
    
	if (optn == 0) 
	{
    	g_reduce_min<<<1, paddedNumThreads, sharedMemSize>>>(d_result, 
				                                             d_intermediate, 
				                                             numGrids); 
	} else 
	{
    	g_reduce_max<<<1, paddedNumThreads, sharedMemSize>>>(d_result, 
				                                             d_intermediate, 
				                                             numGrids); 
	}

	float h_result; 
	checkCudaErrors(cudaMemcpy(&h_result, d_result, sizeof(float), 
				              cudaMemcpyDeviceToHost)); 

	checkCudaErrors(cudaFree(d_intermediate)); 
	checkCudaErrors(cudaFree(d_result)); 

	return h_result; 
}

__global__ void simple_hdr_histo(unsigned int *d_bins, const float *d_in, 
		                         const int numBins, 
		                         float min_val, float range) 
{
	int myId = threadIdx.x + blockDim.x * blockIdx.x; 
	float myItem = d_in[myId]; 
	unsigned int myBin = min((unsigned int)(numBins - 1), 
			                 (unsigned int)((myItem - min_val) / range * numBins)); 
	atomicAdd(&(d_bins[myBin]), 1); 
}

// Simple implementation of Blelloch Scan. 
// This function assumes the number of blocks is 1. 
// In another word, gridDim.x == 1
// Addtionally, it assumes the number of threads per block
// is a power of 2. 
// The shared data required is of size sizeof(int) * blockDim.x . 
__global__ void excl_prefix_sum(unsigned int* const d_cdf, 
		                        const unsigned int* const d_bins, 
		                        const size_t size) 
{
	extern __shared__ unsigned int idata[];

    // because blockIdx.x == 0, we do not need blockDim offset. 
	int tid = threadIdx.x; 	

	idata[tid] = (tid < size)? d_bins[tid] : 0;
    __syncthreads(); 
    
	// summing up
    for (unsigned int s = 1; s < blockDim.x; s <<= 1)
	{
		unsigned int temp = 0; 
		if ((tid + 1) % (2 * s) == 0)
			temp = idata[tid - s]; 
		__syncthreads();

		if ((tid + 1) % (2 * s) == 0)
			idata[tid] += temp; 
		__syncthreads(); 
	}	
	
	__syncthreads(); 
	// set max idx to identity
	if (tid == blockDim.x - 1) 
	{
		idata[tid] = 0;
	}

	__syncthreads(); 

	// downward sweep
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		int temp1 = 0; 
		int temp2 = 0;

		if ((tid + 1) % (2 * s) == 0)
		{
			temp1 = idata[tid]; 
			temp2 = idata[tid - s]; 
		}
		__syncthreads(); 

		if ((tid + 1) % (2 * s) == 0)
		{
			idata[tid] += temp2; 
			idata[tid - s] = temp1; 
		}
		__syncthreads(); 
	}

	if (tid < size)
	{
		d_cdf[tid] = idata[tid]; 
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
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

  // d_logLuninance is more like a 1-d structure. So we flatten everything. 
  // int threadsPerBlock = 1024; 
  size_t size = numRows * numCols;

  // Step 1 compute the minimum and maximum. 
  
  float maxLum = reduce_extrema(d_logLuminance, size, 1);
  float minLum = reduce_extrema(d_logLuminance, size, 0);  

  printf("GPU min: %f\n", minLum); 
  printf("GPU max: %f\n", maxLum); 

  // Step 2 compute the difference to find the range
  float range = maxLum - minLum;

  // Step 3 generate histogram. 
  int numThreads = THREADS_PER_BLOCK; 
  int numBlocks = (size + numThreads - 1) / numThreads; 
  unsigned int* d_bins; 
  checkCudaErrors(cudaMalloc((void **)&d_bins, sizeof(unsigned int) * numBins));
  cudaMemset(d_bins, 0, sizeof(unsigned int) * numBins); 
  simple_hdr_histo<<<numBlocks, numThreads>>>(d_bins, d_logLuminance, numBins, 
		                                      minLum, range);

  // Step 4 the exclusive scan - assume numBins is a power of 2.
  excl_prefix_sum<<<1, numBins, sizeof(unsigned int) * numBins>>>(d_cdf, 
    	                                                          d_bins, 
    															  numBins); 
  // Cleaning up
  checkCudaErrors(cudaFree(d_bins)); 
}
