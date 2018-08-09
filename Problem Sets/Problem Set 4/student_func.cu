// Udacity HW 4
// Radix Sorting

#include <thrust/host_vector.h>
#include "string.h"
#include "utils.h"

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
	For example [0 0 1 1 0 0 1]
		->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

// This function assumes that there are two bins!
// The mask is used to pick the bit to compute the
// index of the bin.
__global__ void simple_hist(unsigned int *const d_bins,
			    const unsigned int *const d_in, const size_t size,
			    unsigned int mask, unsigned int pos) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= size) return;
    unsigned int binIdx = ((d_in[i] & mask) >> pos);
    atomicAdd(&(d_bins[binIdx]), 1);
}

// Exclusive scan - Naive Hillis and Steele.
// Call with gridSize 1
// and shared memory size blockSize * sizeof(unsigned int)
__global__ void naive_scan(unsigned int *const d_out,
			   const unsigned int *const d_in, const size_t size) {
    extern __shared__ unsigned int sdata[];
    int tid = threadIdx.x;
    sdata[tid] = (tid >= 1 && tid < size) ? d_in[tid - 1] : 0;

    for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
	unsigned int a = 0;
	unsigned int b = 0;
	if (tid >= 2 * s) {
	    a = sdata[tid - s];
	    b = sdata[tid];
	}
	__syncthreads();

	if (tid >= 2 * s) sdata[tid] = a + b;
	__syncthreads();
    }

    if (tid >= size) return;
    d_out[tid] = sdata[tid];
}

void your_sort(unsigned int *const d_inputVals, unsigned int *const d_inputPos,
	       unsigned int *const d_outputVals,
	       unsigned int *const d_outputPos, const size_t numElems) {
    // Temporary testing area...
	// Setting up data
	size_t testSize = 8; 
	size_t testMemSize = testSize * sizeof(unsigned int); 
	unsigned int *h_test_in = (unsigned int *)malloc(testMemSize); 
	unsigned int *h_test_out = (unsigned int *)malloc(testMemSize); 
	for (unsigned int i = 0; i < testSize; i++) {
		h_test_in[i] = i + 1; 
	}
	// CPU reference result
	h_test_out[0] = 0; 
	for (unsigned int i = 1; i < testSize; i++) {
		h_test_out[i] = h_test_in[i- 1] + h_test_out[i - 1]; 
	}

	// GPU test
	unsigned int *d_test_in; 
	unsigned int *d_test_out; 
	unsigned int *h_gpu_out; 
	checkCudaErrors(cudaMalloc((void **)&d_test_in, testMemSize));	
	checkCudaErrors(cudaMalloc((void **)&d_test_out, testMemSize));
    h_gpu_out = (unsigned int *)malloc(testMemSize); 

	checkCudaErrors(cudaMemcpy(d_test_in, h_test_in, testMemSize, cudaMemcpyHostToDevice));
		
	naive_scan<<<1, 8, 8 * sizeof(unsigned int)>>>(d_test_out, d_test_in, testSize); 
	
	checkCudaErrors(cudaMemcpy(h_gpu_out, d_test_out, testMemSize, cudaMemcpyDeviceToHost));

	for (unsigned int i = 0; i < testSize; i++) {
		unsigned int gpu = h_gpu_out[i]; 
		unsigned int cpu = h_test_out[i]; 
		if (gpu != cpu) {
			printf("scan gpu[%u]:%u\t cpu[%u]%u\n", i, gpu, i, cpu); 
		}
	}

	// cleaning up
	free(h_test_in); 
	free(h_test_out); 
	free(h_gpu_out); 
	checkCudaErrors(cudaFree(d_test_in)); 
	checkCudaErrors(cudaFree(d_test_out)); 

	// end of testing area...
    const int numBits = 1;
    const int numBins = 1 << numBits;

    size_t memSize = sizeof(unsigned int) * numElems;

    // CPU code for testing
    unsigned int *binHistogram = new unsigned int[numBins];
    unsigned int *binScan = new unsigned int[numBins];

    unsigned int *vals_src = new unsigned int[numElems];
    unsigned int *pos_src = new unsigned int[numElems];

    unsigned int *vals_dst = new unsigned int[numElems];
    unsigned int *pos_dst = new unsigned int[numElems];

    checkCudaErrors(
	cudaMemcpy(vals_src, d_inputVals, memSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(
	cudaMemcpy(pos_src, d_inputPos, memSize, cudaMemcpyDeviceToHost));

    // Temporary CPU arrays
    unsigned int *h_binHistogram = new unsigned int[numBins];
    unsigned int *h_binScan = new unsigned int[numBins];

    // GPU arrays
    unsigned int *d_binHistogram;
    unsigned int *d_binScan;
    size_t g_memSize = sizeof(unsigned int) * numBins;
    checkCudaErrors(cudaMalloc((void **)&d_binHistogram, g_memSize));
    checkCudaErrors(cudaMalloc((void **)&d_binScan, g_memSize));

    // GPU temporary pointers
    unsigned int *d_vals_src = d_inputVals;
    unsigned int *d_pos_src = d_inputPos;
    unsigned int *d_vals_dst = d_outputVals;
    unsigned int *d_pos_dst = d_outputPos;

    // GPU kernel dimensions
    unsigned int numThreads = 1024;
    unsigned int numBlocks = (numElems + numThreads - 1) / numThreads;
    dim3 blockSize(numThreads, 1, 1);
    dim3 gridSize(numBlocks, 1, 1);

    // a simple radix sort - only guaranteed to work for numBits that are
    // multiples of 2
    // main loop
    for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
	unsigned int mask = (numBins - 1) << i;

	memset(binHistogram, 0,
	       sizeof(unsigned int) * numBins);  // zero out the bins
	memset(binScan, 0, sizeof(unsigned int) * numBins);  // zero out the
							     // bins

	// Reset GPU arrays
	checkCudaErrors(cudaMemset(d_binHistogram, 0, g_memSize));
	checkCudaErrors(cudaMemset(d_binScan, 0, g_memSize));

	// CPU histogram
	// perform histogram of data & mask into bins
	for (unsigned int j = 0; j < numElems; ++j) {
	    unsigned int bin = (vals_src[j] & mask) >> i;
	    binHistogram[bin]++;
	}

	// GPU histogram
	simple_hist<<<gridSize, blockSize>>>(d_binHistogram, d_vals_src,
					     numElems, mask, i);
	checkCudaErrors(cudaMemcpy(h_binHistogram, d_binHistogram, g_memSize,
				   cudaMemcpyDeviceToHost));
	// check GPU results
	for (unsigned int j = 0; j < numBins; j++) {
	    if (h_binHistogram[j] != binHistogram[j]) {
		printf("GPU hist[%d]:%u\tCPU hist[%d]:%u\n", j,
		       h_binHistogram[j], j, binHistogram[j]);
	    }
	}

	// CPU scan
	// perform exclusive prefix sum (scan) on binHistogram to get starting
	// location for each bin
	for (unsigned int j = 1; j < numBins; ++j) {
	    binScan[j] = binScan[j - 1] + binHistogram[j - 1];
	}

	// GPU scan
	/*__global__ void naive_scan(unsigned int* const d_out,
			     const unsigned int* const d_in,
			     const size_t size)*/

	// Gather everything into the correct location
	// need to move vals and positions
	for (unsigned int j = 0; j < numElems; ++j) {
	    unsigned int bin = (vals_src[j] & mask) >> i;
	    vals_dst[binScan[bin]] = vals_src[j];
	    pos_dst[binScan[bin]] = pos_src[j];
	    binScan[bin]++;
	}

	// swap the buffers (pointers only)
	std::swap(vals_dst, vals_src);
	std::swap(pos_dst, pos_src);
    }

    // we did an even number of iterations, need to copy from input buffer into
    // output std::copy(inputVals, inputVals + numElems, outputVals);
    // std::copy(inputPos, inputPos + numElems, outputPos);

    delete[] binHistogram;
    delete[] binScan;
}
