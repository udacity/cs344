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
        if (tid >= s) {
            a = sdata[tid - s];
            b = sdata[tid];
        }
        __syncthreads();

        if (tid >= s) sdata[tid] = a + b;
        __syncthreads();
    }

    if (tid >= size) return;
    d_out[tid] = sdata[tid];
}

size_t nextPow2(size_t x) {
    --x; 
    x |= x >> 1; 
    x |= x >> 2; 
    x |= x >> 4; 
    x |= x >> 8; 
    x |= x >> 16; 
    x |= x >> 32; 
    return ++x; 
}

// naive inclusive scan across blocks
__global__ void naive_inclusive_scan_per_block(unsigned int *const d_out,
        unsigned int *const d_intermediate, 
        const unsigned int *const d_in, const size_t size) {
    extern __shared__ unsigned int sdata[];
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    int tid = threadIdx.x;
    sdata[tid] = (idx < size) ? d_in[idx] : 0;

	__syncthreads(); 
    
	for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
        unsigned int a = 0;
        unsigned int b = 0;
        if (tid >= s) {
            a = sdata[tid - s];
            b = sdata[tid];
        }
        __syncthreads();

        if (tid >= s) sdata[tid] = a + b;
        __syncthreads();
    }

    __syncthreads(); 

    if (tid >= size) return;

    d_out[idx] = sdata[tid];

    if (tid == blockDim.x - 1) {
        d_intermediate[blockIdx.x] = sdata[tid]; 
    }
}

__global__ void scatter_for_multiBlockScan(unsigned int *const d_out, 
        const unsigned int *const d_in, 
        const unsigned int *const d_interm_accum, 
        const size_t size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= size) return; 
    unsigned int self = d_in[idx];
    unsigned int inc = d_interm_accum[blockIdx.x];
    d_out[idx] += inc - self; 
}


__global__ void printUnsignedIntArray(const unsigned int *const arr, 
        size_t size) {
    if (threadIdx.x == 0) {
        for (size_t i = 0; i < size; i++) {
            printf("arr[%lu]:\t%u\n", i, arr[i]); 
        }
    }
}

// Exclusive scan for sizes <= 2^20
void multiBlockScan(unsigned int *const d_out, const unsigned int *const d_in, 
        const size_t size) {
    size_t numThreads = 1024; 
    size_t numBlocks = (nextPow2(size) + numThreads - 1) / numThreads;
    
    // allocate temporary GPU arrays
    unsigned int *d_intermediate; 
    unsigned int *d_interm_accum; 
    checkCudaErrors(cudaMalloc((void **)&d_intermediate, sizeof(unsigned int) * numBlocks)); 
    checkCudaErrors(cudaMalloc((void **)&d_interm_accum, sizeof(unsigned int) * numBlocks)); 

    // inclusive scan to produce prefix sum in each block
    naive_inclusive_scan_per_block<<<numBlocks, numThreads, sizeof(unsigned int) * numThreads>>>
        (d_out, d_intermediate, d_in, size);
    //printUnsignedIntArray<<<1,1>>>(d_out, size); 
    
    // exclusive scan to produce prefix sum of d_intermediate
    naive_scan<<<1, numBlocks, sizeof(unsigned int) * numBlocks>>>
        (d_interm_accum, d_intermediate, numBlocks);


    // add results of d_intermediate back into each block 
    // and subtract d_in[0] from all elements to get the final results.
    scatter_for_multiBlockScan<<<numBlocks, numThreads>>>
        (d_out, d_in, d_interm_accum, size);
   
    checkCudaErrors(cudaFree(d_intermediate)); 
    checkCudaErrors(cudaFree(d_interm_accum)); 
}

__global__ void map_to_binFlags(unsigned int *const d_binFlags, 
        const unsigned int *const d_in, unsigned int mask, 
        unsigned int pos, unsigned int binIdx, const size_t size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx >= size) return; 
    unsigned int bin = ((d_in[idx] & mask) >> pos);
    d_binFlags[idx] = (bin == binIdx)? 1 : 0; 
}

__global__ void offset_by_base(unsigned int *const d_out, 
        const unsigned int *const d_binScan,
        unsigned int binIdx, 
        const size_t size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx >= size) return; 
    unsigned amount = d_binScan[binIdx]; 
    d_out[idx] = d_out[idx] + amount; 
}

__global__ void reorder(unsigned int *d_out, const unsigned int *const d_in, 
        const unsigned int *const d_flags, const unsigned int *d_addr, 
        const size_t size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; 
    if (idx >= size) return; 
    if (d_flags[idx]) {
        d_out[d_addr[idx]] = d_in[idx]; 
    } 
}

void swap_device_ptr(unsigned int **d1, unsigned int **d2) {
    unsigned int *temp = *d1; 
    *d1 = *d2; 
    *d2 = temp; 
}

void your_sort(unsigned int *const d_inputVals, unsigned int *const d_inputPos,
        unsigned int *const d_outputVals,
        unsigned int *const d_outputPos, const size_t numElems) {
    const int numBits = 2;
    const int numBins = 1 << numBits;

    size_t memSize = sizeof(unsigned int) * numElems;

    // GPU arrays
    unsigned int *d_binHistogram;
    unsigned int *d_binScan;
    size_t g_memSize = sizeof(unsigned int) * numBins;
    checkCudaErrors(cudaMalloc((void **)&d_binHistogram, g_memSize));
    checkCudaErrors(cudaMalloc((void **)&d_binScan, g_memSize));
    unsigned int *d_binFlags; // Indicate if a particular value belongs to 
                              // a certain bin
    unsigned int *d_binAddress; 
    checkCudaErrors(cudaMalloc((void **)&d_binFlags, memSize));
    checkCudaErrors(cudaMalloc((void **)&d_binAddress, memSize));

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

        // Reset GPU arrays
        checkCudaErrors(cudaMemset(d_binHistogram, 0, g_memSize));
        checkCudaErrors(cudaMemset(d_binScan, 0, g_memSize));
        checkCudaErrors(cudaMemset(d_binFlags, 0, memSize)); 
        checkCudaErrors(cudaMemset(d_binAddress, 0, memSize)); 

        // GPU histogram
        simple_hist<<<gridSize, blockSize>>>(d_binHistogram, d_vals_src,
                numElems, mask, i);

        // GPU scan
        naive_scan<<<1, numBins, numBins * sizeof(unsigned int)>>>(
                d_binScan, d_binHistogram, numBins);

        // GPU computation to remap the elements
        for (unsigned int j = 0; j < numBins; ++j) {
            map_to_binFlags<<<gridSize, blockSize>>>(d_binFlags, d_vals_src, 
                                                     mask, i, j, numElems); 
            multiBlockScan(d_binAddress, d_binFlags, numElems); 
            
            offset_by_base<<<gridSize, blockSize>>>(d_binAddress, d_binScan, 
                    j, numElems);
            
            reorder<<<gridSize, blockSize>>>(d_vals_dst, d_vals_src, 
                    d_binFlags, d_binAddress, numElems); 
            
            reorder<<<gridSize, blockSize>>>(d_pos_dst, d_pos_src, 
                    d_binFlags, d_binAddress, numElems); 
            
            // reset d_binFlags after processing for the current
            // bin. 
            checkCudaErrors(cudaMemset(d_binFlags, 0, memSize)); 
            checkCudaErrors(cudaMemset(d_binAddress, 0, memSize)); 
        }

        unsigned int *h_vals_dst = (unsigned int*)malloc(memSize); 
        checkCudaErrors(cudaMemcpy(h_vals_dst, d_vals_dst, 
                    memSize, cudaMemcpyDeviceToHost));

        // swap device pointers
        swap_device_ptr(&d_vals_dst, &d_vals_src);
        swap_device_ptr(&d_pos_dst, &d_pos_src); 
    }

    // we did an even number of iterations, need to copy from input buffer into
    // output 
	checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, memSize, cudaMemcpyDeviceToDevice)); 
	checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, memSize, cudaMemcpyDeviceToDevice)); 

	checkCudaErrors(cudaFree(d_binHistogram));
   	checkCudaErrors(cudaFree(d_binScan)); 
    checkCudaErrors(cudaFree(d_binFlags)); 
	checkCudaErrors(cudaFree(d_binAddress)); 
}
