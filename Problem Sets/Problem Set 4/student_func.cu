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
    // Temporary testing area...
    printf("numElems %lu\n", numElems); 
    // Setting up data
    size_t testSize = 512;
    size_t testMemSize = testSize * sizeof(unsigned int);
    unsigned int *h_test_in = (unsigned int *)malloc(testMemSize);
    unsigned int *h_test_out = (unsigned int *)malloc(testMemSize);
    for (unsigned int i = 0; i < testSize; i++) {
        h_test_in[i] = 1;
    }
    // CPU reference result
    h_test_out[0] = 0;
    for (unsigned int i = 1; i < testSize; i++) {
        h_test_out[i] = h_test_in[i - 1] + h_test_out[i - 1];
    }

    // GPU test
    unsigned int *d_test_in;
    unsigned int *d_test_out;
    unsigned int *h_gpu_out;
    checkCudaErrors(cudaMalloc((void **)&d_test_in, testMemSize));
    checkCudaErrors(cudaMalloc((void **)&d_test_out, testMemSize));
    h_gpu_out = (unsigned int *)malloc(testMemSize);

    checkCudaErrors(
            cudaMemcpy(d_test_in, h_test_in, testMemSize, cudaMemcpyHostToDevice));

    /*naive_scan<<<1, 8, 8 * sizeof(unsigned int)>>>(d_test_out, d_test_in,
            testSize);*/

    multiBlockScan(d_test_out, d_test_in, testSize); 

    checkCudaErrors(
            cudaMemcpy(h_gpu_out, d_test_out, testMemSize, cudaMemcpyDeviceToHost));

    for (unsigned int i = 0; i < testSize; i++) {
        unsigned int gpu = h_gpu_out[i];
        unsigned int cpu = h_test_out[i];
        if (gpu != cpu) {
            printf("multiblock scan gpu[%u]:%u\t cpu[%u]%u\n", i, gpu, i, cpu);
        }
    }

    unsigned int *d_intermediate; 
    checkCudaErrors(cudaMalloc((void **)&d_intermediate, 4 * sizeof(unsigned int))); 
    //naive_inclusive_scan_per_block<<<4, 2, sizeof(unsigned int) * 2>>>
    //    (d_test_out, d_intermediate, d_test_in, testSize); 

    // Testing reordering
    /*map_to_binFlags<<<gridSize, blockSize>>>(d_binFlags, d_vals_src, 
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
    checkCudaErrors(cudaMemset(d_binAddress, 0, memSize)); */

    // cleaning up
    free(h_test_in);
    free(h_test_out);
    free(h_gpu_out);
    checkCudaErrors(cudaFree(d_test_in));
    checkCudaErrors(cudaFree(d_test_out));

    // end of testing area...

    const int numBits = 4;
    const int numBins = 1 << numBits;

    size_t memSize = sizeof(unsigned int) * numElems;

    // CPU code for testing
    unsigned int *binHistogram = new unsigned int[numBins];
    unsigned int *binScan = new unsigned int[numBins];

    unsigned int *vals_src = new unsigned int[numElems];
    unsigned int *pos_src = new unsigned int[numElems];

    unsigned int *vals_dst = new unsigned int[numElems];
    unsigned int *pos_dst = new unsigned int[numElems];

    unsigned int *binFlags = new unsigned int[numElems]; 
    unsigned int *binAddress = new unsigned int[numElems]; 
    
    checkCudaErrors(
            cudaMemcpy(vals_src, d_inputVals, memSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(
            cudaMemcpy(pos_src, d_inputPos, memSize, cudaMemcpyDeviceToHost));

    // Temporary CPU arrays
    unsigned int *h_binHistogram = new unsigned int[numBins];
    unsigned int *h_binScan = new unsigned int[numBins];
    unsigned int *h_binFlags = new unsigned int[numElems]; 
    unsigned int *h_binAddress = new unsigned int[numElems]; 

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
        printf("XXXXXXXXXX mask at pos %u XXXXXXXXXXX\n", i); 
        unsigned int mask = (numBins - 1) << i;

        memset(binHistogram, 0,
                sizeof(unsigned int) * numBins);  // zero out the bins
        memset(binScan, 0, sizeof(unsigned int) * numBins);  // zero out the
        // bins
        memset(binFlags, 0, memSize); 
        memset(binAddress, 0, memSize); 

        // Reset GPU arrays
        checkCudaErrors(cudaMemset(d_binHistogram, 0, g_memSize));
        checkCudaErrors(cudaMemset(d_binScan, 0, g_memSize));
        checkCudaErrors(cudaMemset(d_binFlags, 0, memSize)); 
        checkCudaErrors(cudaMemset(d_binAddress, 0, memSize)); 

        // CPU histogram
        // perform histogram of data & mask into bins
        for (unsigned int j = 0; j < numElems; ++j) {
            unsigned int bin = (vals_src[j] & mask) >> i;
            binHistogram[bin]++;
        }

        // GPU histogram
        simple_hist<<<gridSize, blockSize>>>(d_binHistogram, d_vals_src,
                numElems, mask, i);
        cudaDeviceSynchronize(); 
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
        naive_scan<<<1, numBins, numBins * sizeof(unsigned int)>>>(
                d_binScan, d_binHistogram, numBins);

        cudaDeviceSynchronize(); 
        checkCudaErrors(cudaMemcpy(h_binScan, d_binScan,
                    numBins * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost));
        // check GPU results
        for (unsigned int j = 0; j < numBins; j++) {
            if (h_binScan[j] != binScan[j]) {
                printf("GPU scan[%d]:%u\tCPU scan[%d]:%u\n", j,
                        h_binHistogram[j], j, binHistogram[j]);
            }
        }

        // Gather everything into the correct location
        // need to move vals and positions
        for (unsigned int j = 0; j < numElems; ++j) {
            unsigned int bin = (vals_src[j] & mask) >> i;
            vals_dst[binScan[bin]] = vals_src[j];
            pos_dst[binScan[bin]] = pos_src[j];
            binScan[bin]++;
        }

        // GPU computation to remap the elements
        for (unsigned int j = 0; j < numBins; ++j) {
            map_to_binFlags<<<gridSize, blockSize>>>(d_binFlags, d_vals_src, 
                                                     mask, i, j, numElems); 

            cudaDeviceSynchronize(); 
            for (unsigned k = 0; k < numElems; k++) {
                unsigned int bin = ((vals_src[k] & mask) >> i);
                binFlags[k] = (bin == j)? 1 : 0; 
            }

            cudaMemcpy(h_binFlags, d_binFlags, memSize, cudaMemcpyDeviceToHost);

            for (unsigned k = 0; k < numElems; k++) {
                unsigned int gpu = h_binFlags[k]; 
                unsigned int cpu = binFlags[k]; 

                if (gpu != cpu) {
                    printf("gpu bin flag[%u]:%u\tcpu bin flag[%u]:%u\n", 
                            k, gpu, k, cpu); 
                    exit(1); 
                }
            }
            
            multiBlockScan(d_binAddress, d_binFlags, numElems); 
            
            binAddress[0] = 0; 
            for (unsigned k = 1; k < numElems; k++) {
                binAddress[k] = binAddress[k - 1] + binFlags[k - 1];  
            }

            cudaMemcpy(h_binAddress, d_binAddress, memSize, cudaMemcpyDeviceToHost); 

            for (unsigned k = 0; k < numElems; k++) {
                unsigned int gpu = h_binAddress[k]; 
                unsigned int cpu = binAddress[k]; 

                if (gpu != cpu) {
                    printf("gpu bin addr[%u]:%u\tcpu bin addr[%u]:%u\n", 
                            k, gpu, k, cpu); 
                    exit(1); 
                }
            }


            offset_by_base<<<gridSize, blockSize>>>(d_binAddress, d_binScan, 
                    j, numElems);
            
            reorder<<<gridSize, blockSize>>>(d_vals_dst, d_vals_src, 
                    d_binFlags, d_binAddress, numElems); 
            
            reorder<<<gridSize, blockSize>>>(d_pos_dst, d_pos_src, 
                    d_binFlags, d_binAddress, numElems); 
            
            // reset d_binFlags after processing for the current
            // bin. 
            memset(binFlags, 0, memSize); 
            memset(binAddress, 0, memSize); 
            checkCudaErrors(cudaMemset(d_binFlags, 0, memSize)); 
            checkCudaErrors(cudaMemset(d_binAddress, 0, memSize)); 
        }

        unsigned int *h_vals_dst = (unsigned int*)malloc(memSize); 
        checkCudaErrors(cudaMemcpy(h_vals_dst, d_vals_dst, 
                    memSize, cudaMemcpyDeviceToHost));

        for (size_t k = 0; k < numElems; k++) {
            unsigned int gpu = h_vals_dst[k]; 
            unsigned int cpu = vals_dst[k]; 
            if (gpu != cpu) {
                printf("gpu vals_dst[%lu]:%u\tcpu vals_dst[%lu]:%u\n", 
                       k, gpu, k, cpu); 
                exit(1); 
            }
        }

        // swap the buffers (pointers only)
        std::swap(vals_dst, vals_src);
        std::swap(pos_dst, pos_src);

        // swap device pointers
        swap_device_ptr(&d_vals_dst, &d_vals_src);
        swap_device_ptr(&d_pos_dst, &d_pos_src); 
    }

    // we did an even number of iterations, need to copy from input buffer into
    // output std::copy(inputVals, inputVals + numElems, outputVals);
    // std::copy(inputPos, inputPos + numElems, outputPos);
	checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, memSize, cudaMemcpyDeviceToDevice)); 
	checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, memSize, cudaMemcpyDeviceToDevice)); 

    delete[] binHistogram;
    delete[] binScan;
}
