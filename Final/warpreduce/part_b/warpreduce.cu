#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "compare.h"
#include "gputimer.h"

// Subpart b:
// Compute capability 2.0+ GPUs have support for 3 per-warp instructions.
// Namely, these instructions are:
//
// int __popc(int x) Population Count: Returns the number of bits that are set
// to 1 in the 32-bit integer x.
//
// int __clz(int x) Count Leading Zeros: Returns the number of consecutive zero
// bits beginning at the most significant bit of the 32-bit integer x.
//
// int __ballot(int p) Returns a 32-bit integer in which bit k is set if and only
// if the predicate p provided by the thread in lane k of the warp is non-zero.

__device__ unsigned int warp_reduce(unsigned int p, volatile unsigned int * s) {
    // Assumes values in 'p' are either 1 or 0
    // Should not use 's'
    // Sums p across warp, returning the result.
    // You can do this without using the character '+' in your code at all
    //
    // TODO: Fill in the rest of this function
    //
}

__global__ void reduce(unsigned int * d_out_warp, 
                       const unsigned int * d_in)
{
    extern __shared__ unsigned int s[];
    int t = threadIdx.x;
    int p = d_in[t];

    unsigned int wr = warp_reduce(p, s);
    if (t == 0)
    {
        *d_out_warp = wr;
    }
}

int main(int argc, char **argv)
{
    const int ARRAY_SIZE = 32;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(unsigned int);

    // generate the input array on the host
    unsigned int h_in[ARRAY_SIZE];
    unsigned int sum = 0;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [0, 1]
        h_in[i] = (float)random()/(float)RAND_MAX > 0.5f ? 1 : 0;
        sum += h_in[i];
    }

    // declare GPU memory pointers
    unsigned int * d_in, * d_out_warp;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out_warp, sizeof(unsigned int));

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

    GpuTimer timer;
    timer.Start();
    // launch the kernel
    reduce<<<1, ARRAY_SIZE, ARRAY_SIZE * sizeof(unsigned int)>>>
        (d_out_warp, d_in);
    timer.Stop();

    printf("Your code executed in %g ms\n", timer.Elapsed());  

    unsigned int h_out_warp;
    // copy back the sum from GPU
    cudaMemcpy(&h_out_warp, d_out_warp, sizeof(unsigned int), 
               cudaMemcpyDeviceToHost);

    // compare your result against the expected reduce sum
    compare(h_out_warp, sum);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_out_warp);
        
}

