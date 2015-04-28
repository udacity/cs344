//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <memory>

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
#define NUM_BITS 4
#define RADIX (1 << NUM_BITS)
#define BLOCK_SIZE 256 // no more than 254 in order to fit all relative positions into the array of bytes

__device__ __host__ unsigned int get_digit(unsigned int num, unsigned int shift)
{
  int mask = (RADIX - 1) << shift;
  return (num & mask) >> shift;
}

// histogram of digits within one block
__global__ void hist_digits(const unsigned int* const in, 
                            const size_t size, 
                            unsigned int shift, 
                            const int nBins, 
                            unsigned int* const  outBins)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
  {
    return;
  }

  unsigned int digit = get_digit(in[idx], shift);

  atomicAdd(&outBins[digit], 1);
}

// # of each digit within a block.
// used to compute relative positions of each digit within the array
__global__ void num_digits_per_block(const unsigned int* const in, 
                                     const size_t size, 
                                     unsigned int shift, 
                                     const int nBins, 
                                     unsigned int* const  outBins)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
  {
    return;
  }

  unsigned int digit = get_digit(in[idx], shift);

  atomicAdd(&outBins[blockIdx.x * RADIX + digit], 1);
}

// add per-block cdf of digits to get a truly relative position of the digit (relative to the start of the sequence)
// and then add cdf of all digits to get the absolute position of the element in the new order
__global__ void get_absolute_positions(const unsigned int * const in, unsigned int * const relPos, size_t size, unsigned int shift, unsigned int * const maxVals, unsigned int * const digitCdf)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= size)
  {
    return;
  }
  unsigned int digit = get_digit(in[idx], shift);

  relPos[idx] += maxVals[blockIdx.x * RADIX + digit];
  relPos[idx] += digitCdf[digit];
}

// Build a relative positional scan of digit, shifted by "shift"
// in - array of numbers to sort
// inPos - positions of the keys that also need to move
// size - size of the array
// shift - which significant digit are we concerned about
// rel - array of relative positions of the numbers in the in array (relative to current block), based on the digit of interest.
__global__ void rel_pos_per_block(const unsigned int * const in, 
                                  const size_t size, unsigned int shift, 
                                  unsigned int * const rel)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int digit_pos = get_digit(in[idx], shift);

  int offset = digit_pos * blockDim.x;

  __shared__ unsigned int relativePosition[RADIX * BLOCK_SIZE];
  for(int i = 0; i < RADIX; i++)
  {
    relativePosition[threadIdx.x + i * blockDim.x] = 0;
  }
  __syncthreads();

  // initialize to the "startup" positions for each digit.
  // since we always set them to "1" and the scan is supposed
  // to be exclusive, we will need to subtract "1" at the end
  relativePosition[threadIdx.x + offset] = 1;

  // accumulate position values for each digit
  for(int step = 1; step < size; step <<= 1)
  {
    for(int j = 0; j < RADIX; j++)
    {
      __syncthreads();
      // position start of the digit in the relatvePosition array
      int offset = j * blockDim.x;
      int tSum = relativePosition[threadIdx.x + offset];

      if(threadIdx.x >= step)
      {
        tSum += relativePosition[threadIdx.x + offset - step];
      }
      __syncthreads();
      relativePosition[threadIdx.x + offset] = tSum;
    }
  }

  __syncthreads();

  rel[idx] = relativePosition[threadIdx.x + offset] - 1;
}

// Convert relative postions to absolute ones by adding up to the histogram
// rel array of relative positions
// cdf - exclusive cdf of the counts histogram
__global__ void move_to_out(unsigned int * const dest, 
                            unsigned int * const destPos, 
                            unsigned int * const source, 
                            unsigned int * const sourcePos, 
                            unsigned int * const rel, 
                            size_t size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= size)
  {
    return;
  }

  int pos = rel[idx];
  dest[pos] = source[idx];
  destPos[pos] = sourcePos[idx];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 

  const int blockSize = BLOCK_SIZE;
  int nBlocks = (numElems + blockSize) / blockSize;
  int nWholeBlocks = numElems / blockSize;
  int remainder = numElems % blockSize; // elements that don't fit into the blocks must be handled separately
  bool swapInOut = false; //flag to show if we need to swap input & output at the end

  unsigned int *d_outBins, *d_rel, *d_maxDigitPerBlock;

  std::auto_ptr<unsigned int> h_hist(new unsigned int[RADIX]);
  std::auto_ptr<unsigned int> cdf(new unsigned int [RADIX]);
  std::auto_ptr<unsigned int> h_rel(new unsigned int[numElems]);
  std::auto_ptr<unsigned int> h_digits_per_block(new unsigned int[RADIX * nBlocks]);
  std::auto_ptr<unsigned int> cdf_per_block(new unsigned int[RADIX * nBlocks]);

  checkCudaErrors(cudaMalloc(&d_outBins, RADIX * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_rel, numElems * sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_maxDigitPerBlock, nBlocks * RADIX * sizeof(int)));

  for (unsigned int shift = 0; shift < 8 * sizeof(unsigned int); shift += NUM_BITS)
  {
    unsigned int * d_input, * d_output, * d_inPos, * d_outPos;
    if (!swapInOut)
    {
      d_input = d_inputVals;
      d_output = d_outputVals;
      d_inPos = d_inputPos;
      d_outPos = d_outputPos;
    }
    else
    {
      d_output = d_inputVals;
      d_input = d_outputVals;
      d_outPos = d_inputPos;
      d_inPos = d_outputPos;
    }

    //1. Compute histogram of digits
    checkCudaErrors(cudaMemset(d_outBins, 0, RADIX * sizeof(int)));

    hist_digits<<<nBlocks, blockSize>>>(d_input, numElems, shift, RADIX, d_outBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_hist.get(), d_outBins, RADIX * sizeof(int), cudaMemcpyDeviceToHost));

    //2. Compute exclusive scan of the histogram
    cdf.get()[0] = 0;

    for(int digit = 1; digit < RADIX; digit++)
    {
      cdf.get()[digit] = cdf.get()[digit - 1] + h_hist.get()[digit - 1];
    }

    //3. Compute relative position of each digit per block.
    //a. Compute # of each digit per block
    checkCudaErrors(cudaMemset(d_maxDigitPerBlock, 0, RADIX * nBlocks * sizeof(int)));
    num_digits_per_block<<<nBlocks, blockSize>>>(d_input, numElems, shift, RADIX, d_maxDigitPerBlock);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_digits_per_block.get(), d_maxDigitPerBlock, RADIX * sizeof(int) * nBlocks, cudaMemcpyDeviceToHost));

    //b. Compute exclusive scan per block of numbers of digits per block
    memset(cdf_per_block.get(), 0, RADIX * nBlocks * sizeof(unsigned int));

    for(int block = 1; block < nBlocks; block++)
    {
      for(int digit = 0; digit < RADIX; digit++)
      {
        cdf_per_block.get()[block * RADIX + digit] = cdf_per_block.get()[(block - 1) * RADIX + digit] + h_digits_per_block.get()[(block - 1) * RADIX + digit];
      }
    }

    //c. Compute relative position
    rel_pos_per_block<<<nWholeBlocks, blockSize>>>(d_input, blockSize, shift, d_rel);

    if(remainder > 0)
    {
      rel_pos_per_block<<<1, remainder>>>(&d_input[nWholeBlocks * blockSize], remainder, shift, &d_rel[nWholeBlocks * blockSize]);
    }
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // d. Add things up to get the real relative position of each digit (relative to the start of the sequence, not per block)
    checkCudaErrors(cudaMemcpy(d_maxDigitPerBlock, cdf_per_block.get(), RADIX * nBlocks * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_outBins, cdf.get(), RADIX * sizeof(int), cudaMemcpyHostToDevice));

    get_absolute_positions<<<nBlocks, blockSize>>>(d_input, d_rel, numElems, shift, d_maxDigitPerBlock, d_outBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //4. now we can update the output
    move_to_out<<<nBlocks, blockSize>>>(d_output, d_outPos, d_input, d_inPos, d_rel, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    swapInOut = !swapInOut;
  }

  // when it's all over we must make sure output values are where they ought to be.
  if(!swapInOut)
  {
    checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  }

  checkCudaErrors(cudaFree(d_outBins));
  checkCudaErrors(cudaFree(d_rel));
  checkCudaErrors(cudaFree(d_maxDigitPerBlock));

