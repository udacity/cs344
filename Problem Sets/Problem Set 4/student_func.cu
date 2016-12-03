//Udacity HW 4
//Radix Sorting

#include <float.h>
#include <math.h>
#include <stdio.h>

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

__global__
void histogram_kernel(unsigned int pass,
                      unsigned int * d_bins,
                      unsigned int* const d_input, 
                      const int size) {  
    int mid = threadIdx.x + blockDim.x * blockIdx.x;
    if(mid >= size)
        return;
    unsigned int one = 1;
    int bin = ((d_input[mid] & (one<<pass)) == (one<<pass)) ? 1 : 0;
    if(bin) 
         atomicAdd(&d_bins[1], 1);
    else
         atomicAdd(&d_bins[0], 1);
}

// we will run 1 exclusive scan, but then when we 
// do the move, for zero vals, we iwll take mid - val of scan there
__global__
void exclusive_scan_kernel(unsigned int pass,
                    unsigned int const * d_inputVals,
                    unsigned int * d_output,
                    const int size,
                    unsigned int base,
                    unsigned int threadSize) {
    int mid = threadIdx.x + threadSize * base;
    int block = threadSize*base;
        unsigned int one = 1;

    if(mid >= size)
        return;
      unsigned int val = 0;
    if(mid > 0)
        val = ((d_inputVals[mid-1] & (one<<pass))  == (one<<pass)) ? 1 : 0;
    else
        val = 0;

    d_output[mid] = val;
    
    __syncthreads();
    
    for(int s = 1; s <= threadSize; s *= 2) {
        int spot = mid - s; 
         
        if(spot >= 0 && spot >=  threadSize*base)
             val = d_output[spot];
        __syncthreads();
        if(spot >= 0 && spot >= threadSize*base)
            d_output[mid] += val;
        __syncthreads();
    }
    if(base > 0)
        d_output[mid] += d_output[base*threadSize - 1];
    
}

__global__
void test_kernel(unsigned int pass,
                 unsigned int * d_output,
                  size_t numElems)
{
    int mid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int one=1;
    unsigned int val = (unsigned int)mid;
    if(mid < numElems) {
        d_output[mid] = (val & (one<<pass)) == (one<<pass) ? 1 : 0; 
    }
}

void test(unsigned int pass) { 
    int numtest = 24;
    unsigned int* d_out;
    
    checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned int)*numtest));
    
    unsigned int h_out[numtest];
    
    test_kernel<<<dim3(1), dim3(numtest)>>>(pass, d_out, numtest);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
       
    checkCudaErrors(cudaMemcpy(&h_out, d_out, numtest*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    
    for(int i = 0; i< numtest; i++) {
        printf("%d: %d, ", i, h_out[i]);
    }
    printf("\n");
    
    checkCudaErrors(cudaFree(d_out));
}

__global__
void move_kernel(
    unsigned int pass,
    unsigned int* const d_inputVals,
    unsigned int* const d_inputPos,
    unsigned int* d_outputVals,
    unsigned int* d_outputPos,
    unsigned int* d_outputMove,
    unsigned int* const d_scanned,
    unsigned int  one_pos,
    const size_t numElems) {
    
    int mid = threadIdx.x + blockDim.x * blockIdx.x;
    if(mid >= numElems)
        return;
    
    unsigned int scan=0;
    unsigned int base=0;
    unsigned int one= 1;
    if( ( d_inputVals[mid] & (one<<pass)) == (1<<pass)) {
        scan = d_scanned[mid]; 
        base = one_pos;
    } else {
        scan = (mid) - d_scanned[mid];
        base = 0;
    }
    
    d_outputMove[mid] = base+scan;
    d_outputPos[base+scan]  = d_inputPos[mid];//d_inputPos[0];
    d_outputVals[base+scan] = d_inputVals[mid];//base+scan;//d_inputVals[0];
    
}
   
int debug = 1;
void debug_device_array(char* name, int l, unsigned int * d_arr, int numElems) {
    
   
    if(!debug)
        return;
    unsigned int h_arr[l];
    checkCudaErrors(cudaMemcpy(&h_arr, d_arr, l*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    printf(name);
    printf(" ");
    for(int i=0; i < l; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");
    unsigned int max = 0;
    unsigned int min = 1000000;
    unsigned int h_arr2[numElems];
    checkCudaErrors(cudaMemcpy(&h_arr2, d_arr, numElems*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    for(int i = 0; i < numElems; i++) {
        if(h_arr2[i] < min)
            min = h_arr2[i];
         if(h_arr2[i] > max)
            max = h_arr2[i];
    }
    printf("max %d min %d\n", max, min);
}

void verify_scan(unsigned int * d_arr, unsigned int * d_scanned, int numElems, int pass) {
    unsigned int h_arr[3000];
    unsigned int one  =1;
    unsigned int h_scanned[3000];
    checkCudaErrors(cudaMemcpy(&h_arr, d_arr, 3000*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_scanned, d_scanned, 3000*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int acc = 0;
    for(int i = 0; i < 3000; i++) {
        if(acc != h_scanned[i]) {
               printf("wrong at %d %d != %d\n", i, acc, h_scanned[i]);
        }
        acc+= ((h_arr[i] & (one<<pass)) == (one<<pass)) ? 1 : 0;
    }
}
/*
void verify_sort(unsigned int * d_sorted, int numElems, int pass) {
    unsigned int h_sorted[3000];
    checkCudaErrors(cudaMemcpy(&h_sorted, d_sorted, 3000*sizeof(unsigned int), cudaMemcpyDeviceToHost));
    unsigned int last = h_scanned[0];
    for(int i = 1; i < 3000; i++) {
        if(acc != h_scanned[i]) {
               printf("wrong at %d %d != %d\n", i, acc, h_scanned[i]);
        }
        acc+= ((h_arr[i] & (1<<pass) == (1<<pass)) ? 1 : 0);
    }
}*/
// gives you a good max size for n/d
int get_max_size(int n, int d) {
    return (int)ceil( (float)n/(float)d ) + 1;
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{  
    /*for(unsigned int i = 0; i < 32; i++) {
        printf("testing %d \n", i);
        test(i);
    }*/
    
    //return;
    unsigned int* d_bins;
    unsigned int  h_bins[2];
    unsigned int* d_scanned;
    unsigned int* d_moved;
    const size_t histo_size = 2*sizeof(unsigned int);
    const size_t arr_size   = numElems*sizeof(unsigned int);
    
    checkCudaErrors(cudaMalloc(&d_bins, histo_size));
    checkCudaErrors(cudaMalloc(&d_scanned, arr_size));
    checkCudaErrors(cudaMalloc(&d_moved, arr_size));
    
    
    // just keep thread dim at 1024
    dim3 thread_dim(1024 );
    dim3 hist_block_dim(get_max_size(numElems, thread_dim.x));
    
    // get number of elements
    //printf("numElems %d\n", numElems);
    
    debug_device_array("input", 100, d_inputVals, numElems);
        
    
    for(unsigned int pass = 0; pass < 32; pass++) {
        unsigned int one = 1;
        /*if((one<<pass) <= 0) {
            printf("breaking at pass %d ", pass);
            break;
        }*/
        checkCudaErrors(cudaMemset(d_bins, 0, histo_size));
        checkCudaErrors(cudaMemset(d_scanned, 0, arr_size));
        checkCudaErrors(cudaMemset(d_outputVals, 0, arr_size));
        checkCudaErrors(cudaMemset(d_outputPos, 0, arr_size));
        
        histogram_kernel<<<hist_block_dim, thread_dim>>>(pass, d_bins, d_inputVals, numElems);
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
       
        // copy the histogram data to host
        checkCudaErrors(cudaMemcpy(&h_bins, d_bins, histo_size, cudaMemcpyDeviceToHost));
        
        printf("hey guys %d %d %d %d %d \n", h_bins[0], h_bins[1], h_bins[0]+h_bins[1], numElems, (one<<pass));
     
        // now we have 0, and 1 start position.. 
        // get the scan of 1's
        
        for(int i = 0; i < get_max_size(numElems, thread_dim.x); i++) {
            exclusive_scan_kernel<<<dim3(1), thread_dim>>>(
                   pass,
                   d_inputVals,
                   d_scanned,
                   numElems,
                   i, 
                   thread_dim.x
            );
            cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        
        }        
        //printf("made it past scanned\n");
        
        //debug_device_array("input", 100, d_inputVals, numElems);
        //debug_device_array("scanned", 100, d_scanned, numElems);
        //verify_scan(d_inputVals, d_scanned, numElems, pass);

        // calculate the move positions
        move_kernel<<<hist_block_dim, thread_dim>>>(
            pass,
            d_inputVals,
            d_inputPos,
            d_outputVals,
            d_outputPos,
            d_moved,
            d_scanned,
            h_bins[0],
            numElems
        );
        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
        
        //debug_device_array("move", 100, d_moved, numElems);
        //debug_device_array("output vals ", 100, d_outputVals, numElems);
        //debug_device_array("output pos  ", 100, d_outputPos, numElems);
        
       
       // printf("made it past move calculation \n");
        
        //finall
         // copy the histogram data to input
        checkCudaErrors(cudaMemcpy(d_inputVals, d_outputVals, arr_size, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_inputPos, d_outputPos, arr_size, cudaMemcpyDeviceToDevice));

        cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
       
        
        
    }
    //printf("Made to end\n");
    //      debug_device_array("output vals ", 100000, d_outputVals, numElems);
    //    debug_device_array("output pos  ", 100, d_outputPos, numElems);
        
    
    checkCudaErrors(cudaFree(d_moved));
    checkCudaErrors(cudaFree(d_scanned));
    checkCudaErrors(cudaFree(d_bins));
}

