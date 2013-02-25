//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"
#include <thrust/host_vector.h>

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


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  /****************************************************************************
  * You can use the code below to help with debugging, but make sure to       *
  * comment it out again before submitting your assignment for grading,       *
  * otherwise this code will take too much time and make it seem like your    *
  * GPU implementation isn't fast enough.                                     *
  *                                                                           *
  * This code MUST RUN BEFORE YOUR CODE in case you accidentally change       *
  * the input values when implementing your radix sort.                       *
  *                                                                           *
  * This code performs the reference radix sort on the host and compares your *
  * sorted values to the reference.                                           *
  *                                                                           *
  * Thrust containers are used for copying memory from the GPU                *
  * ************************************************************************* */
  
  /* thrust::host_vector<unsigned int> h_inputVals(thrust::device_ptr<unsigned int>(d_inputVals),
                                                thrust::device_ptr<unsigned int>(d_inputVals) + numElems);
  thrust::host_vector<unsigned int> h_inputPos(thrust::device_ptr<unsigned int>(d_inputPos),
                                               thrust::device_ptr<unsigned int>(d_inputPos) + numElems);

  thrust::host_vector<unsigned int> h_outputVals(numElems);
  thrust::host_vector<unsigned int> h_outputPos(numElems);

  reference_calculation(&h_inputVals[0], &h_inputPos[0],
                        &h_outputVals[0], &h_outputPos[0],
                        numElems);
  */
   

  //TODO
  //PUT YOUR SORT HERE

  /* *********************************************************************** *
   * Uncomment the code below to do the correctness checking between your    *
   * result and the reference.                                               *
   **************************************************************************/

  /*
  thrust::host_vector<unsigned int> h_yourOutputVals(thrust::device_ptr<unsigned int>(d_outputVals),
                                                     thrust::device_ptr<unsigned int>(d_outputVals) + numElems);
  thrust::host_vector<unsigned int> h_yourOutputPos(thrust::device_ptr<unsigned int>(d_outputPos),
                                                    thrust::device_ptr<unsigned int>(d_outputPos) + numElems);

  checkResultsExact(&h_outputVals[0], &h_yourOutputVals[0], numElems);
  checkResultsExact(&h_outputPos[0], &h_yourOutputPos[0], numElems);
  */
}
