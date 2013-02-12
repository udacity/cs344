// Homework 2
// Gaussian Blur
//
// In this homework we are blurring an image by convolving
// the image with a filter.  For those of you unfamiliar with
// convolution, imagine that we have a square much smaller than
// image itself.  For every pixel in the image we are going
// to lay this square down and then multiply each pair of numbers
// that line up.  Then we add up all of the generated numbers and assign
// that value to our output for the pixel.

// For a color image that has multiple channels it is easiest to do
// this if we first separate the different color channels so that
// each color is continuous instead of being interleaved

// That is instead of RGBARGBARGBARGBA... we transform to three images(as
// in the previous homework we ignore the alpha channel again):
//  1) RRRRRRRR...
//  2) GGGGGGGG...
//  3) BBBBBBBB...
//
// The original layout is known an Array of Structures (AoS) whereas the
// format we are converting to is known as a Structure of Arrays (SoA).

// The kernels that do this separation and then recombination have been written
// for you.  It is your job to write the "meat" of the assignment which is the
// kernel that performs the actual blur.

/*
Filter:

   0 1 0
   1 1 1
   0 1 0

Image (applying filter to pixel at the center of the square):
   1  2  5  2  0  3
      -------
   3 |2  5  1| 6  0         0  5  0
     |       |              
   4 |3  6  2| 1  4   ->    3  6  2   ->  3 + 5 + 6 + 2  ->  16
     |       |
   0 |4  0  3| 4  2         0  0  0
      -------
   9  6  5  0  3  9

        (1)                   (2)            (3)             (4)
   */

#include "reference_calc.cpp"
#include "utils.h"


//You must fill in this kernel to perform the convolution of the inputChannel
//with the filter and put the result in the outputChannel.

//A good starting place is to map each thread to a pixel as you have before
//Then every thread can perform the steps 2 through 4 in the diagram above
//completely independently of one another.

//Once you have gotten that working correctly, then you can think about using
//shared memory and having the threads cooperate to achieve better performance.
__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  //TODO
}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows, int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{

  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  uchar4 rgba = inputImageRGBA[thread_1D_pos];

  redChannel[thread_1D_pos]   = rgba.x;
  greenChannel[thread_1D_pos] = rgba.y;
  blueChannel[thread_1D_pos]  = rgba.z;
}


//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows, int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        const float* const h_filter, const int filterWidth)
{
  unsigned char *d_red, *d_green, *d_blue;
  unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;
  float         *d_filter;

  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRows * numCols));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRows * numCols));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRows * numCols));

  //blurred
  checkCudaErrors(cudaMalloc(&d_redBlurred,   sizeof(unsigned char) * numRows * numCols));
  checkCudaErrors(cudaMalloc(&d_greenBlurred, sizeof(unsigned char) * numRows * numCols));
  checkCudaErrors(cudaMalloc(&d_blueBlurred,  sizeof(unsigned char) * numRows * numCols));

  //TODO:
  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc

  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!

  const int warpSize = 32;
  const dim3 blockSize(warpSize, 16, 1);;
  //TODO: Set the correct grid size for the image
  const dim3 gridSize;

  //first kernel to split RGBA into separate channels
  //We take care of launching this one for you
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //second phase does 3 convolutions, one on each color channel
  //TODO: Call your convolution kernel here 3 times, once for each
  //color channel.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //last phase recombines
  //We also take care of launching this one as well
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred, d_greenBlurred, d_blueBlurred,
                                             d_outputImageRGBA, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  /****************************************************************************
  * You can use the code below to help with debugging, but make sure to       *
  * comment it out again before submitting your assignment for grading,       *
  * otherwise this code will take too much time and make it seem like your    *
  * GPU implementation isn't fast enough.                                     *
  *                                                                           *
  * This code generates a reference image on the host by running the          *
  * reference calculation we have given you.  It then copies your GPU         *
  * generated image back to the host and calls a function that compares the   *
  * the two and will output the first location they differ by too much.       *
  * ************************************************************************* */

  /*uchar4 *h_outputImage     = new uchar4[numRows * numCols];
  uchar4 *h_outputReference = new uchar4[numRows * numCols];

  checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImageRGBA, 
                             numRows * numCols * sizeof(uchar4), 
                             cudaMemcpyDeviceToHost));

  referenceCalculation(h_inputImageRGBA, h_outputReference, numRows, numCols,
                       h_filter, filterWidth);

  //the 4 is because there are 4 channels in the image
  checkResultsExact((unsigned char *)h_outputReference, (unsigned char *)h_outputImage, numRows * numCols * 4); 
 
  delete [] h_outputImage;
  delete [] h_outputReference;*/

  //TODO: make sure you free the memory you allocated
  cudaFree(d_red);
  cudaFree(d_redBlurred);
  cudaFree(d_green);
  cudaFree(d_greenBlurred);
  cudaFree(d_blue);
  cudaFree(d_blueBlurred);
  cudaFree(d_filter);
}
