//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

#include "utils.h"
#include <thrust/host_vector.h>

__global__ void computeMask(
  const uchar4* sourceImg, unsigned char* mask, const int numRowsSource, const int numColsSource)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= numRowsSource * numColsSource) return;

  mask[index] = ((sourceImg[index].x + sourceImg[index].y + sourceImg[index].z) < 765) ? 1: 0;
}

__global__ void  computerInteriorAndBorder(
    unsigned char *mask,
    unsigned char *interior,
    unsigned char *border,
    const int numRowsSource,
    const int numColsSource)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= numRowsSource * numColsSource) return;

  if (index < numColsSource || index < (numColsSource*numRowsSource)-numColsSource-1) // check for access at image borders (top / bottom &? left/right)
  {
    interior = border = 0;
    return;
  }

  int neighborSum = mask[index-1] + mask[index+1] + mask[index - numRowsSource] + mask[index + numRowsSource];
  if (neighborSum == 4)
  {
    interior[index] = 1;
    border[index] = 0;
  } 
  else if(neighborSum == 3)
  {
    interior[index] = 0;
    border[index] = 1;
  }
  else
  {
    interior = border = 0;
  }
}

__global__ void separateChannels(
  uchar4* sourceImg,
  unsigned char* red,
  unsigned char* green,
  unsigned char* blue,
  const int numRowsSource,
  const int numColsSource)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numColsSource * numRowsSource) return;

  red[index] = sourceImg[index].x;
  green[index] = sourceImg[index].z;
  blue[index] = sourceImg[index].y;
}

__global__ void initialize(
  unsigned char* source,
  float* blended,
  const int numRowsSource,
  const int numColsSource)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numColsSource * numRowsSource) return;

  blended[index] = static_cast<float>(source[index]);
}

__global__ void performIteration(
  float * currentGuess,
  float * nextGuess,
  unsigned char * sourceChannel,
  unsigned char * destChannel,
  unsigned char * interiorMask,
  unsigned char * borderMask,
  const int numRowsSource,
  const int numColsSource)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numColsSource * numRowsSource) return;
  if (interiorMask[index] == 0) return;

  float sum1 = 0;
  sum1 += interiorMask[index-numColsSource] * currentGuess[index-numColsSource] + borderMask[index-numColsSource] * destChannel[index-numColsSource];
  sum1 += interiorMask[index+numColsSource] * currentGuess[index+numColsSource] + borderMask[index+numColsSource] * destChannel[index+numColsSource];
  sum1 += interiorMask[index-1] * currentGuess[index-1] + borderMask[index-1] * destChannel[index-1];
  sum1 += interiorMask[index+1] * currentGuess[index+1] + borderMask[index+1] * destChannel[index+1];


  float sum2 = 0;
  sum2 += interiorMask[index] * (sourceChannel[index] - sourceChannel[index-numColsSource]);
  sum2 += interiorMask[index] * (sourceChannel[index] - sourceChannel[index+numColsSource]); 
  sum2 += interiorMask[index] * (sourceChannel[index] - sourceChannel[index-1]);
  sum2 += interiorMask[index] * (sourceChannel[index] - sourceChannel[index+1]);

  float newVal = (sum1 + sum2) / 4.f;
  nextGuess[index] = min(255.f, max(0.f, newVal));
}

__global__ void createOutput(
    uchar4 * blendedImg,
    float *finalGuessRed,
    float *finalGuessGreen,
    float *finalGuessBlue,
    unsigned char *interiorMask,
    const int numRowsSource,
    const int numColsSource)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= numColsSource * numRowsSource) return;
  if (!(interiorMask[index]))  return;
  
  blendedImg[index].x = static_cast<unsigned char>(finalGuessRed[index]);
  blendedImg[index].y = static_cast<unsigned char>(finalGuessGreen[index]);
  blendedImg[index].z = static_cast<unsigned char>(finalGuessBlue[index]);
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{  
  uchar4        *d_sourceImg, *d_destImg, *d_blendedImg;
  unsigned char *d_mask, *d_border, *d_interior;
  unsigned char *d_redSrc, *d_greenSrc, *d_blueSrc;
  unsigned char *d_redDst, *d_greenDst, *d_blueDst;


  checkCudaErrors(cudaMalloc(&d_sourceImg,  sizeof(uchar4) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_destImg,    sizeof(uchar4) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_blendedImg, sizeof(uchar4) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_mask,       sizeof(unsigned char) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_border,     sizeof(unsigned char) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_interior,   sizeof(unsigned char) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_redSrc,     sizeof(unsigned char) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_greenSrc,   sizeof(unsigned char) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_blueSrc,    sizeof(unsigned char) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_redDst,     sizeof(unsigned char) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_greenDst,   sizeof(unsigned char) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_blueDst,    sizeof(unsigned char) * numColsSource * numRowsSource));

  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg,  sizeof(uchar4) * numColsSource * numRowsSource, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_destImg,   h_destImg,    sizeof(uchar4) * numColsSource * numRowsSource, cudaMemcpyHostToDevice));

  // 1) Compute a mask of the pixels from the source image to be copied
  //    The pixels that shouldn't be copied are completely white, they
  //    have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

  const int maxThreadsPerBlock = 1024;
  int threadsPerBlock = maxThreadsPerBlock;
  int blocks = (numRowsSource * numColsSource + threadsPerBlock - 1) / threadsPerBlock;

  computeMask<<<blocks, threadsPerBlock>>>(d_sourceImg, d_mask, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // 2) Compute the interior and border regions of the mask.  An interior
  //    pixel has all 4 neighbors also inside the mask.  A border pixel is
  //    in the mask itself, but has at least one neighbor that isn't.

  computerInteriorAndBorder<<<blocks, threadsPerBlock>>>(
    d_mask, d_interior, d_border, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // 3) Separate out the incoming image into three separate channels

  separateChannels<<<blocks, threadsPerBlock>>>(d_sourceImg, d_redSrc, d_greenSrc, d_blueSrc, numRowsSource, numColsSource);
  separateChannels<<<blocks, threadsPerBlock>>>(d_destImg, d_redDst, d_greenDst, d_blueDst, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // 4) Create two float(!) buffers for each color channel that will
  //    act as our guesses.  Initialize them to the respective color
  //    channel of the source image since that will act as our intial guess.

  float *d_blendedRed1;
  float *d_blendedRed2;
  float *d_blendedGreen1;
  float *d_blendedGreen2;
  float *d_blendedBlue1;
  float *d_blendedBlue2;
  checkCudaErrors(cudaMalloc(&d_blendedRed1,    sizeof(float) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_blendedRed2,    sizeof(float) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_blendedGreen1,  sizeof(float) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_blendedGreen2,  sizeof(float) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_blendedBlue1,   sizeof(float) * numColsSource * numRowsSource));
  checkCudaErrors(cudaMalloc(&d_blendedBlue2,   sizeof(float) * numColsSource * numRowsSource));

  initialize<<<blocks, threadsPerBlock>>>(d_redSrc,    d_blendedRed1,    numRowsSource, numColsSource);
  initialize<<<blocks, threadsPerBlock>>>(d_redDst,    d_blendedRed2,    numRowsSource, numColsSource);
  initialize<<<blocks, threadsPerBlock>>>(d_greenSrc,  d_blendedGreen1,  numRowsSource, numColsSource);
  initialize<<<blocks, threadsPerBlock>>>(d_greenDst,  d_blendedGreen2,  numRowsSource, numColsSource);
  initialize<<<blocks, threadsPerBlock>>>(d_blueSrc,   d_blendedBlue1,   numRowsSource, numColsSource);
  initialize<<<blocks, threadsPerBlock>>>(d_blueDst,   d_blendedBlue2,   numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // 5) For each color channel perform the Jacobi iteration described 
  //    above 800 times.
  const int iterations(800);
  for(int i = 0; i < iterations; ++i)
  {
    performIteration<<<blocks, threadsPerBlock>>>(
      d_blendedRed1,d_blendedRed2, d_redSrc, d_redDst, d_interior, d_border, numRowsSource, numColsSource);
    performIteration<<<blocks, threadsPerBlock>>>(
      d_blendedGreen1,d_blendedGreen2, d_greenSrc, d_greenDst, d_interior, d_border, numRowsSource, numColsSource);
    performIteration<<<blocks, threadsPerBlock>>>(
      d_blendedBlue1,d_blendedBlue2, d_blueSrc, d_blueDst, d_interior, d_border, numRowsSource, numColsSource);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    std::swap(d_blendedRed1, d_blendedRed2);
    std::swap(d_blendedGreen1, d_blendedGreen2);
    std::swap(d_blendedBlue1, d_blendedBlue2);
  }
  // even number of iterations, so swap once more
  std::swap(d_blendedRed1, d_blendedRed2);
  std::swap(d_blendedGreen1, d_blendedGreen2);
  std::swap(d_blendedBlue1, d_blendedBlue2);

  // 6) Create the output image by replacing all the interior pixels
  //    in the destination image with the result of the Jacobi iterations.
  //    Just cast the floating point values to unsigned chars since we have
  //    already made sure to clamp them to the correct range.
  checkCudaErrors(cudaMemcpy(d_blendedImg, d_destImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));

  createOutput<<<blocks, threadsPerBlock>>>(
    d_blendedImg, d_blendedRed2, d_blendedGreen2, d_blendedBlue2, d_interior, numRowsSource, numColsSource);

  checkCudaErrors(cudaMemcpy(h_blendedImg, d_blendedImg, sizeof(uchar4) * numRowsSource * numColsSource,cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //
  // DONT FORGET TO FREE ALL OF THE MEMORY !!!
  //
  checkCudaErrors(cudaFree(d_sourceImg));
  checkCudaErrors(cudaFree(d_destImg));
  checkCudaErrors(cudaFree(d_blendedImg));
  checkCudaErrors(cudaFree(d_mask));
  checkCudaErrors(cudaFree(d_border));
  checkCudaErrors(cudaFree(d_interior));
  checkCudaErrors(cudaFree(d_redSrc));
  checkCudaErrors(cudaFree(d_redDst));
  checkCudaErrors(cudaFree(d_greenSrc));
  checkCudaErrors(cudaFree(d_greenDst));
  checkCudaErrors(cudaFree(d_blueSrc));
  checkCudaErrors(cudaFree(d_blueDst));
  checkCudaErrors(cudaFree(d_blendedRed1));
  checkCudaErrors(cudaFree(d_blendedRed2));
  checkCudaErrors(cudaFree(d_blendedGreen1));
  checkCudaErrors(cudaFree(d_blendedGreen2));
  checkCudaErrors(cudaFree(d_blendedBlue1));
  checkCudaErrors(cudaFree(d_blendedBlue2));
    


  /* The reference calculation is provided below, feel free to use it
     for debugging purposes. 
  */

  /**
  uchar4* h_reference = new uchar4[srcSize];
  reference_calc(h_sourceImg, numRowsSource, numColsSource,
                 h_destImg, h_reference);

  checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * srcSize, 2, .01);
  delete[] h_reference;
  /**/
}
