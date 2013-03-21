#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

float *h_filter__;

size_t numRows() { return imageInputRGBA.rows; }
size_t numCols() { return imageInputRGBA.cols; }

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                unsigned char **d_redBlurred,
                unsigned char **d_greenBlurred,
                unsigned char **d_blueBlurred,
                float **h_filter, int *filterWidth,
                const std::string &filename) {

  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);

  //allocate memory for the output
  imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
  *h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4))); //make sure no memory is left laying around

  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  d_inputImageRGBA__  = *d_inputImageRGBA;
  d_outputImageRGBA__ = *d_outputImageRGBA;

  //now create the filter that they will use
  const int blurKernelWidth = 9;
  const float blurKernelSigma = 2.;

  *filterWidth = blurKernelWidth;

  //create and fill the filter we will convolve with
  *h_filter = new float[blurKernelWidth * blurKernelWidth];
  h_filter__ = *h_filter;

  float filterSum = 0.f; //for normalization

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    }
  }

  //blurred
  checkCudaErrors(cudaMalloc(d_redBlurred,    sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_greenBlurred,  sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_blueBlurred,   sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_redBlurred,   0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_blueBlurred,  0, sizeof(unsigned char) * numPixels));
}

void postProcess(const std::string& output_file, uchar4* data_ptr) {
  cv::Mat output(numRows(), numCols(), CV_8UC4, (void*)data_ptr);

  cv::Mat imageOutputBGR;
  cv::cvtColor(output, imageOutputBGR, CV_RGBA2BGR);
  //output the image
  cv::imwrite(output_file.c_str(), imageOutputBGR);
}

void cleanUp(void)
{
  cudaFree(d_inputImageRGBA__);
  cudaFree(d_outputImageRGBA__);
  delete[] h_filter__;
}


// An unused bit of code showing how to accomplish this assignment using OpenCV.  It is much faster 
//    than the naive implementation in reference_calc.cpp.
void generateReferenceImage(std::string input_file, std::string reference_file, int kernel_size)
{
	cv::Mat input = cv::imread(input_file);
	// Create an identical image for the output as a placeholder
	cv::Mat reference = cv::imread(input_file);
	cv::GaussianBlur(input, reference, cv::Size2i(kernel_size, kernel_size),0);
	cv::imwrite(reference_file, reference);
}
