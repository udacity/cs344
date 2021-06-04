#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <hip/hip_runtime.h>
#include <cassert>
#include <string>

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const std::string &filename) {
  //make sure the context initializes ok
  assert(hipFree(0) == 0);

  cv::Mat image;
  image = cv::imread(filename.c_str(), cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  cv::cvtColor(image, imageRGBA, cv::COLOR_BGR2RGBA);

  //allocate memory for the output
  imageGrey.create(image.rows, image.cols, CV_8UC1);

  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *greyImage  = imageGrey.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  //allocate memory on the device for both input and output
  assert(hipMalloc(d_rgbaImage, sizeof(uchar4) * numPixels) == 0);
  assert(hipMalloc(d_greyImage, sizeof(unsigned char) * numPixels) == 0);
  assert(hipMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)) == 0);  //make sure no memory is left laying around

  //copy input array to the GPU
  assert(hipMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, hipMemcpyHostToDevice) == 0);

  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}

void postProcess(const std::string& output_file, unsigned char* data_ptr) {
  cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);

  //output the image
  cv::imwrite(output_file.c_str(), output);
}

void cleanup()
{
  //cleanup
  hipFree(d_rgbaImage__);
  hipFree(d_greyImage__);
}

void generateReferenceImage(std::string input_filename, std::string output_filename)
{
  cv::Mat reference = cv::imread(input_filename, cv::IMREAD_GRAYSCALE);

  cv::imwrite(output_filename, reference);

}
