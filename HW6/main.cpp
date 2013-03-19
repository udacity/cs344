//Udacity HW6 Driver

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


void preProcess( uchar4 **sourceImg, size_t &numRowsSource,  size_t &numColsSource,
                 uchar4 **destImg,
                 uchar4 **blendedImg, const std::string& source_filename,
                 const std::string& dest_filename);

void postProcess(const uchar4* const blendedImg,
                 const size_t numRowsDest, const size_t numColsDest,
                 const std::string& output_file);

void your_blend(const uchar4* const sourceImg,
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const destImg,
                uchar4* const blendedImg);

int main(int argc, char **argv) {
  uchar4 *h_sourceImg, *h_destImg, *h_blendedImg;
  size_t numRowsSource, numColsSource;

  std::string input_source_file;
  std::string input_dest_file;
  std::string output_file;
  if (argc == 4) {
    input_source_file = std::string(argv[1]);
    input_dest_file   = std::string(argv[2]);
    output_file       = std::string(argv[3]);
  }
  else {
    std::cerr << "Usage: ./hw input_source_file input_dest_file output_file" << std::endl;
    exit(1);
  }
  //load the image and give us our input and output pointers
  preProcess(&h_sourceImg, numRowsSource, numColsSource,
             &h_destImg,
             &h_blendedImg, input_source_file, input_dest_file);

  GpuTimer timer;
  timer.Start();

  //call the students' code
  your_blend(h_sourceImg, numRowsSource, numColsSource,
             h_destImg,
             h_blendedImg);

  timer.Stop();
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  int err = printf("%f msecs.\n", timer.Elapsed());
  printf("\n");
  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  //check results and output the tone-mapped image
  postProcess(h_blendedImg, numRowsSource, numColsSource, output_file);

  delete[] h_destImg;
  delete[] h_sourceImg;
  delete[] h_blendedImg;
  return 0;
}

void compareImages(std::string reference_filename, std::string test_filename, bool useEpsCheck,
				   double perPixelError, double globalError)
{
  cv::Mat reference = cv::imread(reference_filename, -1);
  cv::Mat test = cv::imread(test_filename, -1);

  cv::Mat diff = abs(reference - test);

  cv::Mat diffSingleChannel = diff.reshape(1, 0); //convert to 1 channel, same # rows

  double minVal, maxVal;

  cv::minMaxLoc(diffSingleChannel, &minVal, &maxVal, NULL, NULL); //NULL because we don't care about location

  //now perform transform so that we bump values to the full range

  diffSingleChannel = (diffSingleChannel - minVal) * (255. / (maxVal - minVal));

  diff = diffSingleChannel.reshape(reference.channels(), 0);

  cv::imwrite("HW6_differenceImage.png", diff);
  //OK, now we can start comparing values...
  unsigned char *referencePtr = reference.ptr<unsigned char>(0);
  unsigned char *testPtr = test.ptr<unsigned char>(0);

  if (useEpsCheck) {
    checkResultsEps(referencePtr, testPtr, reference.rows * reference.cols * reference.channels(), perPixelError, globalError);
  }
  else
  {
    checkResultsExact(referencePtr, testPtr, reference.rows * reference.cols * reference.channels());
  }

  std::cout << "PASS" << std::endl;
  return;
}
