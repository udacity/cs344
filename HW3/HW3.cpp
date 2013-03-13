#include <opencv2/opencv.hpp>
#include "utils.h"

#include "reference_calc.h"

void generateReferenceImage(std::string reference_file, const float* const h_logLuminance, unsigned int* const h_cdf,
                          const size_t numRows, const size_t numCols, const size_t numBins)
{
	float min_logLum=0.0f, max_logLum=1.0f;
	referenceCalculation(h_logLuminance, h_cdf, numRows, numCols, numBins, min_logLum, max_logLum);
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

  cv::imwrite("HW3_differenceImage.png", diff);
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
