#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <iostream>

int main(int argc, char **argv) {
  if (!(argc == 3 || argc == 5)) {
    std::cerr << "Usage: ./compare goldImage testImage [per-pixel-error-tolerance global-error-tolerance]" << std::endl;
    exit(1);
  }

  cv::Mat gold = cv::imread(argv[1], -1);
  cv::Mat test = cv::imread(argv[2], -1);

  if (gold.empty()) {
    std::cerr << "Couldn't open file: " << argv[1] << std::endl;
    exit(1);
  }

  if (test.empty()) {
    std::cerr << "Couldn't open file: " << argv[2] << std::endl;
    exit(1);
  }

  if (!gold.isContinuous() || !test.isContinuous()) {
    std::cerr << "Matrices aren't continuous!" << std::endl;
    exit(1);
  }

  if (gold.empty() || test.empty()) {
    std::cerr << "Inputs couldn't be read! " << argv[1] << " " << argv[2] << std::endl;
    exit(1);
  }

  if (gold.channels() != test.channels()) {
    std::cerr << "Images have different number of channels! " << gold.channels() << " " << test.channels() << std::endl;
    exit(1);
  }

  if (gold.size() != test.size()) {
    std::cerr << "Images have different sizes! [" << gold.rows << ", " << gold.cols << "] ["
              << test.rows << ", " << test.cols << "]" << std::endl;
    exit(1);
  }

  cv::Mat diff = abs(gold - test);

  cv::Mat diffSingleChannel = diff.reshape(1, 0); //convert to 1 channel, same # rows

  double minVal, maxVal;

  cv::minMaxLoc(diffSingleChannel, &minVal, &maxVal, NULL, NULL); //NULL because we don't care about location

  //now perform transform so that we bump values to the full range

  diffSingleChannel = (diffSingleChannel - minVal) * (255. / (maxVal - minVal));

  diff = diffSingleChannel.reshape(gold.channels(), 0);

  cv::imwrite("differenceImage.png", diff);
  //OK, now we can start comparing values...
  unsigned char *goldPtr = gold.ptr<unsigned char>(0);
  unsigned char *testPtr = test.ptr<unsigned char>(0);

  if (argc == 3)
    checkResultsExact(goldPtr, testPtr, gold.rows * gold.cols * gold.channels());
  else {
    double perPixelError = atof(argv[3]);
    double globalError   = atof(argv[4]);
    checkResultsEps(goldPtr, testPtr, gold.rows * gold.cols * gold.channels(), perPixelError, globalError);
  }

  std::cout << "PASS" << std::endl;
  return 0;
}
