//Udacity HW6 Driver

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "reference_calc.h"
#include "compare.h"

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

  std::string reference_file;
  double perPixelError = 0.0;
  double globalError   = 0.0;
  bool useEpsCheck = false;

  switch (argc)
  {
  	case 3:
  	  input_source_file  = std::string(argv[1]);
  	  input_dest_file = std::string(argv[2]);
      output_file = "HW6_output.png";
  	  reference_file = "HW6_reference.png";
  	  break;
  	case 4:
  	  input_source_file  = std::string(argv[1]);
  	  input_dest_file = std::string(argv[2]);
      output_file = std::string(argv[3]);
  	  reference_file = "HW6_reference.png";
  	  break;
  	case 5:
  	  input_source_file  = std::string(argv[1]);
  	  input_dest_file = std::string(argv[2]);
  	  output_file = std::string(argv[3]);
  	  reference_file = std::string(argv[4]);
  	  break;
  	case 7:
  	  useEpsCheck=true;
  	  input_source_file  = std::string(argv[1]);
  	  input_dest_file = std::string(argv[2]);
  	  output_file = std::string(argv[3]);
  	  reference_file = std::string(argv[4]);
  	  perPixelError = atof(argv[5]);
      globalError   = atof(argv[6]);
  	  break;
  	default:
        std::cerr << "Usage: ./HW6 input_source_file input_dest_filename [output_filename] [reference_filename] [perPixelError] [globalError]" << std::endl;
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
  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());
  printf("\n");
  if (err < 0) {
    //Couldn't print! Probably the student closed stdout - bad news
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  //check results and output the tone-mapped image
  postProcess(h_blendedImg, numRowsSource, numColsSource, output_file);

  // calculate the reference image
  uchar4* h_reference = new uchar4[numRowsSource*numColsSource];
  reference_calc(h_sourceImg, numRowsSource, numColsSource,
                   h_destImg, h_reference);

  // save the reference image
  postProcess(h_reference, numRowsSource, numColsSource, reference_file);

  compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);

  delete[] h_reference;
  delete[] h_destImg;
  delete[] h_sourceImg;
  delete[] h_blendedImg;
  return 0;
}

