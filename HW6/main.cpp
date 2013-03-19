//Udacity HW6 Driver

#include <iostream>
#include "timer.h"
#include "utils.h"
#include <string>
#include <stdio.h>

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
