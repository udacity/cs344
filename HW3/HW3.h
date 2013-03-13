#ifndef HW3_H__
#define HW3_H__

void compareImages(std::string reference_filename, std::string test_filename, bool useEpsCheck,
				   double perPixelError, double globalError);

void generateReferenceImage(std::string reference_file, const float* const h_logLuminance, unsigned int* const h_cdf,
                            const size_t numRows, const size_t numCols, const size_t numBins);

#endif
