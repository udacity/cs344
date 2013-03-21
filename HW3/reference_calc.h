#ifndef REFERENCE_H__
#define REFERENCE_H__

void referenceCalculation(const float* const h_logLuminance, unsigned int* const h_cdf,
                          const size_t numRows, const size_t numCols, const size_t numBins, 
						  float &logLumMin, float &logLumMax);

#endif
