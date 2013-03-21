#include <algorithm>
#include <cassert>

void referenceCalculation(const float* const h_logLuminance, unsigned int* const h_cdf,
                          const size_t numRows, const size_t numCols, const size_t numBins, 
						  float &logLumMin, float &logLumMax)
{
  logLumMin = h_logLuminance[0];
  logLumMax = h_logLuminance[0];

  //Step 1
  //first we find the minimum and maximum across the entire image
  for (size_t i = 1; i < numCols * numRows; ++i) {
    logLumMin = std::min(h_logLuminance[i], logLumMin);
    logLumMax = std::max(h_logLuminance[i], logLumMax);
  }

  //Step 2
  float logLumRange = logLumMax - logLumMin;

  //Step 3
  //next we use the now known range to compute
  //a histogram of numBins bins
  unsigned int *histo = new unsigned int[numBins];

  for (size_t i = 0; i < numBins; ++i) histo[i] = 0;

  for (size_t i = 0; i < numCols * numRows; ++i) {
    unsigned int bin = std::min(static_cast<unsigned int>(numBins - 1),
                           static_cast<unsigned int>((h_logLuminance[i] - logLumMin) / logLumRange * numBins));
    histo[bin]++;
  }

  //Step 4
  //finally we perform and exclusive scan (prefix sum)
  //on the histogram to get the cumulative distribution
  h_cdf[0] = 0;
  for (size_t i = 1; i < numBins; ++i) {
    h_cdf[i] = h_cdf[i - 1] + histo[i - 1];
  }

  delete[] histo;
}