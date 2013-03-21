//Udacity HW 6
//Poisson Blending Reference Calculation

#include "utils.h"
#include <thrust/host_vector.h>

//Performs one iteration of the solver
void computeIteration(const unsigned char* const dstImg,
                      const unsigned char* const strictInteriorPixels,
                      const unsigned char* const borderPixels,
                      const std::vector<uint2>& interiorPixelList,
                      const size_t numColsSource,
                      const float* const f,
                      const float* const g,
                      float* const f_next)
{
  unsigned int off = interiorPixelList[0].x * numColsSource + interiorPixelList[0].y;

  for (size_t i = 0; i < interiorPixelList.size(); ++i) {
    float blendedSum = 0.f;
    float borderSum  = 0.f;

    uint2 coord = interiorPixelList[i];

    unsigned int offset = coord.x * numColsSource + coord.y;

    //process all 4 neighbor pixels
    //for each pixel if it is an interior pixel
    //then we add the previous f, otherwise if it is a
    //border pixel then we add the value of the destination
    //image at the border.  These border values are our boundary
    //conditions.
    if (strictInteriorPixels[offset - 1]) {
      blendedSum += f[offset - 1];
    }
    else {
      borderSum += dstImg[offset - 1];
    }

    if (strictInteriorPixels[offset + 1]) {
      blendedSum += f[offset + 1];
    }
    else {
      borderSum += dstImg[offset + 1];
    }

    if (strictInteriorPixels[offset - numColsSource]) {
      blendedSum += f[offset - numColsSource];
    }
    else {
      borderSum += dstImg[offset - numColsSource];
    }

    if (strictInteriorPixels[offset + numColsSource]) {
      blendedSum += f[offset + numColsSource];
    }
    else {
      borderSum += dstImg[offset + numColsSource];
    }

    float f_next_val = (blendedSum + borderSum + g[offset]) / 4.f;

    f_next[offset] = std::min(255.f, std::max(0.f, f_next_val)); //clip to [0, 255]
  }

}

//pre-compute the values of g, which depend only the source image
//and aren't iteration dependent.
void computeG(const unsigned char* const channel,
              float* const g,
              const size_t numColsSource,
              const std::vector<uint2>& interiorPixelList)
{
  for (size_t i = 0; i < interiorPixelList.size(); ++i) {
    uint2 coord = interiorPixelList[i];
    unsigned int offset = coord.x * numColsSource + coord.y;

    float sum = 4.f * channel[offset];

    sum -= (float)channel[offset - 1] + (float)channel[offset + 1];
    sum -= (float)channel[offset + numColsSource] + (float)channel[offset - numColsSource];

    g[offset] = sum;
  }
}

void reference_calc(const uchar4* const h_sourceImg,
                    const size_t numRowsSource, const size_t numColsSource,
                    const uchar4* const h_destImg,
                    uchar4* const h_blendedImg){

  //we need to create a list of border pixels and interior pixels
  //this is a conceptually simple implementation, not a particularly efficient one...

  //first create mask
  size_t srcSize = numRowsSource * numColsSource;
  unsigned char* mask = new unsigned char[srcSize];

  for (int i = 0; i < srcSize; ++i) {
    mask[i] = (h_sourceImg[i].x + h_sourceImg[i].y + h_sourceImg[i].z < 3 * 255) ? 1 : 0;
  }

  //next compute strictly interior pixels and border pixels
  unsigned char *borderPixels = new unsigned char[srcSize];
  unsigned char *strictInteriorPixels = new unsigned char[srcSize];

  std::vector<uint2> interiorPixelList;

  //the source region in the homework isn't near an image boundary, so we can
  //simplify the conditionals a little...
  for (size_t r = 1; r < numRowsSource - 1; ++r) {
    for (size_t c = 1; c < numColsSource - 1; ++c) {
      if (mask[r * numColsSource + c]) {
        if (mask[(r -1) * numColsSource + c] && mask[(r + 1) * numColsSource + c] &&
            mask[r * numColsSource + c - 1] && mask[r * numColsSource + c + 1]) {
          strictInteriorPixels[r * numColsSource + c] = 1;
          borderPixels[r * numColsSource + c] = 0;
          interiorPixelList.push_back(make_uint2(r, c));
        }
        else {
          strictInteriorPixels[r * numColsSource + c] = 0;
          borderPixels[r * numColsSource + c] = 1;
        }
      }
      else {
          strictInteriorPixels[r * numColsSource + c] = 0;
          borderPixels[r * numColsSource + c] = 0;

      }
    }
  }

  //split the source and destination images into their respective
  //channels
  unsigned char* red_src   = new unsigned char[srcSize];
  unsigned char* blue_src  = new unsigned char[srcSize];
  unsigned char* green_src = new unsigned char[srcSize];

  for (int i = 0; i < srcSize; ++i) {
    red_src[i]   = h_sourceImg[i].x;
    blue_src[i]  = h_sourceImg[i].y;
    green_src[i] = h_sourceImg[i].z;
  }

  unsigned char* red_dst   = new unsigned char[srcSize];
  unsigned char* blue_dst  = new unsigned char[srcSize];
  unsigned char* green_dst = new unsigned char[srcSize];

  for (int i = 0; i < srcSize; ++i) {
    red_dst[i]   = h_destImg[i].x;
    blue_dst[i]  = h_destImg[i].y;
    green_dst[i] = h_destImg[i].z;
  }

  //next we'll precompute the g term - it never changes, no need to recompute every iteration
  float *g_red   = new float[srcSize];
  float *g_blue  = new float[srcSize];
  float *g_green = new float[srcSize];

  memset(g_red,   0, srcSize * sizeof(float));
  memset(g_blue,  0, srcSize * sizeof(float));
  memset(g_green, 0, srcSize * sizeof(float));

  computeG(red_src,   g_red,   numColsSource, interiorPixelList);
  computeG(blue_src,  g_blue,  numColsSource, interiorPixelList);
  computeG(green_src, g_green, numColsSource, interiorPixelList);

  //for each color channel we'll need two buffers and we'll ping-pong between them
  float *blendedValsRed_1 = new float[srcSize];
  float *blendedValsRed_2 = new float[srcSize];

  float *blendedValsBlue_1 = new float[srcSize];
  float *blendedValsBlue_2 = new float[srcSize];

  float *blendedValsGreen_1 = new float[srcSize];
  float *blendedValsGreen_2 = new float[srcSize];

  //IC is the source image, copy over
  for (size_t i = 0; i < srcSize; ++i) {
    blendedValsRed_1[i] = red_src[i];
    blendedValsRed_2[i] = red_src[i];
    blendedValsBlue_1[i] = blue_src[i];
    blendedValsBlue_2[i] = blue_src[i];
    blendedValsGreen_1[i] = green_src[i];
    blendedValsGreen_2[i] = green_src[i];
  }

  //Perform the solve on each color channel
  const size_t numIterations = 800;
  for (size_t i = 0; i < numIterations; ++i) {
    computeIteration(red_dst, strictInteriorPixels, borderPixels,
                     interiorPixelList, numColsSource, blendedValsRed_1, g_red,
                     blendedValsRed_2);

    std::swap(blendedValsRed_1, blendedValsRed_2);
  }

  for (size_t i = 0; i < numIterations; ++i) {
    computeIteration(blue_dst, strictInteriorPixels, borderPixels,
                     interiorPixelList, numColsSource, blendedValsBlue_1, g_blue,
                     blendedValsBlue_2);

    std::swap(blendedValsBlue_1, blendedValsBlue_2);
  }

  for (size_t i = 0; i < numIterations; ++i) {
    computeIteration(green_dst, strictInteriorPixels, borderPixels,
                     interiorPixelList, numColsSource, blendedValsGreen_1, g_green,
                     blendedValsGreen_2);

    std::swap(blendedValsGreen_1, blendedValsGreen_2);
  }
  std::swap(blendedValsRed_1,   blendedValsRed_2);   //put output into _2
  std::swap(blendedValsBlue_1,  blendedValsBlue_2);  //put output into _2
  std::swap(blendedValsGreen_1, blendedValsGreen_2); //put output into _2

  //copy the destination image to the output
  memcpy(h_blendedImg, h_destImg, sizeof(uchar4) * srcSize);

  //copy computed values for the interior into the output
  for (size_t i = 0; i < interiorPixelList.size(); ++i) {
    uint2 coord = interiorPixelList[i];

    unsigned int offset = coord.x * numColsSource + coord.y;

    h_blendedImg[offset].x = blendedValsRed_2[offset];
    h_blendedImg[offset].y = blendedValsBlue_2[offset];
    h_blendedImg[offset].z = blendedValsGreen_2[offset];
  }

  //wow, we allocated a lot of memory!
  delete[] mask;
  delete[] blendedValsRed_1;
  delete[] blendedValsRed_2;
  delete[] blendedValsBlue_1;
  delete[] blendedValsBlue_2;
  delete[] blendedValsGreen_1;
  delete[] blendedValsGreen_2;
  delete[] g_red;
  delete[] g_blue;
  delete[] g_green;
  delete[] red_src;
  delete[] red_dst;
  delete[] blue_src;
  delete[] blue_dst;
  delete[] green_src;
  delete[] green_dst;
  delete[] borderPixels;
  delete[] strictInteriorPixels;
}

