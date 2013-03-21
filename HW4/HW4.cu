#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>

#include "loadSaveImage.h"
#include <stdio.h>

//simple cross correlation kernel copied from Mike's IPython Notebook
__global__ void naive_normalized_cross_correlation(
    float*         d_response,
    unsigned char* d_original,
    unsigned char* d_template,
    int            num_pixels_y,
    int            num_pixels_x,
    int            template_half_height,
    int            template_height,
    int            template_half_width,
    int            template_width,
    int            template_size,
    float          template_mean
    )
{
  int  ny             = num_pixels_y;
  int  nx             = num_pixels_x;
  int  knx            = template_width;
  int2 image_index_2d = make_int2( ( blockIdx.x * blockDim.x ) + threadIdx.x, ( blockIdx.y * blockDim.y ) + threadIdx.y );
  int  image_index_1d = ( nx * image_index_2d.y ) + image_index_2d.x;

  if ( image_index_2d.x < nx && image_index_2d.y < ny )
  {
    //
    // compute image mean
    //
    float image_sum = 0.0f;

    for ( int y = -template_half_height; y <= template_half_height; y++ )
    {
      for ( int x = -template_half_width; x <= template_half_width; x++ )
      {
        int2 image_offset_index_2d         = make_int2( image_index_2d.x + x, image_index_2d.y + y );
        int2 image_offset_index_2d_clamped = make_int2( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
        int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;

        unsigned char image_offset_value = d_original[ image_offset_index_1d_clamped ];

        image_sum += (float)image_offset_value;
      }
    }

    float image_mean = image_sum / (float)template_size;

    //
    // compute sums
    //
    float sum_of_image_template_diff_products = 0.0f;
    float sum_of_squared_image_diffs          = 0.0f;
    float sum_of_squared_template_diffs       = 0.0f;

    for ( int y = -template_half_height; y <= template_half_height; y++ )
    {
      for ( int x = -template_half_width; x <= template_half_width; x++ )
      {
        int2 image_offset_index_2d         = make_int2( image_index_2d.x + x, image_index_2d.y + y );
        int2 image_offset_index_2d_clamped = make_int2( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
        int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;

        unsigned char image_offset_value = d_original[ image_offset_index_1d_clamped ];
        float         image_diff         = (float)image_offset_value - image_mean;

        int2 template_index_2d = make_int2( x + template_half_width, y + template_half_height );
        int  template_index_1d = ( knx * template_index_2d.y ) + template_index_2d.x;

        unsigned char template_value = d_template[ template_index_1d ];
        float         template_diff  = template_value - template_mean;

        float image_template_diff_product = image_offset_value   * template_diff;
        float squared_image_diff          = image_diff           * image_diff;
        float squared_template_diff       = template_diff        * template_diff;

        sum_of_image_template_diff_products += image_template_diff_product;
        sum_of_squared_image_diffs          += squared_image_diff;
        sum_of_squared_template_diffs       += squared_template_diff;
      }
    }


    //
    // compute final result
    //
    float result_value = 0.0f;

    if ( sum_of_squared_image_diffs != 0 && sum_of_squared_template_diffs != 0 )
    {
      result_value = sum_of_image_template_diff_products / sqrt( sum_of_squared_image_diffs * sum_of_squared_template_diffs );
    }

    d_response[ image_index_1d ] = result_value;
  }
}


__global__ void remove_redness_from_coordinates(
    const unsigned int*  d_coordinates,
    unsigned char* d_r,
    unsigned char* d_b,
    unsigned char* d_g,
    unsigned char* d_r_output,
    int    num_coordinates,
    int    num_pixels_y,
    int    num_pixels_x,
    int    template_half_height,
    int    template_half_width
    )
{
  int ny              = num_pixels_y;
  int nx              = num_pixels_x;
  int global_index_1d = ( blockIdx.x * blockDim.x ) + threadIdx.x;

  int imgSize = num_pixels_x * num_pixels_y;

  if ( global_index_1d < num_coordinates )
  {
    unsigned int image_index_1d = d_coordinates[ imgSize - global_index_1d - 1 ];
    ushort2 image_index_2d = make_ushort2(image_index_1d % num_pixels_x, image_index_1d / num_pixels_x);

    for ( int y = image_index_2d.y - template_half_height; y <= image_index_2d.y + template_half_height; y++ )
    {
      for ( int x = image_index_2d.x - template_half_width; x <= image_index_2d.x + template_half_width; x++ )
      {
        int2 image_offset_index_2d         = make_int2( x, y );
        int2 image_offset_index_2d_clamped = make_int2( min( nx - 1, max( 0, image_offset_index_2d.x ) ), min( ny - 1, max( 0, image_offset_index_2d.y ) ) );
        int  image_offset_index_1d_clamped = ( nx * image_offset_index_2d_clamped.y ) + image_offset_index_2d_clamped.x;

        unsigned char g_value = d_g[ image_offset_index_1d_clamped ];
        unsigned char b_value = d_b[ image_offset_index_1d_clamped ];

        unsigned int gb_average = ( g_value + b_value ) / 2;

        d_r_output[ image_offset_index_1d_clamped ] = (unsigned char)gb_average;
      }
    }

  }
}




struct splitChannels : thrust::unary_function<uchar4, thrust::tuple<unsigned char, unsigned char, unsigned char> >{
  __host__ __device__
  thrust::tuple<unsigned char, unsigned char, unsigned char> operator()(uchar4 pixel) {
    return thrust::make_tuple(pixel.x, pixel.y, pixel.z);
  }
};

struct combineChannels : thrust::unary_function<thrust::tuple<unsigned char, unsigned char, unsigned char>, uchar4> {
  __host__ __device__
  uchar4 operator()(thrust::tuple<unsigned char, unsigned char, unsigned char> t) {
    return make_uchar4(thrust::get<0>(t), thrust::get<1>(t), thrust::get<2>(t), 255);
  }
};

struct combineResponses : thrust::unary_function<float, thrust::tuple<float, float, float> > {
  __host__ __device__
  float operator()(thrust::tuple<float, float, float> t) {
    return thrust::get<0>(t) * thrust::get<1>(t) * thrust::get<2>(t);
  }
};

//we need to save the input so we can remove the redeye for the output
static thrust::device_vector<unsigned char> d_red;
static thrust::device_vector<unsigned char> d_blue;
static thrust::device_vector<unsigned char> d_green;

static size_t numRowsImg;
static size_t numColsImg;
static size_t templateHalfWidth;
static size_t templateHalfHeight;

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
void preProcess(unsigned int **inputVals,
                unsigned int **inputPos,
                unsigned int **outputVals,
                unsigned int **outputPos,
                size_t &numElem,
                const std::string& filename,
				const std::string& templateFilename) {
  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  uchar4 *inImg;
  uchar4 *eyeTemplate;

  size_t numRowsTemplate, numColsTemplate;

  loadImageRGBA(filename, &inImg, &numRowsImg, &numColsImg);
  loadImageRGBA(templateFilename, &eyeTemplate, &numRowsTemplate, &numColsTemplate);

  templateHalfWidth = (numColsTemplate - 1) / 2;
  templateHalfHeight = (numRowsTemplate - 1) / 2;

  //we need to split each image into its separate channels
  //use thrust to demonstrate basic uses

  numElem = numRowsImg * numColsImg;
  size_t templateSize = numRowsTemplate * numColsTemplate;

  thrust::device_vector<uchar4> d_Img(inImg, inImg + numRowsImg * numColsImg);
  thrust::device_vector<uchar4> d_Template(eyeTemplate, eyeTemplate + numRowsTemplate * numColsTemplate);

  d_red.  resize(numElem);
  d_blue. resize(numElem);
  d_green.resize(numElem);

  thrust::device_vector<unsigned char> d_red_template(templateSize);
  thrust::device_vector<unsigned char> d_blue_template(templateSize);
  thrust::device_vector<unsigned char> d_green_template(templateSize);

  //split the image
  thrust::transform(d_Img.begin(), d_Img.end(), thrust::make_zip_iterator(
                                                  thrust::make_tuple(d_red.begin(),
                                                                     d_blue.begin(),
                                                                     d_green.begin())),
                                                splitChannels());

  //split the template
  thrust::transform(d_Template.begin(), d_Template.end(), 
                    thrust::make_zip_iterator(thrust::make_tuple(d_red_template.begin(),
                                                                 d_blue_template.begin(),
                                                                 d_green_template.begin())),
                                                splitChannels());

  
  thrust::device_vector<float> d_red_response(numElem);
  thrust::device_vector<float> d_blue_response(numElem);
  thrust::device_vector<float> d_green_response(numElem);

  //need to compute the mean for each template channel
  unsigned int r_sum = thrust::reduce(d_red_template.begin(), d_red_template.end(), 0);
  unsigned int b_sum = thrust::reduce(d_blue_template.begin(), d_blue_template.end(), 0);
  unsigned int g_sum = thrust::reduce(d_green_template.begin(), d_green_template.end(), 0);

  float r_mean = (double)r_sum / templateSize;
  float b_mean = (double)b_sum / templateSize;
  float g_mean = (double)g_sum / templateSize;

  const dim3 blockSize(32, 8, 1);
  const dim3 gridSize( (numColsImg + blockSize.x - 1) / blockSize.x, (numRowsImg + blockSize.y - 1) / blockSize.y, 1);

  //now compute the cross-correlations for each channel

  naive_normalized_cross_correlation<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_red_response.data()),
                                                              thrust::raw_pointer_cast(d_red.data()),
                                                              thrust::raw_pointer_cast(d_red_template.data()),
                                                              numRowsImg, numColsImg,
                                                              templateHalfHeight, numRowsTemplate,
                                                              templateHalfWidth, numColsTemplate,
                                                              numRowsTemplate * numColsTemplate, r_mean);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
                                                             
  naive_normalized_cross_correlation<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_blue_response.data()),
                                                              thrust::raw_pointer_cast(d_blue.data()),
                                                              thrust::raw_pointer_cast(d_blue_template.data()),
                                                              numRowsImg, numColsImg,
                                                              templateHalfHeight, numRowsTemplate,
                                                              templateHalfWidth, numColsTemplate,
                                                              numRowsTemplate * numColsTemplate, b_mean);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  naive_normalized_cross_correlation<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_green_response.data()),
                                                              thrust::raw_pointer_cast(d_green.data()),
                                                              thrust::raw_pointer_cast(d_green_template.data()),
                                                              numRowsImg, numColsImg,
                                                              templateHalfHeight, numRowsTemplate,
                                                              templateHalfWidth, numColsTemplate,
                                                              numRowsTemplate * numColsTemplate, g_mean);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //generate combined response - multiply all channels together


  thrust::device_vector<float> d_combined_response(numElem);

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                        d_red_response.begin(),
                        d_blue_response.begin(),
                        d_green_response.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                        d_red_response.end(),
                        d_blue_response.end(),
                        d_green_response.end())),
                    d_combined_response.begin(),
                    combineResponses());

  //find max/min of response

  typedef thrust::device_vector<float>::iterator floatIt;
  thrust::pair<floatIt, floatIt> minmax = thrust::minmax_element(d_combined_response.begin(), d_combined_response.end());

  float bias = *minmax.first;

  //we need to make all the numbers positive so that the students can sort them without any bit twiddling
  thrust::transform(d_combined_response.begin(), d_combined_response.end(), thrust::make_constant_iterator(-bias), 
                    d_combined_response.begin(), thrust::plus<float>());

  //now we need to create the 1-D coordinates that will be attached to the keys
  thrust::device_vector<unsigned int> coords(numElem);
  thrust::sequence(coords.begin(), coords.end()); //[0, ..., numElem - 1]

  //allocate memory for output and copy since our device vectors will go out of scope
  //and be deleted
  checkCudaErrors(cudaMalloc(inputVals,  sizeof(unsigned int) * numElem));
  checkCudaErrors(cudaMalloc(inputPos,   sizeof(unsigned int) * numElem));
  checkCudaErrors(cudaMalloc(outputVals, sizeof(unsigned int) * numElem));
  checkCudaErrors(cudaMalloc(outputPos,  sizeof(unsigned int) * numElem));

  cudaMemcpy(*inputVals, thrust::raw_pointer_cast(d_combined_response.data()), sizeof(unsigned int) * numElem, cudaMemcpyDeviceToDevice);
  cudaMemcpy(*inputPos,  thrust::raw_pointer_cast(coords.data()), sizeof(unsigned int) * numElem, cudaMemcpyDeviceToDevice);
  checkCudaErrors(cudaMemset(*outputVals, 0, sizeof(unsigned int) * numElem));
  checkCudaErrors(cudaMemset(*outputPos, 0,  sizeof(unsigned int) * numElem));
}

void postProcess(const unsigned int* const outputVals,
                 const unsigned int* const outputPos,
                 const size_t numElems,
                 const std::string& output_file){

  thrust::device_vector<unsigned char> d_output_red = d_red;

  const dim3 blockSize(256, 1, 1);
  const dim3 gridSize( (40 + blockSize.x - 1) / blockSize.x, 1, 1);

  remove_redness_from_coordinates<<<gridSize, blockSize>>>(outputPos,
                                                           thrust::raw_pointer_cast(d_red.data()),
                                                           thrust::raw_pointer_cast(d_blue.data()),
                                                           thrust::raw_pointer_cast(d_green.data()),
                                                           thrust::raw_pointer_cast(d_output_red.data()),
                                                           40,
                                                           numRowsImg, numColsImg,
                                                           9, 9);


  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //combine the new red channel with original blue and green for output
  thrust::device_vector<uchar4> d_outputImg(numElems);

  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                          d_output_red.begin(),
                          d_blue.begin(),
                          d_green.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(
                          d_output_red.end(),
                          d_blue.end(),
                          d_green.end())),
                    d_outputImg.begin(),
                    combineChannels());

  thrust::host_vector<uchar4> h_Img = d_outputImg;

  saveImageRGBA(&h_Img[0], numRowsImg, numColsImg, output_file);

  //Clear the global vectors otherwise something goes wrong trying to free them
  d_red.clear(); d_red.shrink_to_fit();
  d_blue.clear(); d_blue.shrink_to_fit();
  d_green.clear(); d_green.shrink_to_fit();
}

