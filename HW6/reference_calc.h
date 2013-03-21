#ifndef REFERENCE_H__
#define REFERENCE_H__

void reference_calc(const uchar4* const h_sourceImg,
                    const size_t numRowsSource, const size_t numColsSource,
                    const uchar4* const h_destImg,
                      uchar4* const h_blendedImg);

#endif
