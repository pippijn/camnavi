#pragma once

#include <opencv/cv.h>

namespace detail
{
  void apply_filter (cv::Mat &fourier, cv::Mat const &filtersrc);

  /*
   * src IPL_DEPTH_64F
   * dst IPL_DEPTH_8U
   */
  void dft_plot (cv::Mat const &src, cv::Mat &dst);
}

void dft_filter (cv::Mat const &src, cv::Mat const &filter, cv::Mat *dst, cv::Mat *plot = NULL);
void fft_filter (cv::Mat const &src, cv::Mat const &filter, cv::Mat *dst, cv::Mat *plot = NULL);
