#pragma once

#include <opencv/cv.h>

struct complex_image
{
  complex_image (cv::Size size)
    : re (size, CV_64FC1)
    , im (size, CV_64FC1)
  {
  }

  operator cv::Mat * () { return &re; }

  // Real part
  cv::Mat re;
  // Imaginary part
  cv::Mat im;
};
