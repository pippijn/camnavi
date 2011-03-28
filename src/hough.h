#pragma once

#include "cvfwd.h"

struct line_detector
{
  // canny parameters

  // The thresholds. The smallest of threshold1 and threshold2 is used for
  // edge linking, the largest to find initial segments of strong edges.
  double threshold1;
  double threshold2;
  // Aperture parameter for Sobel operator.
  int apertureSize;
  bool L2gradient;


  // hough transform parameters

  // Distance resolution in pixel-related units.
  double rho;
  // Angle resolution measured in radians.
  double theta;
  // Threshold parameter. A line is returned by the function if the
  // corresponding accumulator value is greater than threshold.
  int threshold;
  // The minimum line length.
  double minLineLength;
  // The maximum gap between line segments lieing on the same line to treat
  // them as the single line segment (i.e. to join them).
  double maxLineGap;

  line_detector ();

  void detect_lines (cv::Mat const &src, cv::Mat &color_dst);
  void operator () (cv::Mat const &src, cv::Mat &color_dst)
  {
    detect_lines (src, color_dst);
  }
};
