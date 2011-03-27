#include "hough.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <cstdio>

#include <boost/foreach.hpp>

#include "timer.h"

using cv::Mat;

line_detector::line_detector ()
  // canny parameters
  : threshold1 (5)
  , threshold2 (50)
  , apertureSize (3)
  , L2gradient (false)

  // hough transform parameters
  , rho (1)
  , theta (180)
  , threshold (80)
  , minLineLength (100)
  , maxLineGap (0)
{
}

void
line_detector::operator () (Mat const &src, Mat &colour_dst)
{
  timer const T (__func__);

  // Detect edges using the Canny algorithm
  printf ("Canny (src, dst, %g, %g, %d, %s);\n", threshold1, threshold2, apertureSize, L2gradient ? "true" : "false");
  Mat dst (src.size (), CV_8UC1);
  Canny (src, dst, threshold1, threshold2, apertureSize, L2gradient);

  // Copy the grayscale edge image onto the BGR colour_dst
  colour_dst.create (src.size (), CV_8UC3);
  cvtColor (dst, colour_dst, CV_GRAY2BGR);

  // Find lines on the original binary image created by Canny
  printf ("HoughLinesP (src, lines, %g, CV_PI / %g, %d, %g, %g);\n", rho, theta, threshold, minLineLength, maxLineGap);
  cv::vector<cv::Vec4i> lines;
  HoughLinesP (dst, lines, rho, CV_PI / theta, threshold, minLineLength, maxLineGap);

  // Draw the lines in red colour onto the BGR colour_dst image
  BOOST_FOREACH (cv::Vec4i const &line, lines)
    {
      cv::line (colour_dst,
                cv::Point (line[0], line[1]),
                cv::Point (line[2], line[3]),
                CV_RGB (255, 0, 0),
                3, 8);
    }
}

