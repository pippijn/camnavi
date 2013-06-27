#include "goodfeatures.h"

#include <opencv2/imgproc/imgproc.hpp>

#include "foreach.h"
#include "timer.h"

using cv::Mat;

void
goodfeatures (Mat const &src, Mat &dst)
{
  timer const T (__func__);

  cv::vector<cv::Point2f> corners;
  goodFeaturesToTrack (src, corners, 500, 0.01, 10);

  foreach (cv::Point2f const &p, corners)
    dst.at<uchar> (p.y, p.x) = 255;
}
