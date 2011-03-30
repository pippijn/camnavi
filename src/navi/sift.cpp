#include "sift.h"

#include <opencv2/features2d/features2d.hpp>

#include "foreach.h"
#include "timer.h"

using cv::Mat;

void
sift (Mat const &src, Mat &dst)
{
  timer const T (__func__);

  cv::SIFT sift;

  cv::vector<cv::KeyPoint> keypoints;
  sift (src, cv::Mat (), keypoints);

  foreach (cv::KeyPoint const &p, keypoints)
    dst.at<uchar> (p.pt.y, p.pt.x) = 255;
}
