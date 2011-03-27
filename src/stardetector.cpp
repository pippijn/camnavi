#include "stardetector.h"

#include <opencv2/features2d/features2d.hpp>

#include <boost/foreach.hpp>

#include "timer.h"

using cv::Mat;

void
stardetector (Mat const &src, Mat &dst)
{
  timer const T (__func__);

  cv::StarDetector stardetector;

  cv::vector<cv::KeyPoint> keypoints;
  stardetector (src, keypoints);

  BOOST_FOREACH (cv::KeyPoint const &p, keypoints)
    {
      dst.at<uchar> (p.pt.y, p.pt.x) = 255;
    }
}
