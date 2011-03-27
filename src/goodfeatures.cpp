#include "goodfeatures.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <boost/foreach.hpp>

#include "timer.h"

using cv::Mat;

void
goodfeatures (Mat const &src, Mat &dst)
{
  timer const T (__func__);

  cv::vector<cv::Point2f> corners;
  goodFeaturesToTrack (src, corners, 500, 0.01, 10);

  BOOST_FOREACH (cv::Point2f const &p, corners)
    {
      if (p.y > 3 && p.x > 3)
        {
          dst.at<uchar> (p.y - 3, p.x - 3) = 0;
          dst.at<uchar> (p.y - 2, p.x - 3) = 0;
          dst.at<uchar> (p.y - 1, p.x - 3) = 0;
          dst.at<uchar> (p.y - 0, p.x - 3) = 0;
          dst.at<uchar> (p.y - 3, p.x - 2) = 0;
          dst.at<uchar> (p.y - 2, p.x - 2) = 0;
          dst.at<uchar> (p.y - 1, p.x - 2) = 0;
          dst.at<uchar> (p.y - 0, p.x - 2) = 0;
          dst.at<uchar> (p.y - 3, p.x - 1) = 0;
          dst.at<uchar> (p.y - 2, p.x - 1) = 0;
          dst.at<uchar> (p.y - 1, p.x - 1) = 0;
          dst.at<uchar> (p.y - 0, p.x - 1) = 0;
          dst.at<uchar> (p.y - 3, p.x - 0) = 0;
          dst.at<uchar> (p.y - 2, p.x - 0) = 0;
          dst.at<uchar> (p.y - 1, p.x - 0) = 0;
          dst.at<uchar> (p.y - 0, p.x - 0) = 0;
        }
    }
}
