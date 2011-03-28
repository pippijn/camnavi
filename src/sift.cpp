#include "sift.h"

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/foreach.hpp>

extern "C" {
#include "sift/sift.h"
#include "sift/imgfeatures.h"
}

#include "timer.h"

using cv::Mat;

void
sift (Mat const &src, Mat &dst)
{
  timer const T (__func__);

  cv::SIFT sift;

  cv::vector<cv::KeyPoint> keypoints;
  sift (src, cv::Mat (), keypoints);

  BOOST_FOREACH (cv::KeyPoint const &p, keypoints)
    {
      dst.at<uchar> (p.pt.y, p.pt.x) = 255;
    }
}

void
fast_sift (Mat const &src, Mat &dst)
{
  timer const T (__func__);

  IplImage csrc = src;
  feature *feat;

  int const nfeat = sift_features (&csrc, &feat);

  dst = src;
  draw_features (&csrc, feat, nfeat);

  free (feat);
}
