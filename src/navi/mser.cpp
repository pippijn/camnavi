#include "mser.h"

#include <opencv2/features2d/features2d.hpp>

#include "foreach.h"
#include "timer.h"

using cv::Mat;

void
mser (Mat const &src, Mat &dst)
{
  timer const T (__func__);

  cv::MSER mser (
                 /* delta */ 3,
                 /* min_area */ 60,
                 /* max_area */ 400,
                 /* max_variation */ .25f,
                 /* min_diversity */ .2f,
                 /* max_evolution */ 2000,
                 /* area_threshold */ 1.01,
                 /* min_margin */ .003,
                 /* edge_blur_size */ 5
                );

  cv::vector<cv::vector<cv::Point> > msers;
  mser (src, msers, cv::Mat ());

  foreach (cv::vector<cv::Point> const &r, msers)
    foreach (cv::Point const &p, r)
      dst.at<uchar> (p.y, p.x) = 255;
}
