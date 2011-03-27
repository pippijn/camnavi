#include "mser.h"

#include <boost/foreach.hpp>

using cv::Mat;

void
mser (Mat const &src, Mat &dst)
{
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

  dst = Mat::zeros (src.size (), src.type ());
  BOOST_FOREACH (cv::vector<cv::Point> const &r, msers)
    {
      BOOST_FOREACH (cv::Point const &p, r)
        {
          dst.at<uchar> (p.y, p.x) = 255;
        }
    }
}
