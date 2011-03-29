#include "fast_sift.h"

#include <opencv2/core/core.hpp>

extern "C" {
#include "sift/sift.h"
#include "sift/imgfeatures.h"
}

#include "timer.h"

using cv::Mat;

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
