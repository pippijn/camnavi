#include "kernel.h"

#include <opencv2/imgproc/imgproc.hpp>

using cv::Mat;

static void
transform1 (Mat const &src, Mat &dst)
{
  // Make sure the destination is of the same type as the source.
  dst.create (src.size (), src.type ());

  // Gaussian convolution kernel
  filter2D (src, dst, 0,
            Kernel
            (1,  4,  7,  4, 1)
            (4, 16, 26, 16, 4)
            (7, 26, 41, 26, 7)
            (4, 16, 26, 16, 4)
            (1,  4,  7,  4, 1)
            / 273);
}

static void
transform2 (Mat const &src, Mat &dst)
{
  // Make sure the destination is of the same type as the source.
  dst.create (src.size (), src.type ());

  // Edge detect vertical
  filter2D (src, dst, 0,
            Kernel
            (-1, 0, 1)
            (-1, 0, 1)
            (-1, 0, 1)
           );
}

static void
transform3 (Mat const &src, Mat &dst)
{
  // Make sure the destination is of the same type as the source.
  dst.create (src.size (), src.type ());

  // Edge detect horizontal
  filter2D (src, dst, 0,
            Kernel
            (-1, -1, -1)
            ( 0,  0,  0)
            ( 1,  1,  1)
           );
}
