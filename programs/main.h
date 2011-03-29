#include <cstdio>

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "fast_sift.h"
#include "fourier.h"
#include "goodfeatures.h"
#include "gpusift.h"
#include "gpusurf.h"
#include "hough.h"
#include "mser.h"
#include "phase_correlation.h"
#include "sift.h"
#include "stardetector.h"
#include "surf_analyser.h"
#include "surf.h"

#include "timer.h"

using cv::Mat;

static void process (Mat const &src, Mat &dst);

int
main (int argc, char *argv[])
{
  cv::namedWindow ("source");
  cv::namedWindow ("transformed");

  bool paused = true;
  for (int i = 0; i <= 395; i++)
    {
      char filename[20];
      sprintf (filename, SRCDIR"/images/%03d.jpg", i);

      Mat const src = cv::imread (filename, CV_LOAD_IMAGE_GRAYSCALE);
      assert (src.size () != cv::Size ());
      Mat dst = Mat::zeros (src.size (), src.type ());
      imshow ("source", src);

      process (src, dst);

      sprintf (filename, SRCDIR"/output/%03d.jpg", i);
      imwrite (filename, dst);
      imshow ("transformed", dst);

      switch (cv::waitKey (!paused))
        {
        case 'q':
          return 0;
        case 'a':
          i -= 2;
          break;
        case ' ':
          paused = !paused;
          break;
        }
    }

  cv::destroyWindow ("transformed");
  cv::destroyWindow ("source");

  return 0;
}
