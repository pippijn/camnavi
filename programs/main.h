#include <cstdio>

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

#if !NO_VIEW
static void process (Mat const &src, Mat &dst);
#else
static void process (std::string const &name, Mat const &src);
#endif

int
main (int argc, char *argv[])
{
#if !NO_VIEW
  cv::namedWindow ("source");
  cv::namedWindow ("transformed");
#endif

#if !NO_VIEW
  bool paused = true;
#else
  bool paused = false;
#endif
  for (int i = 0; i <= 395; i++)
    {
      char filename[PATH_MAX];
      snprintf (filename, sizeof filename, SRCDIR"/images/%03d.jpg", i);

      Mat src = cv::imread (filename, CV_LOAD_IMAGE_GRAYSCALE);
      //equalizeHist (src, src);
      assert (src.size () != cv::Size ());
#if !NO_VIEW
      Mat dst = Mat::zeros (src.size (), src.type ());
      imshow ("source", src);

      process (src, dst);

      sprintf (filename, SRCDIR"/output/%03d.jpg", i);
      imwrite (filename, dst);
      imshow ("transformed", dst);
#else
      process (filename, src);
#endif

#if !NO_VIEW
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
#endif
    }

#if !NO_VIEW
  cv::destroyWindow ("transformed");
  cv::destroyWindow ("source");
#endif

  return 0;
}
