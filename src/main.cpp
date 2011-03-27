#include "robot.h"

#include <cstdio>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "fourier.h"
#include "hough.h"
#include "mser.h"

using cv::Mat;

#if 0
#include "kernel.h"

static void
transform (Mat const &src, Mat &dst)
{
  // Make sure the destination is of the same type as the source.
  dst.create (src.size (), src.type ());

#if 0
  // Gaussian convolution kernel
  filter2D (src, dst, 0,
            Kernel
            (1,  4,  7,  4, 1)
            (4, 16, 26, 16, 4)
            (7, 26, 41, 26, 7)
            (4, 16, 26, 16, 4)
            (1,  4,  7,  4, 1)
            / 273);
#elif 1
  // Edge detect vertical
  filter2D (src, dst, 0,
            Kernel
            (-1, 0, 1)
            (-1, 0, 1)
            (-1, 0, 1)
           );
#else
  // Edge detect horizontal
  filter2D (src, dst, 0,
            Kernel
            (-1, -1, -1)
            ( 0,  0,  0)
            ( 1,  1,  1)
           );
#endif
}
#endif

int
main (int argc, char *argv[])
{
#if 0
  std::string hostname = "172.26.1.1";
  //std::string hostname = "172.26.1.2";

  if (argc > 1)
    hostname = argv[1];

  Robot robot (hostname);
  robot.run ();
#else
  cv::namedWindow ("source");
  //cv::namedWindow ("transformed");

  Mat const frequency_filter = cv::imread ("filter.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  line_detector linedet;

  bool paused = false;
  Mat dst;
  for (int i = 0; i <= 395; i++)
    {
      if (paused)
        i--;

      char filename[20];
      sprintf (filename, "images/%03d.jpg", i);

      Mat const src = cv::imread (filename, CV_LOAD_IMAGE_GRAYSCALE);
      dst.create (src.size (), src.type ());
      //equalizeHist (src, src);
      imshow ("source", src);

      //dft_filter (src, frequency_filter, &dst);
      //fft_filter (src, frequency_filter, &dst);
      //fft_filter (src, frequency_filter, NULL, &dst);
      //linedet (src, dst);
      //transform (src, dst);
      mser (src, dst);
      imshow ("transformed", dst);

      switch (cv::waitKey (5000))
        {
        case 'q':
          return 0;
        case 'a':
          i--;
          break;
        case 'e':
          i++;
          break;

          static int const C1 = 1;
          static int const C2 = 5;

          static int const H1 = 18;
          static int const H2 = 8;
          static int const H3 = 3;
          static int const H4 = 1;
        case 'p': linedet.threshold1 -= C1; break;
        case 'i': linedet.threshold1 += C1; break;
        case 'y': linedet.threshold2 -= C2; break;
        case 'u': linedet.threshold2 += C2; break;

        case 'f': linedet.rho -= H1; break;
        case 'h': linedet.rho += H1; break;
        case 'g': linedet.theta -= H2; break;
        case 'd': linedet.theta += H2; break;
        case 'c': linedet.threshold -= H3; break;
        case 'r': linedet.threshold += H3; break;
        case 't': linedet.minLineLength -= H4; break;
        case 'n': linedet.minLineLength += H4; break;
        case 'z': linedet.maxLineGap -= H4; break;
        case 's': linedet.maxLineGap += H4; break;

        case ' ':
          paused = !paused;
          break;
        }
    }
#endif

  return 0;
}
