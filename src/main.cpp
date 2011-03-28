#include "robot.h"

#include <cstdio>

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "fourier.h"
#include "hough.h"
#include "mser.h"
#include "surf.h"
#include "sift.h"
#include "stardetector.h"
#include "goodfeatures.h"

#include "timer.h"

using cv::Mat;

static bool
init_gpu (bool verbose = false)
{
  using namespace cv::gpu;

  timer const T (__func__);
  int num_devices = getCudaEnabledDeviceCount ();
  printf ("Detected %d CUDA enabled device(s)\n", num_devices);
  for (int i = 0; i < num_devices; ++i)
    {
      DeviceInfo dev_info (i);
      if (verbose)
        {
          printf ("- Device #%d (%s, CC %d.%d, %lu/%lu MiB free, %d GPUs, v",
                  i,
                  dev_info.name ().c_str (),
                  dev_info.majorVersion (),
                  dev_info.minorVersion (),
                  dev_info.freeMemory () / 1024 / 1024,
                  dev_info.totalMemory () / 1024 / 1024,
                  dev_info.multiProcessorCount ());
          if (dev_info.supports (FEATURE_SET_COMPUTE_21))
            printf ("2.1 (global atomics, native doubles)");
          else if (dev_info.supports (FEATURE_SET_COMPUTE_20))
            printf ("2.0 (global atomics, native doubles)");
          else if (dev_info.supports (FEATURE_SET_COMPUTE_13))
            printf ("1.3 (global atomics, native doubles)");
          else if (dev_info.supports (FEATURE_SET_COMPUTE_12))
            printf ("1.2 (global atomics)");
          else if (dev_info.supports (FEATURE_SET_COMPUTE_11))
            printf ("1.1 (global atomics)");
          else if (dev_info.supports (FEATURE_SET_COMPUTE_10))
            printf ("1.0");
          puts (")");
        }
      if (!dev_info.isCompatible ())
        return false;
    }
  return true;
}

int
main (int argc, char *argv[])
{
  if (!init_gpu ())
    return EXIT_FAILURE;
#if 0
  std::string hostname = "172.26.1.1";
  //std::string hostname = "172.26.1.2";

  if (argc > 1)
    hostname = argv[1];

  Robot robot (hostname);
  robot.run ();
#else
  cv::namedWindow ("source");
  cv::namedWindow ("transformed");

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
      //dst.create (src.size (), src.type ());
      dst = Mat::zeros (src.size (), src.type ());
      //equalizeHist (src, src);
      imshow ("source", src);

      dft_filter (src, frequency_filter, &dst);
      dft_filter (src, frequency_filter, NULL, &dst);
      fft_filter (src, frequency_filter, &dst);
      fft_filter (src, frequency_filter, NULL, &dst);
      linedet (src, dst);
      mser (src, dst);
      surf (src, dst);
      stardetector (src, dst);
      goodfeatures (src, dst);
      fast_sift (src, dst);
      sift (src, dst);

      sprintf (filename, "output/%03d.jpg", i);
      imwrite (filename, dst);
      imshow ("transformed", dst);

      switch (cv::waitKey (50))
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

  cv::destroyWindow ("transformed");
  cv::destroyWindow ("source");
#endif

  return 0;
}
