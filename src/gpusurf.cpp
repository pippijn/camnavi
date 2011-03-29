#include "sift.h"

#include "gpu/gpusurf/GpuSurfDetector.hpp"

#include <boost/foreach.hpp>

#include "timer.h"

using cv::Mat;

void
gpusurf (Mat const &src, Mat &dst)
{
  timer const T (__func__);

  // Create the configuration object with all default values
  asrl::GpuSurfConfiguration config;

  // Create a detector initialised with the configuration
  asrl::GpuSurfDetector detector (config);

  // Run each step of the SURF algorithm
  detector.buildIntegralImage (src);
  detector.detectKeypoints ();

#if 0
  // Compute descriptors or orientation
  detector.findOrientation ();
  detector.computeDescriptors ();
#endif

  // Retrieve the keypoints from the GPU
  cv::vector<cv::KeyPoint> keypoints;
  detector.getKeypoints (keypoints);

#if 0
  // Retrieve the descriptors from the GPU
  cv::vector<float> descriptors;
  detector.getDescriptors (descriptors);
#endif

  // Draw keypoints onto the picture
  BOOST_FOREACH (cv::KeyPoint const &p, keypoints)
    {
      dst.at<uchar> (p.pt.y, p.pt.x) = 255;
    }
}
