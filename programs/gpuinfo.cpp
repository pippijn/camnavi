#include <cstdio>

#include <opencv2/gpu/gpu.hpp>

#include "timer.h"

int
main (int argc, char *argv[])
{
  using namespace cv::gpu;

  timer const T (__func__);
  int num_devices = getCudaEnabledDeviceCount ();
  printf ("Detected %d CUDA enabled device(s)\n", num_devices);
  for (int i = 0; i < num_devices; ++i)
    {
      DeviceInfo dev_info (i);
      printf ("- Device #%d (%s, CC %d.%d, %lu/%lu MiB free, %d GPU(s), v",
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
      if (!dev_info.isCompatible ())
        return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}
