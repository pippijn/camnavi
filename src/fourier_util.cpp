#include "fourier.h"
#include "complex_image.h"
#include "timer.h"

#if 0
#include <opencv2/gpu/gpu.hpp>
using cv::gpu::GpuMat;
#endif

using cv::Mat;

/*
 * src IPL_DEPTH_64F
 * dst IPL_DEPTH_8U
 */
void
detail::dft_plot (Mat const &src, Mat &dst)
{
  assert (src.type () == CV_64FC2);
  assert (dst.type () == CV_8UC1);

  complex_image image (src.size ());

  split (src, image);

  // Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
  // store the result in Re.
  pow (image.re, 2.0, image.re);
  pow (image.im, 2.0, image.im);
  add (image.re, image.im, image.re);
  sqrt (image.re, image.re);

  // Compute log(1 + Mag);
  add (image.re, cv::Scalar (1.0, 0, 0, 0), image.re); // 1 + Mag
  log (image.re, image.re); // log(1 + Mag)

  // Rearrange the quadrants of Fourier image so that the origin is at the image center
  int const rows = src.size ().height;
  int const cols = src.size ().width;
  int const cy = rows / 2; // image center
  int const cx = cols / 2;

  for (int j = 0; j < cy; j++)
    for (int i = 0; i < cx; i++)
      {
        std::swap (image.re.at<double> (j, i     ), image.re.at<double> (j + cy, i + cx));
        std::swap (image.re.at<double> (j, i + cx), image.re.at<double> (j + cy, i     ));
      }

  // Localise minimum and maximum values
  double min;
  double max;
  minMaxLoc (image.re, &min, &max);

  // Normalise image (0 - 255) to be observed as an u8 image
  double const scale = 255 / (max - min);
  double const shift = -min * scale;

  convertScaleAbs (image.re, dst, scale, shift);
}

void
detail::apply_filter (Mat &fourier, Mat const &filtersrc)
{
  assert (fourier.type () == CV_64FC2);
  assert (filtersrc.type () == CV_8UC1);

  Mat channels[] = {
    Mat (filtersrc.size (), CV_64FC1),
    Mat (filtersrc.size (), CV_64FC1),
  };

  filtersrc.convertTo (channels[0], channels[0].type ());
  filtersrc.convertTo (channels[1], channels[1].type ());

  // create a 2-channel image
  Mat filter (filtersrc.size (), fourier.type ());
  merge (channels, 2, filter);

  {
#if 0
    timer const T (__func__);
#endif
    mulSpectrums (fourier, filter, fourier, 0);
  }

#if 0
  fourier.convertTo (fourier, CV_32FC2);
  GpuMat gpuFourier (fourier.size (), fourier.type ());
  gpuFourier.upload (fourier);

  filter.convertTo (filter, CV_32FC2);
  GpuMat gpuFilter (filter.size (), filter.type ());
  gpuFilter.upload (filter);

  {
    timer const T (__func__);
    mulSpectrums (gpuFourier, gpuFilter, gpuFourier, 0);
  }
#endif
}
