#include "fourier.h"
#include "complex_image.h"

using cv::Mat;

static void
dft2 (Mat const &src, Mat &dst)
{
  complex_image image (src.size ());

  // 2 channels (image.re, image.im)
  Mat Fourier (src.size (), CV_64FC2);
  // Real part conversion from u8 to 64f (double)
  src.convertTo (image.re, image.re.type ());
  // Join real and imaginary parts and stock them in Fourier image
  merge (image, 2, Fourier);
  // Application of the forward Fourier transform
  dft (Fourier, dst);
}

void
dft_filter (Mat const &src, Mat const &filter, Mat *dst, Mat *plot)
{
  assert (src.type () == CV_8UC1);

  // forward FT
  Mat fourier (src.size (), CV_64FC2);
  dft2 (src, fourier);

  // apply filters on frequency image
  detail::apply_filter (fourier, filter);

  // plot FT
  if (plot)
    {
      assert (plot->type () == CV_8UC1);
      assert (src.size () == plot->size ());

      detail::dft_plot (fourier, *plot);
    }

  // Calculate inverse DFT and write back to dst
  if (dst)
    {
      assert (dst->type () == CV_8UC1);
      assert (src.size () == dst->size ());

      // inverse DFT
      Mat inverse (src.size (), CV_64FC2);
      dft (fourier, inverse, cv::DFT_SCALE | cv::DFT_INVERSE);

      // split in real and imaginary parts
      complex_image image (src.size ());
      split (inverse, image);

      // Calculate magnitude and assign to re;
      // mag = √(re^2, im^2)
      pow (image.re, 2, image.re);
      pow (image.im, 2, image.im);
      add (image.re, image.im, image.re);
      sqrt (image.re, image.re);

      double min;
      double max;
      minMaxLoc (image.re, &min, &max);

      double const scale = 255 / (max - min);
      double const shift = -min * scale;

      // extract image.re
      image.re.convertTo (*dst, dst->type (), scale, shift);
    }
}
