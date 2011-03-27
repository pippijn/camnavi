#include "fourier.h"
#include "timer.h"

#include <fftw3.h>

using cv::Mat;

static void
store_wisdom ()
{
  if (FILE *wisdom = fopen ("wisdom.lsp", "w"))
    {
      fftw_export_wisdom_to_file (wisdom);
      fclose (wisdom);
    }
}

static void
load_wisdom ()
{
  if (FILE *wisdom = fopen ("wisdom.lsp", "r"))
    {
      fftw_import_wisdom_from_file (wisdom);
      fclose (wisdom);
    }
}

struct fftw
{
  fftw_complex *const data_in;
  fftw_complex *const fft;
  fftw_complex *const ifft;

  fftw_plan const plan_f;
  fftw_plan const plan_b;

  fftw (cv::Size size)
    /* initialise arrays for fftw operations */
    : data_in (alloc_array (size))
    ,  fft    (alloc_array (size))
    , ifft    (alloc_array (size))

    /* create plans */
    , plan_f (make_plan (size, data_in,  fft, FFTW_FORWARD ))
    , plan_b (make_plan (size, fft    , ifft, FFTW_BACKWARD))
  {
    puts ("fourier transform plans initialised");
    store_wisdom ();
  }

  ~fftw ()
  {
    /* free memory */
    fftw_destroy_plan (plan_b);
    fftw_destroy_plan (plan_f);
    fftw_free (ifft);
    fftw_free (fft);
    fftw_free (data_in);
  }

  fftw_complex *alloc_array (cv::Size size)
  {
    return (fftw_complex *)fftw_malloc (sizeof (fftw_complex) * size.width * size.height);
  }

  fftw_plan make_plan (cv::Size size, fftw_complex *in, fftw_complex *out, int sign)
  {
    timer const t ("planning");
    printf ("planning %sfourier transform...\n", sign == FFTW_BACKWARD ? "inverse " : "");
    fftw_plan plan = fftw_plan_dft_1d (size.width * size.height, in, out, sign, FFTW_PATIENT);
    return plan;
  }
};

void
fft_filter (cv::Mat const &src, cv::Mat const &filter, cv::Mat *dst, cv::Mat *plot)
{
  load_wisdom ();

  static fftw const ft (src.size ());
  int const width  = src.size ().width;
  int const height = src.size ().height;

  timer const t ("fourier transform");

  /* load img1's data to fftw input */
  for (int i = 0, k = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      {
        ft.data_in[k][0] = src.at<uchar> (i, j);
        ft.data_in[k][1] = 0.0;
        k++;
      }

  /* perform FFT */
  fftw_execute (ft.plan_f);

  Mat fourier (src.size (), CV_64FC2);
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++)
      {
        fourier.at<double> (y, x        ) = ft.fft[y * width + x][0];
        fourier.at<double> (y, x + width) = ft.fft[y * width + x][1];
      }

  if (plot)
    // plot FT
    detail::dft_plot (fourier, *plot);

  /* perform IFFT */
  fftw_execute (ft.plan_b);

  /* normalise IFFT result */
  for (int i = 0; i < width * height; i++)
    ft.ifft[i][0] /= width * height;

  /* copy IFFT result to img2's data */
  if (dst)
    {
      dst->create (src.size (), src.type ());
      for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
          dst->at<uchar> (i, j) = ft.ifft[i * width + j][0];
    }
}
