#include "phase_correlation.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include <fftw3.h>

#include "fourier.h"
#include "timer.h"

using cv::Mat;

extern "C" void phase_correlation (IplImage *ref, IplImage *tpl, IplImage *poc);

struct phase_correlation::pimpl
{
  Mat ref;

  pimpl ()
  {
  }

  void phase_correlation (Mat const &src, Mat &dst);
};

void
phase_correlation::pimpl::phase_correlation (Mat const &src, Mat &dst)
{
  timer const T (__func__);
  if (ref.size () != cv::Size ())
    {
      IplImage tpl = src;
      IplImage ref = this->ref;

      /* create a new image, to store phase correlation result */
      IplImage *poc = cvCreateImage (cvSize (tpl.width, tpl.height), IPL_DEPTH_64F, 1);

      /* get phase correlation of input images */
      ::phase_correlation (&ref, &tpl, poc);

      /* find the maximum value and its location */
      CvPoint minloc, maxloc;
      double minval, maxval;
      cvMinMaxLoc (poc, &minval, &maxval, &minloc, &maxloc, 0);

      cvReleaseImage (&poc);

      /* print it */
      fprintf (stdout, "Maxval at (%d, %d) = %2.4f\n", maxloc.x, maxloc.y, maxval);
    }

  ref = src;
}


phase_correlation::phase_correlation ()
  : self (new pimpl)
{
}

phase_correlation::~phase_correlation ()
{
}

void
phase_correlation::operator () (Mat const &src, Mat &dst)
{
  self->phase_correlation (src, dst);
}
