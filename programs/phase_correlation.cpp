#include "main.h"

static phase_correlation phase_correlation;

static void
process (Mat const &src, Mat &dst)
{
  phase_correlation (src, dst);
}
