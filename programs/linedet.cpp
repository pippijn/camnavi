#include "main.h"

static line_detector linedet;

static void
process (Mat const &src, Mat &dst)
{
  linedet (src, dst);
}
