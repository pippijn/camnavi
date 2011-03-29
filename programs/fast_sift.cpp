#include "main.h"

static void
process (Mat const &src, Mat &dst)
{
  fast_sift (src, dst);
}
