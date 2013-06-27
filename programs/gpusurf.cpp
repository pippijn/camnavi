#include "main.h"

static void
process (Mat const &src, Mat &dst)
{
  gpusurf (src, dst);
}
