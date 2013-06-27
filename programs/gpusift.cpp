#include "main.h"

static gpusift gpusifter;

static void
process (Mat const &src, Mat &dst)
{
  gpusifter (src, dst);
}
