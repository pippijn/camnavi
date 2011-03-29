#include "main.h"

static surf_analyser surf_analyser;

static void
process (Mat const &src, Mat &dst)
{
  surf_analyser (src, dst);
}
