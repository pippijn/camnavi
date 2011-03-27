#include "timer.h"

#include <cstdio>

timer::timer (char const *name)
  : name (name)
{
  gettimeofday (&start, NULL);
}

timer::~timer ()
{
  timeval end;
  gettimeofday (&end, NULL);

  timeval diff;
  timersub (&end, &start, &diff);

  printf ("%s: %lu.%06lu sec\n", name, diff.tv_sec, diff.tv_usec);
}
