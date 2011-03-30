#pragma once

#include <sys/time.h>

struct timer
{
  char const *name;
  timeval start;

  timer (char const *name);
  ~timer ();
};
