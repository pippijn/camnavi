#pragma once

#include <memory>
#include "cvfwd.h"

struct phase_correlation
{
  phase_correlation ();
  ~phase_correlation ();

  void operator () (cv::Mat const &src, cv::Mat &dst);

private:
  struct pimpl;
  std::auto_ptr<pimpl> self;
};
