#pragma once

#include "cvfwd.h"

#include <memory>

struct gpusift
{
  gpusift ();
  ~gpusift ();

  void operator () (cv::Mat const &src, cv::Mat &dst);

private:
  struct pimpl;
  std::auto_ptr<pimpl> self;
};
