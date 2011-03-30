#pragma once

#include <memory>
#include "cvfwd.h"

struct surf_analyser
{
  surf_analyser ();
  ~surf_analyser ();

  void operator () (cv::Mat const &src, cv::Mat &dst);

private:
  struct pimpl;
  std::auto_ptr<pimpl> self;
};
