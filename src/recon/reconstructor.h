#pragma once

#include "cvfwd.h"

#include <iosfwd>
#include <memory>

struct reconstructor
{
  reconstructor ();
  ~reconstructor ();

  void operator () (std::string const &name, cv::Mat const &frame);

private:
  struct pimpl;
  std::auto_ptr<pimpl> const self;
};
