#pragma once

#include <memory>
#include <string>

struct Robot
{
  Robot (std::string const &host);

  void run ();


  struct pimpl;
  std::auto_ptr<pimpl> impl;
};
