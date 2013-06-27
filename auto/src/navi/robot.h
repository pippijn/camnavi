#pragma once

#include <memory>
#include <string>

struct Robot
{
  Robot (std::string const &host);

  void run ();

private:
  struct pimpl;
  std::auto_ptr<pimpl> const impl;
};
