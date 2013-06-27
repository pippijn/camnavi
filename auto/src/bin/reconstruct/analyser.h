#pragma once

#include <memory>

class QImage;

class Analyser
{
public:
  Analyser ();
  ~Analyser ();

  void operator () (QImage const &img);

private:
  struct pimpl;
  std::auto_ptr<pimpl> const self;
};
