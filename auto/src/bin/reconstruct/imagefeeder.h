#pragma once

#include <memory>

class QImage;

struct ImageFeeder
{
  ImageFeeder ();
  ~ImageFeeder ();

  QImage const &next ();

private:
  struct pimpl;
  std::auto_ptr<pimpl> const self;
};
