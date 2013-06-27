#include "reconstructor.h"

#include <string>
#include <cstdio>

#include "texture.h"



using cv::Mat;

struct reconstructor::pimpl
{
  pimpl ();
  ~pimpl ();

  void process_frame (std::string const &name, Mat const &frame);
};


reconstructor::pimpl::pimpl ()
{
}

reconstructor::pimpl::~pimpl ()
{
}


void
reconstructor::pimpl::process_frame (std::string const &name, Mat const &frame)
{
  printf ("processing %s\n", name.c_str ());
}


reconstructor::reconstructor ()
  : self (new pimpl)
{
}

reconstructor::~reconstructor ()
{
}

void
reconstructor::operator () (std::string const &name, Mat const &frame)
{
  self->process_frame (name, frame);
}
