#include "sift.h"

#include <memory>

#include "siftgpu/SiftGPU.h"
#include "foreach.h"

static std::auto_ptr<SiftGPU> analyser;
static std::auto_ptr<SiftMatchGPU> matcher;

SiftGPU &
Sift::analyser ()
{
  if (::analyser.get ())
    return *::analyser;

  ::analyser.reset (new SiftGPU);
  char const *argv[] = {
    "-t", "0.01",
    "-v", "0",
    //"-cuda",
  };
  size_t argc = sizeof argv / sizeof *argv;
  ::analyser->ParseParam (argc, const_cast<char **> (argv));
  if (::analyser->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
    throw 0;

  return analyser ();
}

SiftMatchGPU &
Sift::matcher ()
{
  if (::matcher.get ())
    return *::matcher;

  ::matcher.reset (new SiftMatchGPU (4096));
  ::matcher->VerifyContextGL ();

  return matcher ();
}
