#define NO_VIEW 1
#include "main.h"

#include "recon/reconstructor.h"

static reconstructor recon;

static void
process (std::string const &name, Mat const &src)
{
  recon (name, src);
}
