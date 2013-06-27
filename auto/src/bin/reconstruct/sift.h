#pragma once

#include "siftgpu/SiftGPU.h"

struct Sift
{
  static SiftGPU &analyser ();
  static SiftMatchGPU &matcher ();
};
