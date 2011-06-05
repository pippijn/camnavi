#pragma once

#include <rec/robotino/com/Camera.h>

#include "cvfwd.h"

struct ImageReceiver
  : rec::robotino::com::Camera
{
  ImageReceiver ();
  ~ImageReceiver ();

  void imageReceivedEvent (const unsigned char *data, unsigned int dataSize, unsigned int width, unsigned int height, unsigned int numChannels, unsigned int bitsPerChannel, unsigned int step);

  cv::Mat *src;
};
