#pragma once

#include <rec/robotino/com/Camera.h>

struct _IplImage;

struct ImageReceiver
  : rec::robotino::com::Camera
{
  ImageReceiver ();
  ~ImageReceiver ();

  void imageReceivedEvent (const unsigned char *data, unsigned int dataSize, unsigned int width, unsigned int height, unsigned int numChannels, unsigned int bitsPerChannel, unsigned int step);

  _IplImage *src;
};
