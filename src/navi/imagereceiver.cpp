#include <iostream>
#include <cstdio>

#include <opencv2/highgui/highgui.hpp>

#include "imagereceiver.h"

ImageReceiver::ImageReceiver ()
  : src (new cv::Mat)
{
  cv::namedWindow ("Live Image", 1);
  setStreaming (true);
  setResolution (640, 480);
}

ImageReceiver::~ImageReceiver ()
{
  cv::destroyWindow ("Live Image");
  delete src;
}

void
ImageReceiver::imageReceivedEvent (const unsigned char *data,
                                   unsigned int dataSize,
                                   unsigned int width,
                                   unsigned int height,
                                   unsigned int numChannels,
                                   unsigned int bitsPerChannel,
                                   unsigned int step)
{
  src->create (cv::Size (width, height), CV_8UC3);

  memcpy (src->ptr (), data, dataSize);

  puts ("Image received");
  imshow ("Live Image", *src);
}
