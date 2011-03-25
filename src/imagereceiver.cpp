#include <iostream>
#include <cstdio>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "imagereceiver.h"

ImageReceiver::ImageReceiver ()
  : src (NULL)
{
  cvNamedWindow ("Live Image", 1);
  setStreaming (true);
  setResolution (640, 480);
}

ImageReceiver::~ImageReceiver ()
{
  cvReleaseImage (&src);
  cvDestroyWindow ("Live Image");
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
  CvSize size = { width, height };

  if (src == NULL)
    src = cvCreateImage (size, IPL_DEPTH_8U, 3);

  memcpy (src->imageData, data, dataSize);

  puts ("Image received");
  cvShowImage ("Live Image", src);
}
