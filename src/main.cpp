#include "robot.h"

#include <cstdio>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "laplace.h"

template<int N>
static IplImage *transform (IplImage const *src);

template<>
IplImage *
transform<0> (IplImage const *src)
{
  IplImage *Fourier; //žµÀïÒ¶ÏµÊý
  IplImage *dst;
  IplImage *ImageRe;
  IplImage *ImageIm;
  IplImage *Image;
  IplImage *ImageDst;
  double m, M;
  double scale;
  double shift;

  Fourier = cvCreateImage (cvGetSize (src), IPL_DEPTH_64F, 2);
  dst = cvCreateImage (cvGetSize (src), IPL_DEPTH_64F, 2);
  ImageRe = cvCreateImage (cvGetSize (src), IPL_DEPTH_64F, 1);
  ImageIm = cvCreateImage (cvGetSize (src), IPL_DEPTH_64F, 1);
  Image = cvCreateImage (cvGetSize (src), src->depth, src->nChannels);
  ImageDst = cvCreateImage (cvGetSize (src), src->depth, src->nChannels);
  fft2 (src, Fourier);              //žµÀïÒ¶±ä»»
  fft2shift (Fourier, Image);       //ÖÐÐÄ»¯
  cvDFT (Fourier, dst, CV_DXT_INV_SCALE, 0); //ÊµÏÖžµÀïÒ¶Äæ±ä»»£¬²¢¶Ôœá¹ûœøÐÐËõ·Å
  cvSplit (dst, ImageRe, ImageIm, 0, 0);

  //¶ÔÊý×éÃ¿žöÔªËØÆœ·œ²¢ŽæŽ¢ÔÚµÚ¶þžö²ÎÊýÖÐ
  cvPow (ImageRe, ImageRe, 2);
  cvPow (ImageIm, ImageIm, 2);
  cvAdd (ImageRe, ImageIm, ImageRe, NULL);
  cvPow (ImageRe, ImageRe, 0.5);
  cvMinMaxLoc (ImageRe, &m, &M, NULL, NULL, NULL);
  scale = 255 / (M - m);
  shift = -m * scale;
  //œ«shiftŒÓÔÚImageRež÷ÔªËØ°Ž±ÈÀýËõ·ÅµÄœá¹ûÉÏ£¬ŽæŽ¢ÎªImageDst
  cvConvertScale (ImageRe, ImageDst, scale, shift);

  cvReleaseImage (&Image);
  cvReleaseImage (&ImageIm);
  cvReleaseImage (&ImageRe);
  cvReleaseImage (&Fourier);
  cvReleaseImage (&dst);

  return ImageDst;
}

static int C1 = 1;
static int C2 = 5;

static int c1 = 5;
static int c2 = 50;

static int H1 = 18;
static int H2 = 8;
static int H3 = 3;
static int H4 = 1;

static int h1 = 180;
static int h2 = 80;
static int h3 = 30;
static int h4 = 10;

template<>
IplImage *
transform<1> (IplImage const *src)
{
  IplImage *dst = cvCreateImage (cvGetSize (src), 8, 1);
  IplImage *color_dst = cvCreateImage (cvGetSize (src), 8, 3);

  CvMemStorage *storage = cvCreateMemStorage (0);

  printf ("cvCanny (src, dst, %d, %d, 3);\n", c1, c2);
  cvCanny (src, dst, c1, c2, 3);
  cvCvtColor (dst, color_dst, CV_GRAY2BGR);

  printf ("cvHoughLines2 (..., CV_PI / %d, %d, %d, %d);\n", h1, h2, h3, h4);
  CvSeq *lines = cvHoughLines2 (dst, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI / h1, h2, h3, h4);
  for (int i = 0; i < lines->total; i++)
    {
      CvPoint *line = (CvPoint *)cvGetSeqElem (lines, i);
      cvLine (color_dst, line[0], line[1], CV_RGB (255, 0, 0), 3, 8);
    }

  return color_dst;
}

int
main (int argc, char *argv[])
try
{
#if 0
  std::string hostname = "172.26.1.1";
  //std::string hostname = "172.26.1.2";

  if (argc > 1)
    hostname = argv[1];

  Robot robot (hostname);
  robot.run ();
#else
  cvNamedWindow ("source", 1);
  cvNamedWindow ("transformed", 1);

  bool paused = false;
  for (int i = 0; i <= 395; i++)
    {
      if (paused)
        i--;

      char filename[20];
      sprintf (filename, "images/%03d.jpg", i);

      IplImage *src = cvLoadImage (filename, 0);
      cvShowImage ("source", src);
      IplImage *dst = transform<1> (src);
      cvReleaseImage (&src);

      cvShowImage ("transformed", dst);
      cvReleaseImage (&dst);

      switch (cvWaitKey (100))
        {
        case 'q':
          return 0;
        case 'a':
          i--;
          break;
        case 'e':
          i++;
          break;

        case 'p': c1 -= C1; break;
        case 'i': c1 += C1; break;
        case 'y': c2 -= C2; break;
        case 'u': c2 += C2; break;

        case 'g': h1 -= H1; break;
        case 'd': h1 += H1; break;
        case 'c': h2 -= H2; break;
        case 'r': h2 += H2; break;
        case 't': h3 -= H3; break;
        case 'n': h3 += H3; break;
        case 'z': h4 -= H4; break;
        case 's': h4 += H4; break;

        case ' ':
          paused = !paused;
          break;
        }
    }
  cvDestroyWindow ("Live Image");
#endif

  return 0;
}
catch (cv::Exception const &e)
{
  puts (e.what ());
}
