#include "laplace.h"
#include <ctype.h>
#include <stdio.h>

//傅里叶正变换
void
fft2 (IplImage const *src, IplImage *dst)
{
  //实部、虚部
  IplImage *image_Re = 0, *image_Im = 0, *Fourier = 0;

  //   int i, j;
  image_Re = cvCreateImage (cvGetSize (src), IPL_DEPTH_64F, 1); //实部
  //Imaginary part
  image_Im = cvCreateImage (cvGetSize (src), IPL_DEPTH_64F, 1); //虚部
  //2 channels (image_Re, image_Im)
  Fourier = cvCreateImage (cvGetSize (src), IPL_DEPTH_64F, 2);
  // Real part conversion from u8 to 64f (double)
  cvConvertScale (src, image_Re, 1, 0);
  // Imaginary part (zeros)
  cvZero (image_Im);
  // Join real and imaginary parts and stock them in Fourier image
  cvMerge (image_Re, image_Im, 0, 0, Fourier);
  // Application of the forward Fourier transform
  cvDFT (Fourier, dst, CV_DXT_FORWARD, 0);
  cvReleaseImage (&image_Re);
  cvReleaseImage (&image_Im);
  cvReleaseImage (&Fourier);
}

/**************************************************************************
*  //src IPL_DEPTH_64F
*
*  //dst IPL_DEPTH_8U
*
**************************************************************************/
void
fft2shift (IplImage const *src, IplImage *dst)
{
  IplImage *image_Re = 0, *image_Im = 0;
  int nRow, nCol, i, j, cy, cx;
  double scale, shift, tmp13, tmp24;
  double minVal = 0, maxVal = 0;

  image_Re = cvCreateImage (cvGetSize (src), IPL_DEPTH_64F, 1);
  //Imaginary part
  image_Im = cvCreateImage (cvGetSize (src), IPL_DEPTH_64F, 1);
  cvSplit (src, image_Re, image_Im, 0, 0);
  //具体原理见冈萨雷斯数字图像处理p123
  // Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
  //计算傅里叶谱
  cvPow (image_Re, image_Re, 2.0);
  cvPow (image_Im, image_Im, 2.0);
  cvAdd (image_Re, image_Im, image_Re, NULL);
  cvPow (image_Re, image_Re, 0.5);
  //对数变换以增强灰度级细节(这种变换使以窄带低灰度输入图像值映射
  //一宽带输出值，具体可见冈萨雷斯数字图像处理p62)
  // Compute log(1 + Mag);
  cvAddS (image_Re, cvScalar (1.0, 0, 0, 0), image_Re, NULL); // 1 + Mag
  cvLog (image_Re, image_Re); // log(1 + Mag)

  //Rearrange the quadrants of Fourier image so that the origin is at the image center
  nRow = src->height;
  nCol = src->width;
  cy = nRow / 2; // image center
  cx = nCol / 2;
  //CV_IMAGE_ELEM为OpenCV定义的宏，用来读取图像的像素值，这一部分就是进行中心变换
  for (j = 0; j < cy; j++)
    for (i = 0; i < cx; i++)
      {
        //中心化，将整体份成四块进行对角交换
        tmp13 = CV_IMAGE_ELEM (image_Re, double, j, i);
        CV_IMAGE_ELEM (image_Re, double, j, i) = CV_IMAGE_ELEM (
          image_Re, double, j + cy, i + cx);
        CV_IMAGE_ELEM (image_Re, double, j + cy, i + cx) = tmp13;

        tmp24 = CV_IMAGE_ELEM (image_Re, double, j, i + cx);
        CV_IMAGE_ELEM (image_Re, double, j, i + cx) =
          CV_IMAGE_ELEM (image_Re, double, j + cy, i);
        CV_IMAGE_ELEM (image_Re, double, j + cy, i) = tmp24;
      }
  //归一化处理将矩阵的元素值归一为[0,255]
  //[(f(x,y)-minVal)/(maxVal-minVal)]*255
  //double minVal = 0, maxVal = 0;
  // Localize minimum and maximum values
  cvMinMaxLoc (image_Re, &minVal, &maxVal, NULL, NULL, NULL);
  // Normalize image (0 - 255) to be observed as an u8 image
  scale = 255 / (maxVal - minVal);
  shift = -minVal * scale;
  cvConvertScale (image_Re, dst, scale, shift);
  cvReleaseImage (&image_Re);
  cvReleaseImage (&image_Im);
}
