//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_IMAGE_OPENCV_H_
#define _REC_CORE_LT_IMAGE_OPENCV_H_

#include "rec/core_lt/image/Image.h"
#include <cv.h>

#define IPLIMAGE_FROM_REC_IMAGE( iplimg, img )\
	iplimg.nSize = sizeof( IplImage );\
	iplimg.nChannels = img.info().numChannels;\
	iplimg.depth = IPL_DEPTH_8U;\
	iplimg.dataOrder = 0;\
	iplimg.width = img.info().width;\
	iplimg.height = img.info().height;\
	iplimg.roi = NULL;\
	iplimg.maskROI = NULL;\
	iplimg.imageId = NULL;\
	iplimg.tileInfo = NULL;\
	iplimg.imageSize = img.info().height * img.step();\
	iplimg.widthStep = img.step();\
	iplimg.imageDataOrigin = iplimg.imageData;

namespace rec
{
	namespace core_lt
	{
		namespace image
		{
			class OpenCV
			{
			public:
				OpenCV( rec::core_lt::image::Image* img )
				{
					CvSize size;
					size.width = img->info().width;
					size.height = img->info().height;

					_iplImg = cvCreateImageHeader( size, IPL_DEPTH_8U, img->info().numChannels );

					_iplImg->imageSize = img->info().height * img->step();
					_iplImg->widthStep = img->step();

					_iplImg->imageData = img->data_s();
					_iplImg->imageDataOrigin = _iplImg->imageData;
				}

				OpenCV( const rec::core_lt::image::Image& img )
				{
					CvSize size;
					size.width = img.info().width;
					size.height = img.info().height;

					_iplImg = cvCreateImageHeader( size, IPL_DEPTH_8U, img.info().numChannels );

					_iplImg->imageSize = img.info().height * img.step();
					_iplImg->widthStep = img.step();

					_iplImg->imageData = const_cast<char*>( img.constData_s() );
					_iplImg->imageDataOrigin = _iplImg->imageData;
				}

				~OpenCV()
				{
					cvReleaseImageHeader( &_iplImg );
				}

				operator IplImage*()
				{
					return _iplImg;
				}

			private:
				IplImage* _iplImg;
			};
		}
	}
}

#endif
