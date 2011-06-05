//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifdef QT_CORE_LIB

#ifndef _REC_CORE_LT_IMAGE_QT_H_
#define _REC_CORE_LT_IMAGE_QT_H_

#include "rec/core_lt/image/Image.h"

#include <boost/cstdint.hpp>

#include <QImage>

namespace rec
{
	namespace core_lt
	{
		namespace image
		{
			static QImage toQImage( const Image& image )
			{
				QImage qImg( image.info().width, image.info().height, QImage::Format_RGB32 );
				unsigned int dstStep = qImg.bytesPerLine();
				unsigned char* dstData = (unsigned char*)qImg.bits();

				const unsigned char* srcData = image.constData();
				unsigned int srcStep = image.step();

				if( rec::core_lt::image::Format_Gray == image.info().format )
				{
					unsigned int x;
					unsigned int y;
					boost::uint32_t* data = reinterpret_cast< boost::uint32_t* >( dstData );
					for( y = 0; y < image.info().height; ++y )
					{
						unsigned char* lineDst = reinterpret_cast< unsigned char* >( data );
						const unsigned char* lineSrc = srcData;
						for( x = 0; x < image.info().width; ++x )
						{
							*data = (*srcData << 16) | (*srcData << 8) | (*srcData) | 0xFF000000;
							++data;
							++srcData;
						}
						data = reinterpret_cast< boost::uint32_t* >(lineDst + dstStep);
						srcData = lineSrc + srcStep;
					} 
				}
				else
				{
					unsigned int x;
					unsigned int y;
					boost::uint32_t* data = reinterpret_cast< boost::uint32_t* >( dstData );
					for( y = 0; y < image.info().height; ++y )
					{
						unsigned char* lineDst = reinterpret_cast< unsigned char* >( data );
						const unsigned char* lineSrc = srcData;
						for( x = 0; x < image.info().width; ++x )
						{
							*data = ( srcData[2] << 16) | ( srcData[1] << 8) | ( srcData[0] ) | 0xFF000000;
							++data;
							srcData += 3;
						}
						data = reinterpret_cast< boost::uint32_t* >(lineDst + dstStep);
						srcData = lineSrc + srcStep;
					}
				}

				return qImg;
			}
		}
	}
}

#endif //_REC_CORE_LT_IMAGE_QT_H_

#endif //QT_CORE_LIB
