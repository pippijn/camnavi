//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_IMAGE_IMAGEINFO_H_
#define _REC_CORE_LT_IMAGE_IMAGEINFO_H_

#include "rec/core_lt/defines.h"

#include <string>

namespace rec
{
	namespace core_lt
	{
		namespace image
		{
			typedef enum { Format_Undefined, Format_RGB, Format_BGR, Format_Gray, Format_YUV, Format_HSV, Format_YCbCr, Format_HLS, Format_Size } Format;

			class ImageInfo
			{
			public:
				ImageInfo()
					: width( 0 )
					, height( 0 )
					, numChannels( 0 )
					, bytesPerChannel( 0 )
					, format( Format_Undefined )
				{
				}

				ImageInfo( unsigned int width_, unsigned int height_, Format format_ = Format_RGB )
					: width( width_ )
					, height( height_ )
					, numChannels( 0 )
					, bytesPerChannel( 0 )
					, format( format_ )
				{
					init();
				}

				ImageInfo( unsigned int width_, unsigned int height_,
					         unsigned int numChannels_, unsigned int bytesPerChannel_,
									 Format format_ )
					: width( width_ )
					, height( height_ )
					, numChannels( numChannels_ )
					, bytesPerChannel( bytesPerChannel_ )
					, format( format_ )
				{
				}

				bool isNull() const
				{
					return ( 0 == width || height == 0 || numChannels == 0 || bytesPerChannel == 0 );
				}

				bool operator!=( const ImageInfo& other ) const
				{
					if( other.width != width ) return true;
					if( other.height != height ) return true;
					if( other.numChannels != numChannels ) return true;
					if( other.bytesPerChannel != bytesPerChannel ) return true;
					if( other.format != format ) return true;
					return false;
				}

				bool operator==( const ImageInfo& other ) const
				{
					return ( ! operator!=( other ) );
				}

				unsigned int width;
				unsigned int height;
				unsigned int numChannels;
				unsigned int bytesPerChannel;
				Format format;

			private:
				void init()
				{
					switch( format )
					{
					case Format_RGB:
					case Format_BGR:
					case Format_YUV:
					case Format_HSV:
					case Format_YCbCr:
					case Format_HLS:
						numChannels = 3;
						bytesPerChannel = 1;
						break;

					case Format_Gray:
						numChannels = 1;
						bytesPerChannel = 1;
						break;
					default:
						break;
					}
				}
			};

			REC_CORE_LT_EXPORT std::string friendlyName( Format format );
		}
	}
}

#endif //_REC_CORE_LT_IMAGE_IMAGEINFO_H_
