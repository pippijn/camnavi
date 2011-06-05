//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

/*this software is based in part on the work of
the Independent JPEG Group*/

#ifndef _REC_CORE_LT_IMAGE_BMP_HPP_
#define _REC_CORE_LT_IMAGE_BMP_HPP_

#include "rec/core_lt/image/Image.h"
#include "rec/core_lt/memory/ByteArrayConst.h"

namespace rec
{
	namespace core_lt
	{
		namespace image
		{
			unsigned int bmp_rowsize( unsigned int width, unsigned int numChannels, unsigned int* numFillBytes = NULL );

			rec::core_lt::image::Image bmp_load( const rec::core_lt::memory::ByteArrayConst& data );

			unsigned int bmp_calculate_header_size( const rec::core_lt::image::Image& image );
			unsigned int bmp_calculate_size( const rec::core_lt::image::Image& image );
			unsigned int bmp_create_header( unsigned char* pData, unsigned int bufferSize, const rec::core_lt::image::Image& image );
			rec::core_lt::memory::ByteArray bmp_save( const rec::core_lt::image::Image& image );
		}
	}
}

#endif //_REC_CORE_LT_IMAGE_BMP_HPP_
