//  Copyright (C) 2004-2008, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_IMAGE_JPEG_H_
#define _REC_CORE_LT_IMAGE_JPEG_H_

#include "rec/core_lt/defines.h"

#include "rec/core_lt/image/Image.h"

namespace rec
{
	namespace core_lt
	{
		namespace image
		{
			// gets information about a jpg image
			REC_CORE_LT_EXPORT ImageInfo jpg_info( const rec::core_lt::memory::ByteArrayConst& data );

			// decompresses data to buffer
			REC_CORE_LT_EXPORT Image jpg_decompress( const rec::core_lt::memory::ByteArrayConst& data );

			// returns -1 on error, or jpeg size
			REC_CORE_LT_EXPORT rec::core_lt::memory::ByteArray jpg_compress( const Image& image, int quality = 75 );
		}
	}
}

#endif //_REC_CORE_LT_IMAGE_JPEG_H_

